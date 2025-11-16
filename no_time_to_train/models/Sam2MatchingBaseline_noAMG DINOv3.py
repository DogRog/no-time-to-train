import copy
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torchvision.ops.boxes import batched_nms
from torchvision.transforms import Normalize
from transformers import AutoModel, pipeline
from huggingface_hub import snapshot_download

from sam2.build_sam import build_sam2_video_predictor
from sam2.utils.amg import batched_mask_to_box

from no_time_to_train.models.matching_baseline_utils import (kmeans,
                                                             vis_kmeans,
                                                             vis_pca)
from no_time_to_train.models.model_utils import concat_all_gather


PRINT_TIMING = False

encoder_predefined_cfgs = {
    "dinov2_large": dict(
        img_size=518,
        patch_size=14,
        init_values=1e-5,
        ffn_layer='mlp',
        block_chunks=0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        feat_dim=1024,
        hf_model_name="facebook/dinov2-large"
    ),
    "dinov3_vitl16": dict(
        img_size=518,
        patch_size=16,
        init_values=1e-5,
        ffn_layer='mlp',
        block_chunks=0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        feat_dim=1024,
        hf_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m"
    )
}



class Sam2MatchingBaselineNoAMG(nn.Module):
    def __init__(
        self,
        sam2_cfg_file,
        sam2_ckpt_path,
        sam2_infer_cfgs,
        encoder_cfg,
        encoder_ckpt_path,
        memory_bank_cfg,
        dataset_name='coco',
        dataset_imgs_path=None,
        class_names=None,
        online_vis=False,
        vis_thr=0.5
    ):
        super(Sam2MatchingBaselineNoAMG, self).__init__()

        self.dataset_name = dataset_name
        self.class_names = class_names
        self.dataset_imgs_path = dataset_imgs_path
        self.online_vis = online_vis
        self.vis_thr = vis_thr
        self.points_per_side = sam2_infer_cfgs.get("points_per_side")
        self.testing_point_bs = sam2_infer_cfgs.get("testing_point_bs")
        self.iou_thr = sam2_infer_cfgs.get("iou_thr")
        self.num_out_instance = sam2_infer_cfgs.get("num_out_instance")
        self.nms_thr = sam2_infer_cfgs.get("nms_thr")
        self.kmeans_k = sam2_infer_cfgs.get("kmeans_k")
        self.n_pca_components = sam2_infer_cfgs.get("n_pca_components")
        self.cls_num_per_mask = sam2_infer_cfgs.get("cls_num_per_mask")

        self.with_negative_refs = sam2_infer_cfgs.get("with_negative_refs", False)

        # Models
        self.sam_transform = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.predictor = build_sam2_video_predictor(sam2_cfg_file, sam2_ckpt_path)
        self.sam_img_size = 1024

        encoder_cfg = copy.deepcopy(encoder_cfg)
        encoder_name = encoder_cfg.pop("name")
        encoder_defaults = copy.deepcopy(encoder_predefined_cfgs.get(encoder_name, {}))

        if not encoder_defaults:
            raise KeyError(f"Unsupported encoder preset '{encoder_name}'.")

        encoder_img_size = encoder_cfg.get("img_size", encoder_defaults.get("img_size"))
        encoder_patch_size = encoder_cfg.get("patch_size", encoder_defaults.get("patch_size"))
        encoder_hw = encoder_img_size // encoder_patch_size

        self.encoder_h, self.encoder_w = encoder_hw, encoder_hw
        self.encoder_img_size = encoder_img_size
        self.encoder_patch_size = encoder_patch_size
        hf_model_id = encoder_cfg.pop("hf_model_name", None)
        if hf_model_id is None:
            if encoder_ckpt_path is not None:
                hf_model_id = encoder_ckpt_path
            else:
                hf_model_id = encoder_defaults.get("hf_model_name")

        self.encoder_transform = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        if hf_model_id is None:
            raise ValueError(
                "A Hugging Face vision model repository ID or local directory (hf_model_name/encoder_ckpt_path) must be provided."
            )

        hf_model_id = self._resolve_encoder_checkpoint(hf_model_id, encoder_defaults)

        try:
            # Use a Transformers pipeline to initialize and cache the encoder weights.
            feature_pipeline = pipeline(
                task="feature-extraction",
                model=hf_model_id,
            )
            self.encoder = feature_pipeline.model
        except OSError as exc:
            raise RuntimeError(
                f"Failed to load vision model from '{hf_model_id}'. Ensure it is a Hugging Face repository ID or"
                " a directory containing a compatible Transformers checkpoint."
            ) from exc
        except Exception as exc:
            try:
                # Fall back to the generic auto model loader if pipeline initialisation fails for non-IO reasons.
                self.encoder = AutoModel.from_pretrained(hf_model_id)
            except Exception as auto_exc:  # pragma: no cover - network dependent
                raise RuntimeError(
                    f"Failed to initialise vision encoder via pipeline or AutoModel for '{hf_model_id}'."
                ) from auto_exc

        self.encoder_dim = self.encoder.config.hidden_size
        self.encoder.to(self.predictor.device)

        self.predictor.eval()
        self.encoder.eval()

        # Others
        memory_bank_cfg["feat_shape"] = (self.encoder_h * self.encoder_w, self.encoder_dim)
        self._init_memory_bank(memory_bank_cfg)

        self._reset()

    def _init_memory_bank(self, memory_bank_cfg):
        assert memory_bank_cfg.pop("enable")

        self.mem_n_classes = memory_bank_cfg.get("category_num")
        self.mem_length = memory_bank_cfg.get("length")
        self.mem_feat_shape = memory_bank_cfg.get("feat_shape")

        assert len(self.mem_feat_shape) == 2
        _mem_n, _mem_c = self.mem_feat_shape

        self.register_buffer(
            "mem_fill_counts", torch.zeros((self.mem_n_classes,), dtype=torch.long)
        )
        self.register_buffer(
            "mem_feats", torch.zeros((self.mem_n_classes, self.mem_length, _mem_n, _mem_c))
        )
        self.register_buffer(
            "mem_masks", torch.zeros((self.mem_n_classes, self.mem_length, _mem_n))
        )
        self.register_buffer(
            "mem_feats_avg", torch.zeros((self.mem_n_classes, _mem_c))
        )
        self.register_buffer(
            "mem_feats_ins_avg", torch.zeros((self.mem_n_classes, self.mem_length, _mem_c))
        )
        self.register_buffer(
            "mem_feats_covariances", torch.zeros((self.mem_n_classes, _mem_c, _mem_c))
        )
        self.register_buffer(
            "mem_feats_centers", torch.zeros((self.mem_n_classes, self.kmeans_k, _mem_c))
        )
        self.register_buffer(
            "mem_ins_sim_avg", torch.zeros((self.mem_n_classes,))
        )
        self.register_buffer(
            "mem_pca_mean", torch.zeros((self.mem_n_classes, _mem_c))
        )
        self.register_buffer(
            "mem_pca_components", torch.zeros((self.mem_n_classes, self.n_pca_components, _mem_c))
        )
        self.register_buffer("mem_postprocessed", torch.zeros((1,), dtype=torch.bool))
        self.memory_ready = False

        if self.with_negative_refs:
            self.mem_length_negative = memory_bank_cfg.get("length_negative")
            self.register_buffer(
                "mem_fill_counts_neg", torch.zeros((self.mem_n_classes,), dtype=torch.long)
            )
            self.register_buffer(
                "mem_feats_neg", torch.zeros((self.mem_n_classes, self.mem_length_negative, _mem_n, _mem_c))
            )
            self.register_buffer(
                "mem_masks_neg", torch.zeros((self.mem_n_classes, self.mem_length_negative, _mem_n))
            )
            self.register_buffer(
                "mem_feats_avg_neg", torch.zeros((self.mem_n_classes, _mem_c))
            )
            self.register_buffer(
                "mem_feats_ins_avg_neg", torch.zeros((self.mem_n_classes, self.mem_length_negative, _mem_c))
            )
            self.register_buffer("mem_postprocessed_neg", torch.zeros((1,), dtype=torch.bool))
            self.memory_neg_ready = False

    def _reset(self):
        self.backbone_features = None
        self.backbone_hr_features = None

    def _compute_semantic_ios(self, masks_binary, labels, obj_sim, use_semantic=True, rank_score=True):
        n_masks = masks_binary.shape[0]
        masks = masks_binary.reshape(n_masks, -1).to(dtype=torch.float32)
        ios = torch.zeros((n_masks,), device=masks_binary.device, dtype=torch.float32)

        for cat_ind in range(self.mem_n_classes):
            select_idxs = (labels == cat_ind)
            _masks = masks[select_idxs]
            _obj_sim = obj_sim[select_idxs][:, select_idxs]
            n_cat = _masks.shape[0]
            if n_cat == 0:
                continue
            pos_num = _masks.sum(dim=-1).to(dtype=torch.float32)
            inter_num = _masks @ _masks.t()
            inter_num.fill_diagonal_(0.0)
            if rank_score:
                inter_num = torch.tril(inter_num, diagonal=0)
            _ios = (inter_num / pos_num[:, None])
            if use_semantic:
                _ios = _ios * _obj_sim
            _ios = _ios.max(dim=-1)[0]
            ios[select_idxs] += _ios
        return ios

    def _forward_encoder(self, imgs):
        assert len(imgs.shape) == 4
        n_skip_tokens = 1 + getattr(self.encoder.config, "num_register_tokens", 0)
        outputs = self.encoder(pixel_values=imgs, output_hidden_states=False)
        seq_feats = outputs.last_hidden_state
        feats = seq_feats[:, n_skip_tokens:, :]
        feats = feats.reshape(imgs.shape[0], -1, self.encoder_dim)
        return feats

    def _resolve_encoder_checkpoint(self, hf_model_id: str, encoder_defaults: dict) -> str:
        """Ensure the encoder checkpoint is a valid Hugging Face repo id or local directory.

        If a legacy *.pth path is provided, download the corresponding repo and return the local directory.
        """

        # Accept already-downloaded directories or valid repo ids directly.
        if os.path.isdir(hf_model_id):
            return hf_model_id

        # If a legacy checkpoint file path is supplied, download the HF model next to it.
        if os.path.isfile(hf_model_id):
            repo_id = encoder_defaults.get("hf_model_name")
            if repo_id is None:
                raise ValueError(
                    "Legacy checkpoint path provided but no default Hugging Face repo id is available for download."
                )

            target_dir = os.path.splitext(hf_model_id)[0] + "_hf"
            if not os.path.isdir(target_dir) or not os.listdir(target_dir):
                try:
                    snapshot_download(
                        repo_id=repo_id,
                        local_dir=target_dir,
                        local_dir_use_symlinks=False,
                    )
                except Exception as exc:  # pragma: no cover - network dependent
                    raise RuntimeError(
                        "Failed to download Hugging Face checkpoint. Please ensure network access or manually "
                        f"download '{repo_id}' into '{target_dir}'."
                    ) from exc
            return target_dir

        # Otherwise assume it's a repo id string understood by Transformers.
        return hf_model_id

    def _forward_sam_decoder(
        self,
        backbone_features,
        sparse_embeddings,
        dense_embeddings,
        backbone_hr_features,
        multimask_output=True
    ):
        B = backbone_features.shape[0]
        device = backbone_features.device
        (
            low_res_multimasks,
            ious,
            _,
            _,
        ) = self.predictor.sam_mask_decoder(
            image_embeddings=backbone_features,
            image_pe=self.predictor.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=False,  # the image is already batched
            high_res_features=backbone_hr_features,
            return_iou_token_out=False,
            disable_custom_iou_embed=True,
            disable_mlp_obj_scores=True,
            output_all_masks=True,
        )

        n_pred = ious.shape[-1]
        assert n_pred == low_res_multimasks.shape[1]

        # We skip the SAM2's multimask_output but use the custom IoU to determine the output mask
        if multimask_output:
            best_iou_inds = torch.argmax(ious[:, 1:], dim=-1) + 1
            batch_inds = torch.arange(B, device=device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds]
            scores = ious[batch_inds, best_iou_inds]
        else:
            low_res_masks = low_res_multimasks[:, 0]
            scores = ious[:, 0]
        return low_res_masks, scores

    def _compute_masks(
        self,
        backbone_features,
        backbone_hr_features,
        point_inputs
    ):
        '''
        Similar to SAM2Base._forward_sam_heads. Putting it here for easy customization
        '''
        B = backbone_features.size(0)
        assert backbone_features.size(1) == self.predictor.sam_prompt_embed_dim
        assert backbone_features.size(2) == self.predictor.sam_image_embedding_size
        assert backbone_features.size(3) == self.predictor.sam_image_embedding_size

        sam_point_coords = point_inputs["point_coords"]
        sam_point_labels = point_inputs["point_labels"]
        assert sam_point_coords.size(0) == B and sam_point_labels.size(0) == B

        sparse_embeddings, dense_embeddings = self.predictor.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels),
            boxes=None,
            masks=None,
        )
        low_res_masks, scores = self._forward_sam_decoder(
            backbone_features,
            sparse_embeddings,
            dense_embeddings,
            backbone_hr_features,
            multimask_output=True
        )
        return low_res_masks, scores

    def _forward_sam(self, imgs, precomputed_points=None, point_normed=True):
        assert len(imgs.shape) == 4
        assert imgs.shape[-2] == imgs.shape[-1]
        assert self.backbone_features is None
        assert self.backbone_hr_features is None

        device = imgs.device

        sam_input_size = imgs.shape[-2]
        points_per_side = self.points_per_side
        testing_point_bs = self.testing_point_bs
        iou_thr = self.iou_thr

        # Prepare input
        if precomputed_points is None:
            x, y = torch.meshgrid(
                torch.linspace(0, sam_input_size-1, points_per_side),
                torch.linspace(0, sam_input_size-1, points_per_side),
                indexing='ij'
            )
            query_points = torch.stack((y.reshape(-1), x.reshape(-1)), dim=-1)
            query_points += 0.5
            query_points = query_points.to(device=device)
        else:
            if point_normed:
                query_points = precomputed_points * sam_input_size
            else:
                query_points = precomputed_points


        # forward model
        backbone_out = self.predictor.forward_image(imgs)
        _, img_vision_features, img_vision_pos_embeds, img_feat_sizes = (
            self.predictor._prepare_backbone_features(backbone_out)
        )

        img_feats = img_vision_features[-1].permute(1, 2, 0).reshape(1, -1, *img_feat_sizes[-1])
        self.backbone_features = img_feats
        img_feats = img_feats.expand(testing_point_bs, -1, -1, -1)

        hr_feats = [
            x.permute(1, 2, 0).reshape(1, -1, *s)
            for x, s in zip(img_vision_features[:-1], img_feat_sizes[:-1])
        ]
        self.backbone_hr_features = hr_feats
        hr_feats = [
            x.expand(testing_point_bs, -1, -1, -1) for x in hr_feats
        ]

        points = query_points.reshape(-1, 2)
        point_labels = torch.ones_like(points[:, 0:1]).to(dtype=torch.int32)
        n_points = points.shape[0]

        mask_scores = []
        lr_masks = []
        for i in range(0, n_points // testing_point_bs):
            i_start = i * testing_point_bs
            i_end = i_start + testing_point_bs
            points_i = points[i_start:i_end, :]
            p_labels_i = point_labels[i_start:i_end, :]
            point_inputs_i = dict(
                point_coords=points_i.reshape(testing_point_bs, 1, 2),
                point_labels=p_labels_i.reshape(testing_point_bs, 1)
            )
            lr_masks_i, scores_i = self._compute_masks(
                img_feats, hr_feats, point_inputs_i
            )
            mask_scores.append(scores_i.reshape(-1))
            lr_masks.append(lr_masks_i.reshape(-1, *lr_masks_i.shape[-2:]))
        scores_all = torch.cat(mask_scores, dim=0).reshape(-1)
        lr_masks_all = torch.cat(lr_masks, dim=0)
        lr_masks_all = lr_masks_all.reshape(-1, *lr_masks_all.shape[-2:])

        inds = scores_all > iou_thr
        points_all = points[inds]
        lr_masks_all = lr_masks_all[inds]
        scores_all = scores_all[inds]

        return lr_masks_all, scores_all, points_all

    def forward_fill_memory(self, input_dicts, is_positive):
        with torch.inference_mode():
            assert len(input_dicts) == 1

            device = self.predictor.device

            ref_cat_ind = list(input_dicts[0]["refs_by_cat"].keys())[0]

            ref_imgs = input_dicts[0]["refs_by_cat"][ref_cat_ind]["imgs"].to(device=device)
            ref_masks = input_dicts[0]["refs_by_cat"][ref_cat_ind]["masks"].to(dtype=ref_imgs.dtype)

            ref_imgs = F.interpolate(
                ref_imgs,
                size=(self.encoder_img_size, self.encoder_img_size),
                mode="bicubic"
            )
            ref_imgs = self.encoder_transform(ref_imgs)
            ref_feats = self._forward_encoder(ref_imgs)
            ref_feats = ref_feats.reshape(1, -1, self.encoder_dim)

            ref_masks = F.interpolate(
                ref_masks.unsqueeze(dim=0),
                size=(self.encoder_h, self.encoder_w),
                mode="nearest"
            ).reshape(1, -1)

            cat_ind_tensor = torch.tensor([ref_cat_ind], dtype=torch.long, device=device).reshape(1, 1)
            cat_ind_all = concat_all_gather(cat_ind_tensor).reshape(-1).to(dtype=torch.long).detach()
            feats_all = concat_all_gather(ref_feats.contiguous())
            masks_all = concat_all_gather(ref_masks.contiguous())

            for i in range(cat_ind_all.shape[0]):
                if is_positive:
                    if dist.is_initialized():
                        assert (self.mem_n_classes * self.mem_length) % dist.get_world_size() == 0
                    fill_ind = self.mem_fill_counts[cat_ind_all[i]]
                    self.mem_feats[cat_ind_all[i], fill_ind] += feats_all[i]
                    self.mem_masks[cat_ind_all[i], fill_ind] += masks_all[i]
                    self.mem_fill_counts[cat_ind_all[i]] += 1
                else:
                    if dist.is_initialized():
                        assert (self.mem_n_classes * self.mem_length_negative) % dist.get_world_size() == 0
                    fill_ind = self.mem_fill_counts_neg[cat_ind_all[i]]
                    self.mem_feats_neg[cat_ind_all[i], fill_ind] += feats_all[i]
                    self.mem_masks_neg[cat_ind_all[i], fill_ind] += masks_all[i]
                    self.mem_fill_counts_neg[cat_ind_all[i]] += 1

            return {}

    def forward_vis_memory(self, input_dicts):
        assert len(input_dicts) == 1
        assert self.mem_fill_counts[0].item() > 0
        assert self.n_pca_components == 3  # RGB

        device = self.predictor.device
        output_dir = "./results_analysis/memory_vis"

        ref_cat_ind = list(input_dicts[0]["refs_by_cat"].keys())[0]

        ref_imgs = input_dicts[0]["refs_by_cat"][ref_cat_ind]["imgs"].to(device=device)
        ref_imgs = F.interpolate(
            ref_imgs,
            size=(self.encoder_img_size, self.encoder_img_size),
            mode="bicubic"
        )
        ref_imgs_normed = self.encoder_transform(ref_imgs)
        ref_feats = self._forward_encoder(ref_imgs_normed)
        ref_feats = ref_feats.reshape(-1, self.encoder_dim)

        ref_masks_ori = input_dicts[0]["refs_by_cat"][ref_cat_ind]["masks"].to(dtype=ref_feats.dtype, device=device)
        ref_masks = F.interpolate(
            ref_masks_ori.unsqueeze(dim=0),
            size=(self.encoder_h, self.encoder_w),
            mode="nearest"
        ).reshape(-1)

        encoder_shape_info = dict(
            height=self.encoder_h,
            width=self.encoder_w,
            patch_size=self.encoder_patch_size
        )

        pca_vis_result = vis_pca(
            ref_imgs,
            ref_masks_ori,
            ref_cat_ind,
            ref_feats,
            ref_masks,
            self.mem_pca_mean,
            self.mem_pca_components,
            encoder_shape_info,
            device,
            transparency=1.0
        )
        kmeans_vis_result = vis_kmeans(
            ref_imgs,
            ref_masks_ori,
            ref_cat_ind,
            ref_feats,
            ref_masks,
            self.mem_feats_centers,
            encoder_shape_info,
            device,
            transparency=1.0
        )
        ori_img = ref_imgs[0].permute(1, 2, 0) * 255.0
        margin = torch.zeros((ori_img.shape[0], 5, 3), dtype=ori_img.dtype, device=device) + 255
        output_final = torch.cat((
            ori_img, margin, kmeans_vis_result, margin, pca_vis_result
        ), dim=1)

        import os

        from PIL import Image

        out_vis_img = Image.fromarray(output_final.cpu().numpy().astype(np.uint8))
        img_id = int(input_dicts[0]["refs_by_cat"][ref_cat_ind]["img_info"][0]['id'])
        out_vis_img.save(os.path.join(output_dir, "%d_%d.png" % (ref_cat_ind, img_id)))
        return {}

    def _compute_sim_global_avg(self, tar_feat, masks_feat_size_bool, softmax=False, temp=1.0, ret_feats=False):
        masks = masks_feat_size_bool.to(dtype=tar_feat.dtype)
        tar_avg_feats = (masks @ tar_feat) / masks.sum(dim=-1, keepdim=True)
        tar_avg_feats = F.normalize(tar_avg_feats, p=2, dim=-1)

        mem_feats_avg = self.mem_feats_ins_avg.mean(dim=1)
        mem_feats_avg = F.normalize(mem_feats_avg, p=2, dim=-1)

        sim_avg = tar_avg_feats @ mem_feats_avg.t()
        if softmax:
            sim_avg = torch.softmax(sim_avg / temp, dim=-1)
        else:
            sim_avg = sim_avg.clamp(min=0.0)
        if not ret_feats:
            return sim_avg
        else:
            return sim_avg, tar_avg_feats

    def _compute_sim_global_avg_with_neg(self, tar_feat, masks_feat_size_bool, sigma=1.0):
        n_masks = masks_feat_size_bool.shape[0]
        c = tar_feat.shape[-1]

        masks = masks_feat_size_bool.to(dtype=tar_feat.dtype)
        tar_avg_feats = (masks @ tar_feat) / masks.sum(dim=-1, keepdim=True)
        tar_avg_feats = F.normalize(tar_avg_feats, p=2, dim=-1)

        mem_feats_avg = self.mem_feats_avg
        mem_feats_avg = F.normalize(mem_feats_avg, p=2, dim=-1)

        mem_feats_ins_avg_neg = self.mem_feats_ins_avg_neg
        mem_feats_ins_avg_neg = F.normalize(mem_feats_ins_avg_neg, p=2, dim=-1).reshape(-1, c)

        sim_pos = tar_avg_feats @ mem_feats_avg.t()
        sim_pos = sim_pos.clamp(min=0.0)

        sim_neg = tar_avg_feats @ mem_feats_ins_avg_neg.t()
        sim_neg = sim_neg.clamp(min=0.0)
        sim_neg = sim_neg.reshape(n_masks, self.mem_n_classes, -1)
        sim_neg, max_inds = sim_neg.max(dim=-1)

        sim_final = sim_pos * torch.exp(-1.0 * (sim_neg - sim_pos).clamp(min=0.0) / sigma)
        return sim_final

    def forward_test(self, input_dicts, with_negative):

        if PRINT_TIMING:
            start_time = time.time()

        assert len(input_dicts) == 1

        device = self.predictor.device

        tar_img = input_dicts[0]["target_img"].to(device=device)
        tar_img_encoder = F.interpolate(
            tar_img.unsqueeze(dim=0),
            size=(self.encoder_img_size, self.encoder_img_size),
            mode="bicubic"
        )
        tar_feat = self._forward_encoder(self.encoder_transform(tar_img_encoder))
        tar_feat = tar_feat.reshape(-1, self.encoder_dim)  # [N, C]

        # SAM inference
        tar_img = tar_img.unsqueeze(dim=0)
        lr_masks, pred_ious, query_points = self._forward_sam(self.sam_transform(tar_img))

        n_masks = lr_masks.shape[0]
        masks_feat_size_bool = lr_masks > 0
        masks_feat_size_bool = masks_feat_size_bool.reshape(n_masks, -1)
        tar_feat = tar_feat.reshape(1, self.encoder_h, self.encoder_w, -1).permute(0, 3, 1, 2)
        tar_feat = F.interpolate(
            tar_feat,
            size=tuple(lr_masks.shape[-2:]),
            mode="bilinear",
            align_corners=False,
            antialias=True
        ).reshape(-1, lr_masks.shape[-2] * lr_masks.shape[-1]).t()

        if not with_negative:
            if PRINT_TIMING:
                start_time_sim_global = time.time()
            sim_global, obj_feats = self._compute_sim_global_avg(tar_feat, masks_feat_size_bool, ret_feats=True)
            if PRINT_TIMING:
                end_time_sim_global = time.time()
                print("--------------------------------")
                print("TIMING SIM GLOBAL: ", end_time_sim_global - start_time_sim_global)
                print("--------------------------------")
        else:
            assert self.with_negative_refs
            assert self.memory_neg_ready
            sim_global = self._compute_sim_global_avg_with_neg(tar_feat, masks_feat_size_bool, sigma=0.8)

        merged_scores = sim_global

        if self.cls_num_per_mask == -1:
            self.cls_num_per_mask = self.mem_n_classes
        top_scores, labels = torch.topk(merged_scores, k=self.cls_num_per_mask)

        if self.cls_num_per_mask == self.mem_n_classes:
            max_scores = top_scores[:, 0:1]
            top_scores = top_scores * (top_scores > (max_scores * 0.6))

        labels = labels.flatten()
        scores_all_class = top_scores.flatten()

        lr_bboxes = batched_mask_to_box(lr_masks > 0)
        lr_bboxes_expand = (
            lr_bboxes.unsqueeze(dim=1)
            .expand(-1, self.cls_num_per_mask, -1)
            .reshape(n_masks * self.cls_num_per_mask, 4)
        )

        expand_ratio = 8
        out_num = int(min(self.num_out_instance * expand_ratio, labels.shape[0]))

        nms_keep_inds = batched_nms(
            lr_bboxes_expand.float(),
            pred_ious.flatten(),
            labels,
            iou_threshold=self.nms_thr
        )[:out_num]
        scores_out = scores_all_class[nms_keep_inds]
        lr_masks_out = lr_masks[nms_keep_inds // self.cls_num_per_mask]
        obj_feats_out = obj_feats[nms_keep_inds // self.cls_num_per_mask]
        labels_out = labels[nms_keep_inds]

        pos_inds = scores_out > 0.0
        scores_out = scores_out[pos_inds]
        lr_masks_out = lr_masks_out[pos_inds]
        obj_feats_out = obj_feats_out[pos_inds]
        labels_out = labels_out[pos_inds]

        # resizing and converting to output format
        ori_h = input_dicts[0]["target_img_info"]["ori_height"]
        ori_w = input_dicts[0]["target_img_info"]["ori_width"]


        if lr_masks_out.shape[0] == 0:
            self._reset()
            return [{
                "binary_masks": torch.zeros((0, ori_h, ori_w), device=device).bool(),
                "bboxes": torch.zeros((0, 4), device=device),
                "scores": torch.zeros((0,), device=device),
                "labels": torch.zeros((0,), dtype=torch.long, device=device),
                "image_info": input_dicts[0]["target_img_info"],
            }]

        masks_out_binary = F.interpolate(
            lr_masks_out.unsqueeze(dim=1),
            size=(ori_h, ori_w),
            mode="bilinear",
            align_corners=False,
            antialias=True
        ).squeeze(dim=1) > 0

        bboxes = batched_mask_to_box(masks_out_binary)

        if PRINT_TIMING:
            start_time_merging = time.time()
        obj_sim = obj_feats_out @ obj_feats_out.t()
        obj_sim = obj_sim.clamp(min=0.0)
        ios = self._compute_semantic_ios(masks_out_binary, labels_out, obj_sim, use_semantic=True, rank_score=True)
        score_decay = 1 - ios
        scores_out = scores_out * torch.pow(score_decay, 0.5)
        if PRINT_TIMING:
            end_time_merging = time.time()
            print("--------------------------------")
            print("TIMING MERGING: ", end_time_merging - start_time_merging)
            print("--------------------------------")

        final_out_num = min(self.num_out_instance, scores_out.shape[0])
        final_out_inds = torch.argsort(scores_out, descending=True)[:final_out_num]

        masks_out_binary = masks_out_binary[final_out_inds]
        bboxes = bboxes[final_out_inds]
        scores_out = scores_out[final_out_inds]
        labels_out = labels_out[final_out_inds]

        output_dict = dict(
            binary_masks=masks_out_binary,
            bboxes=bboxes,
            scores=scores_out,
            labels=labels_out,
            image_info=input_dicts[0]["target_img_info"],
        )

        if self.online_vis:
            self._vis_results_online(output_dict, input_dicts[0]["tar_anns_by_cat"],
                                    score_thr=self.vis_thr,
                                    show_scores=True,
                                    dataset_name=self.dataset_name,
                                    dataset_imgs_path=self.dataset_imgs_path,
                                    class_names=self.class_names)
        self._reset()

        # Calculate and print timing statistics
        if PRINT_TIMING:
            end_time = time.time()
            total_time = end_time - start_time
            print(f"\n===== TIMING FORWARD TEST RESULTS =====")
            print(f"Total processing time: {total_time:.2f} seconds")
            print(f"===========================\n")

        return [output_dict]

    def postprocess_memory(self):
        if PRINT_TIMING:
            start_time = time.time()
        # Compute class-wise average features
        device = self.mem_feats_avg.device
        c = self.mem_feats.shape[-1]

        self.mem_feats_avg *= 0.0
        mem_feats_avg = (
            torch.sum(self.mem_feats * self.mem_masks.unsqueeze(dim=-1), dim=(1, 2))
            / self.mem_masks.sum(dim=(1, 2)).unsqueeze(dim=1)
        )
        self.mem_feats_avg += mem_feats_avg

        mem_feats_ins_avg = (
            torch.sum(self.mem_feats * self.mem_masks.unsqueeze(dim=-1), dim=2)
            / self.mem_masks.sum(dim=2).unsqueeze(dim=2)
        )
        self.mem_feats_ins_avg += mem_feats_ins_avg

        sigmas = []
        for i in range(self.mem_n_classes):
            feats_i = self.mem_feats_ins_avg[i].reshape(-1, c)
            mu_i = feats_i.mean(dim=0, keepdim=True)
            feats_i_centered = feats_i - mu_i

            sigma_i = feats_i_centered.t() @ feats_i_centered / float(feats_i_centered.shape[0])
            sigmas.append(sigma_i.unsqueeze(dim=0))
        sigmas = torch.cat(sigmas, dim=0)
        self.mem_feats_covariances += sigmas

        ins_sims = []
        for i in range(self.mem_n_classes):
            sims_i = []
            for j in range(self.mem_length):
                feat_ij = self.mem_feats[i, j]
                feats_i_rest = torch.cat((self.mem_feats[i, :j], self.mem_feats[i, j+1:]), dim=0)
                mask_ij = self.mem_masks[i, j]
                mask_i_rest = torch.cat((self.mem_masks[i, :j], self.mem_masks[i, j+1:]), dim=0)

                feats_ij = F.normalize(feat_ij[mask_ij > 0].mean(dim=0, keepdim=True), p=2, dim=1)
                feats_i_rest = F.normalize(feats_i_rest[mask_i_rest > 0].mean(dim=0, keepdim=True), p=2, dim=-1)
                sim_ij = feats_ij @ feats_i_rest.t()
                sim_ij = (sim_ij + 1.0) * 0.5
                sims_i.append(sim_ij)
            sim_i = torch.stack(sims_i).mean()
            ins_sims.append(sim_i)
        ins_sims = torch.stack(ins_sims).reshape(self.mem_n_classes)
        self.mem_ins_sim_avg += ins_sims

        # K-means
        kmeans_iters = 100
        for i in range(self.mem_n_classes):
            feats = self.mem_feats[i].reshape(-1, self.encoder_dim)[self.mem_masks[i].reshape(-1) > 0]
            assert feats.shape[0] > 0
            centers_i = kmeans(feats, self.kmeans_k, kmeans_iters)
            self.mem_feats_centers[i] += centers_i

        # PCA
        for i in range(self.mem_n_classes):
            feats = self.mem_feats[i].reshape(-1, self.encoder_dim)[self.mem_masks[i].reshape(-1) > 0]
            assert feats.shape[0] > 0
            feats = feats.cpu().numpy()
            pca = PCA(n_components=self.n_pca_components)
            pca.fit(feats)
            pca_mean = torch.from_numpy(pca.mean_).to(device=device)
            pca_components = torch.from_numpy(pca.components_).to(device=device)
            self.mem_pca_mean[i] += pca_mean
            self.mem_pca_components[i] += pca_components

        self.mem_postprocessed[0] = True

        if PRINT_TIMING:
            end_time = time.time()
            print("--------------------------------")
            print("TIMING POSTPROCESS MEMORY: ", end_time - start_time)
            print("--------------------------------")

    def postprocess_memory_negative(self):
        mem_feats_avg_neg = (
                torch.sum(self.mem_feats_neg * self.mem_masks_neg.unsqueeze(dim=-1), dim=(1, 2))
                / self.mem_masks_neg.sum(dim=(1, 2)).unsqueeze(dim=1)
        )
        self.mem_feats_avg_neg += mem_feats_avg_neg

        mem_feats_ins_avg_neg = (
                torch.sum(self.mem_feats_neg * self.mem_masks_neg.unsqueeze(dim=-1), dim=2)
                / self.mem_masks_neg.sum(dim=2).unsqueeze(dim=2)
        )
        self.mem_feats_ins_avg_neg += mem_feats_ins_avg_neg
        self.mem_postprocessed_neg[0] = True

    def forward(self, input_dicts):
        data_mode = input_dicts[0].pop("data_mode", None)

        assert data_mode is not None
        assert not self.training

        if data_mode == "fill_memory":
            if PRINT_TIMING:
                start_time = time.time()
            results = self.forward_fill_memory(input_dicts, is_positive=True)
            if PRINT_TIMING:
                end_time = time.time()
                print("--------------------------------")
                print("TIMING FILL MEMORY: ", end_time - start_time)
                print("--------------------------------")
            return results
        elif data_mode == "fill_memory_neg":
            assert self.with_negative_refs
            assert not self.memory_neg_ready
            assert not self.mem_postprocessed_neg[0].item()
            return self.forward_fill_memory(input_dicts, is_positive=False)
        elif data_mode == "vis_memory":
            return self.forward_vis_memory(input_dicts)
        elif data_mode == "test":
            if self.with_negative_refs:
                if not self.memory_ready:
                    if self.mem_postprocessed[0].item():
                        self.memory_ready = True
                    else:
                        raise RuntimeError("Memory is not ready!")
                if not self.memory_neg_ready:
                    if self.mem_postprocessed_neg[0].item():
                        self.memory_neg_ready = True
                    else:
                        raise RuntimeError("Negative memory is not ready!")

                return self.forward_test(input_dicts, with_negative=True)
            else:
                if not self.memory_ready:
                    if self.mem_postprocessed[0].item():
                        self.memory_ready = True
                    else:
                        raise RuntimeError("Memory is not ready!")
                return self.forward_test(input_dicts, with_negative=False)
        elif data_mode == "test_support":
            assert self.with_negative_refs
            if not self.memory_ready:
                if self.mem_postprocessed[0].item():
                    self.memory_ready = True
                else:
                    raise RuntimeError("Memory is not ready!")
            assert not self.memory_neg_ready
            assert not self.mem_postprocessed_neg[0].item()
            return self.forward_test(input_dicts, with_negative=False)
        else:
            raise NotImplementedError(f"Unrecognized data mode during inference: {data_mode}")

    def _vis_results_online(self, output_dict, tar_anns_by_cat, score_thr=0.65, show_scores=False, dataset_name=None, dataset_imgs_path=None, class_names=None):
        import os

        from no_time_to_train.dataset.visualization import vis_coco

        scores = output_dict["scores"].cpu().numpy()
        masks_pred = output_dict["binary_masks"].cpu().numpy()
        bboxes = output_dict["bboxes"].cpu().numpy()
        labels = output_dict["labels"].cpu().numpy()

        image_info = output_dict["image_info"]
        if dataset_name == "coco" or dataset_name == "few_shot_classes":
            img_path = os.path.join(f"./data/coco/val2017", image_info["file_name"])
        elif dataset_name == "lvis":
            img_path = os.path.join(f"./data/coco/allimages", image_info["file_name"])
        else:
            img_path = os.path.join(dataset_imgs_path, image_info["file_name"])
        out_path = os.path.join(f"./results_analysis/{dataset_name}", image_info["file_name"])

        gt_masks = []
        gt_bboxes = []
        gt_labels = []

        for cat_ind in tar_anns_by_cat.keys():
            gt_masks.append(tar_anns_by_cat[cat_ind]["masks"].cpu().numpy())
            gt_bboxes.append(tar_anns_by_cat[cat_ind]["bboxes"].cpu().numpy())
            gt_labels.extend([cat_ind for _ in range(len(tar_anns_by_cat[cat_ind]["masks"]))])
        if len(gt_bboxes) > 0:
            gt_bboxes = np.concatenate(gt_bboxes)
            gt_masks = np.concatenate(gt_masks)

            gt_bboxes[:, 0] = gt_bboxes[:, 0] * image_info["ori_width"] / self.sam_img_size
            gt_bboxes[:, 1] = gt_bboxes[:, 1] * image_info["ori_height"] / self.sam_img_size
            gt_bboxes[:, 2] = gt_bboxes[:, 2] * image_info["ori_width"] / self.sam_img_size
            gt_bboxes[:, 3] = gt_bboxes[:, 3] * image_info["ori_height"] / self.sam_img_size

        # Resize gt masks
        if len(gt_masks) > 0:
            gt_masks = F.interpolate(
                torch.from_numpy(gt_masks).unsqueeze(dim=1),
                size=(image_info["ori_height"], image_info["ori_width"]),
                mode="nearest"
            ).squeeze(dim=1).numpy()

        vis_coco(
            gt_bboxes,
            gt_labels,
            gt_masks,
            scores,
            labels,
            bboxes,
            masks_pred,
            score_thr=score_thr,
            img_path=img_path,
            out_path=out_path,
            show_scores=show_scores,
            dataset_name=dataset_name,
            class_names=class_names
        )