import copy
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.ops.boxes import batched_nms
from torchvision.transforms import Normalize
from sam2.build_sam import build_sam2_video_predictor
from sam2.utils.amg import batched_mask_to_box

from no_time_to_train.models.matching_baseline_utils import (
    MemoryBank,
    vis_memory,
    vis_results_online,
    compute_semantic_ios,
    compute_sim_global_avg,
    compute_sim_global_avg_with_neg
)
from no_time_to_train.models.model_utils import (
    build_encoder, 
    concat_all_gather
)

encoder_predefined_cfgs = {
    # --- DINOv2 Variants ---
    "dinov2_small": dict(
        img_size=518,
        patch_size=14,
        init_values=1e-5,
        ffn_layer='mlp',
        block_chunks=0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        feat_dim=384,
        hf_model_name="facebook/dinov2-small"
    ),
    "dinov2_base": dict(
        img_size=518,
        patch_size=14,
        init_values=1e-5,
        ffn_layer='mlp',
        block_chunks=0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        feat_dim=768,
        hf_model_name="facebook/dinov2-base"
    ),
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
    "dinov2_giant": dict(
        img_size=518,
        patch_size=14,
        init_values=1e-5,
        ffn_layer='mlp',
        block_chunks=0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        feat_dim=1536,
        hf_model_name="facebook/dinov2-giant"
    ),

    # --- DINOv3 Variants ---
    "dinov3_small": dict(
        img_size=518,
        patch_size=16,
        # init_values=1e-5,
        ffn_layer='mlp',
        # block_chunks=0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        feat_dim=384,
        hf_model_name="facebook/dinov3-vits16-pretrain-lvd1689m"
    ),
    "dinov3_base": dict(
        img_size=518,
        patch_size=16,
        # init_values=1e-5,
        ffn_layer='mlp',
        # block_chunks=0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        feat_dim=768,
        hf_model_name="facebook/dinov3-vitb16-pretrain-lvd1689m"
    ),
    "dinov3_large": dict(
        img_size=518,
        patch_size=16,
        # init_values=1e-5,
        ffn_layer='mlp',
        # block_chunks=0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        feat_dim=1024,
        hf_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m"
    ),
    "dinov3_huge": dict(
        img_size=518,
        patch_size=16,
        # init_values=1e-5,
        ffn_layer='mlp',
        # block_chunks=0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        feat_dim=1280,
        hf_model_name="facebook/dinov3-vith16plus-pretrain-lvd1689m"
    )
}


class Sam2MatchingBaselineNoAMG(nn.Module):
    """
    SAM2-based Matching Baseline without Automatic Mask Generation (AMG).

    This model uses SAM2 for mask prediction and a separate encoder (e.g., DINOv2)
    for feature extraction and matching against a memory bank of reference examples.
    It supports few-shot segmentation tasks by building a memory bank from support images
    and then segmenting target images based on feature similarity.
    """
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
        """
        Initialize the Sam2MatchingBaselineNoAMG model.

        Args:
            sam2_cfg_file (str): Path to the SAM2 configuration file.
            sam2_ckpt_path (str): Path to the SAM2 checkpoint file.
            sam2_infer_cfgs (dict): Configuration dictionary for SAM2 inference parameters
                                    (e.g., points_per_side, iou_thr, etc.).
            encoder_cfg (str): Configuration key for the encoder (e.g., "dinov2_large").
            encoder_ckpt_path (str): Path to the encoder checkpoint file.
            memory_bank_cfg (dict): Configuration dictionary for the memory bank.
            dataset_name (str, optional): Name of the dataset. Defaults to 'coco'.
            dataset_imgs_path (str, optional): Path to dataset images. Defaults to None.
            class_names (list, optional): List of class names. Defaults to None.
            online_vis (bool, optional): Whether to enable online visualization. Defaults to False.
            vis_thr (float, optional): Visualization threshold. Defaults to 0.5.
        """
        super(Sam2MatchingBaselineNoAMG, self).__init__()

        print("Model Parameters:")
        print(f"sam2_cfg_file: {sam2_cfg_file}")
        print(f"sam2_ckpt_path: {sam2_ckpt_path}")
        print(f"sam2_infer_cfgs: {sam2_infer_cfgs}")
        print(f"encoder_cfg: {encoder_cfg}")
        print(f"encoder_ckpt_path: {encoder_ckpt_path}")
        print(f"memory_bank_cfg: {memory_bank_cfg}")
        print(f"dataset_name: {dataset_name}")

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
        
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        print(f"Using device: {device}")

        self.predictor = build_sam2_video_predictor(sam2_cfg_file, sam2_ckpt_path, device=device)
        self.sam_img_size = 1024

        self.encoder, self.encoder_img_size, self.encoder_patch_size, encoder_hw = build_encoder(
            encoder_cfg, encoder_ckpt_path, encoder_predefined_cfgs, device
        )
        self.encoder_h, self.encoder_w = encoder_hw, encoder_hw
        self.encoder_dim = self.encoder.config.hidden_size
        self.encoder_transform = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        self.predictor.eval()
        
        # Memory Bank
        assert memory_bank_cfg.pop("enable")
        memory_bank_cfg["feat_shape"] = (self.encoder_h * self.encoder_w, self.encoder_dim)
        
        self.memory_bank = MemoryBank(memory_bank_cfg, self.kmeans_k, self.n_pca_components).to(device)
        
        if self.with_negative_refs:
            neg_cfg = copy.deepcopy(memory_bank_cfg)
            neg_cfg["length"] = memory_bank_cfg.get("length_negative")
            self.memory_bank_neg = MemoryBank(neg_cfg, self.kmeans_k, self.n_pca_components).to(device)
        else:
            self.memory_bank_neg = None

        self._reset()

    def _reset(self):
        """Reset the backbone features and high-resolution features."""
        self.backbone_features = None
        self.backbone_hr_features = None

    def _forward_encoder(self, imgs):
        """
        Forward pass through the encoder to extract features.

        Args:
            imgs (torch.Tensor): Input images tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Extracted features of shape (B, N, D), where N is the number of patches
                          and D is the feature dimension.
        """
        assert len(imgs.shape) == 4
        n_skip_tokens = 1 + getattr(self.encoder.config, "num_register_tokens", 0)
        outputs = self.encoder(pixel_values=imgs, output_hidden_states=False)
        seq_feats = outputs.last_hidden_state
        feats = seq_feats[:, n_skip_tokens:, :]
        feats = feats.reshape(imgs.shape[0], -1, self.encoder_dim)
        return feats

    def _forward_sam_decoder(self, backbone_features, sparse_embeddings, dense_embeddings, backbone_hr_features, multimask_output=True):
        """
        Forward pass through the SAM2 mask decoder.

        Args:
            backbone_features (torch.Tensor): Image features from the backbone.
            sparse_embeddings (torch.Tensor): Sparse prompt embeddings (points, boxes).
            dense_embeddings (torch.Tensor): Dense prompt embeddings (masks).
            backbone_hr_features (list): High-resolution features from the backbone.
            multimask_output (bool, optional): Whether to output multiple masks. Defaults to True.

        Returns:
            tuple: A tuple containing:
                - low_res_masks (torch.Tensor): Predicted low-resolution masks.
                - scores (torch.Tensor): IoU scores for the predicted masks.
        """
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
            repeat_image=False,
            high_res_features=backbone_hr_features,
            return_iou_token_out=False,
            disable_custom_iou_embed=True,
            disable_mlp_obj_scores=True,
            output_all_masks=True,
        )

        if multimask_output:
            best_iou_inds = torch.argmax(ious[:, 1:], dim=-1) + 1
            batch_inds = torch.arange(B, device=device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds]
            scores = ious[batch_inds, best_iou_inds]
        else:
            low_res_masks = low_res_multimasks[:, 0]
            scores = ious[:, 0]
        return low_res_masks, scores

    def _compute_masks(self, backbone_features, backbone_hr_features, point_inputs):
        """
        Compute masks given backbone features and point inputs.

        Args:
            backbone_features (torch.Tensor): Image features.
            backbone_hr_features (list): High-resolution features.
            point_inputs (dict): Dictionary containing 'point_coords' and 'point_labels'.

        Returns:
            tuple: (low_res_masks, scores)
        """
        B = backbone_features.size(0)
        sam_point_coords = point_inputs["point_coords"]
        sam_point_labels = point_inputs["point_labels"]

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

    def _get_grid_points(self, sam_input_size, device):
        """
        Generate a grid of query points for SAM.

        Args:
            sam_input_size (int): Input size for SAM.
            device (torch.device): Device to create tensors on.

        Returns:
            torch.Tensor: Grid points coordinates.
        """
        x, y = torch.meshgrid(
            torch.linspace(0, sam_input_size-1, self.points_per_side),
            torch.linspace(0, sam_input_size-1, self.points_per_side),
            indexing='ij'
        )
        query_points = torch.stack((y.reshape(-1), x.reshape(-1)), dim=-1)
        query_points += 0.5
        return query_points.to(device=device)

    def _forward_sam(self, imgs, precomputed_points=None, point_normed=True):
        """
        Run SAM inference on images using grid points or precomputed points.

        Args:
            imgs (torch.Tensor): Input images.
            precomputed_points (torch.Tensor, optional): Precomputed points. Defaults to None.
            point_normed (bool, optional): Whether points are normalized. Defaults to True.

        Returns:
            tuple: (lr_masks_all, scores_all, points_all)
                - lr_masks_all: All predicted low-resolution masks.
                - scores_all: Scores for all masks.
                - points_all: Points used for prediction.
        """
        assert len(imgs.shape) == 4
        assert self.backbone_features is None
        assert self.backbone_hr_features is None

        device = imgs.device
        sam_input_size = imgs.shape[-2]

        if precomputed_points is None:
            query_points = self._get_grid_points(sam_input_size, device)
        else:
            if point_normed:
                query_points = precomputed_points * sam_input_size
            else:
                query_points = precomputed_points

        backbone_out = self.predictor.forward_image(imgs)
        _, img_vision_features, img_vision_pos_embeds, img_feat_sizes = (
            self.predictor._prepare_backbone_features(backbone_out)
        )

        img_feats = img_vision_features[-1].permute(1, 2, 0).reshape(1, -1, *img_feat_sizes[-1])
        self.backbone_features = img_feats
        img_feats = img_feats.expand(self.testing_point_bs, -1, -1, -1)

        hr_feats = [
            x.permute(1, 2, 0).reshape(1, -1, *s)
            for x, s in zip(img_vision_features[:-1], img_feat_sizes[:-1])
        ]
        self.backbone_hr_features = hr_feats
        hr_feats = [
            x.expand(self.testing_point_bs, -1, -1, -1) for x in hr_feats
        ]

        points = query_points.reshape(-1, 2)
        point_labels = torch.ones_like(points[:, 0:1]).to(dtype=torch.int32)
        n_points = points.shape[0]

        mask_scores = []
        lr_masks = []
        for i in range(0, n_points // self.testing_point_bs):
            i_start = i * self.testing_point_bs
            i_end = i_start + self.testing_point_bs
            points_i = points[i_start:i_end, :]
            p_labels_i = point_labels[i_start:i_end, :]
            point_inputs_i = dict(
                point_coords=points_i.reshape(self.testing_point_bs, 1, 2),
                point_labels=p_labels_i.reshape(self.testing_point_bs, 1)
            )
            lr_masks_i, scores_i = self._compute_masks(
                img_feats, hr_feats, point_inputs_i
            )
            mask_scores.append(scores_i.reshape(-1))
            lr_masks.append(lr_masks_i.reshape(-1, *lr_masks_i.shape[-2:]))
        
        scores_all = torch.cat(mask_scores, dim=0).reshape(-1)
        lr_masks_all = torch.cat(lr_masks, dim=0)
        lr_masks_all = lr_masks_all.reshape(-1, *lr_masks_all.shape[-2:])

        inds = scores_all > self.iou_thr
        points_all = points[inds]
        lr_masks_all = lr_masks_all[inds]
        scores_all = scores_all[inds]

        return lr_masks_all, scores_all, points_all

    def forward_fill_memory(self, input_dicts, is_positive):
        """
        Extract features from support images and fill the memory bank.

        Args:
            input_dicts (list): List of input dictionaries containing support images and masks.
            is_positive (bool): Whether to fill the positive memory bank or negative memory bank.

        Returns:
            dict: Empty dictionary.
        """
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

            target_bank = self.memory_bank if is_positive else self.memory_bank_neg
            
            for i in range(cat_ind_all.shape[0]):
                if dist.is_initialized():
                    assert (target_bank.n_classes * target_bank.length) % dist.get_world_size() == 0
                
                fill_ind = target_bank.fill_counts[cat_ind_all[i]]
                target_bank.feats[cat_ind_all[i], fill_ind] += feats_all[i]
                target_bank.masks[cat_ind_all[i], fill_ind] += masks_all[i]
                target_bank.fill_counts[cat_ind_all[i]] += 1

            return {}

    def forward_vis_memory(self, input_dicts):
        """
        Visualize the contents of the memory bank.

        Args:
            input_dicts (list): List of input dictionaries.

        Returns:
            dict: Visualization results.
        """
        return vis_memory(
            input_dicts, 
            self.memory_bank, 
            self.encoder, 
            self.encoder_transform, 
            self.encoder_img_size, 
            self.encoder_h, 
            self.encoder_w, 
            self.encoder_patch_size, 
            self.predictor.device
        )

    def _extract_target_features(self, tar_img, device):
        """
        Extract features from the target image using the encoder.

        Args:
            tar_img (torch.Tensor): Target image.
            device (torch.device): Device.

        Returns:
            tuple: (tar_feat, tar_img)
                - tar_feat: Extracted features.
                - tar_img: Target image tensor.
        """
        tar_img = tar_img.to(device=device)
        tar_img_encoder = F.interpolate(
            tar_img.unsqueeze(dim=0),
            size=(self.encoder_img_size, self.encoder_img_size),
            mode="bicubic"
        )
        tar_feat = self._forward_encoder(self.encoder_transform(tar_img_encoder))
        tar_feat = tar_feat.reshape(-1, self.encoder_dim)
        return tar_feat, tar_img

    def _process_sam_masks(self, lr_masks, tar_feat):
        """
        Process SAM masks and align them with target features.

        Args:
            lr_masks (torch.Tensor): Low-resolution masks from SAM.
            tar_feat (torch.Tensor): Target image features.

        Returns:
            tuple: (masks_feat_size_bool, tar_feat_spatial)
                - masks_feat_size_bool: Boolean masks resized to feature size.
                - tar_feat_spatial: Spatially arranged target features.
        """
        n_masks = lr_masks.shape[0]
        masks_feat_size_bool = lr_masks > 0
        masks_feat_size_bool = masks_feat_size_bool.reshape(n_masks, -1)
        
        tar_feat_spatial = tar_feat.reshape(1, self.encoder_h, self.encoder_w, -1).permute(0, 3, 1, 2)
        tar_feat_spatial = F.interpolate(
            tar_feat_spatial,
            size=tuple(lr_masks.shape[-2:]),
            mode="bilinear",
            align_corners=False,
            antialias=True
        ).reshape(-1, lr_masks.shape[-2] * lr_masks.shape[-1]).t()
        
        return masks_feat_size_bool, tar_feat_spatial

    def forward_test(self, input_dicts, with_negative):
        """
        Perform inference on target images using the memory bank.

        Args:
            input_dicts (list): List of input dictionaries containing target images.
            with_negative (bool): Whether to use negative references.

        Returns:
            list: List of output dictionaries containing masks, bboxes, scores, and labels.
        """
        assert len(input_dicts) == 1
        device = self.predictor.device
        
        tar_feat, tar_img = self._extract_target_features(input_dicts[0]["target_img"], device)
        
        # SAM inference
        tar_img_sam = tar_img.unsqueeze(dim=0)
        lr_masks, pred_ious, query_points = self._forward_sam(self.sam_transform(tar_img_sam))
        
        masks_feat_size_bool, tar_feat_spatial = self._process_sam_masks(lr_masks, tar_feat)

        if not with_negative:
            sim_global, obj_feats = compute_sim_global_avg(
                tar_feat_spatial, masks_feat_size_bool, self.memory_bank.feats_ins_avg, 
                softmax=False, temp=1.0, ret_feats=True
            )
        else:
            sim_global = compute_sim_global_avg_with_neg(
                tar_feat_spatial, masks_feat_size_bool, 
                self.memory_bank.feats_avg, self.memory_bank_neg.feats_ins_avg, 
                self.memory_bank.n_classes, sigma=0.8
            )
            # Recompute obj_feats for later use
            masks = masks_feat_size_bool.to(dtype=tar_feat_spatial.dtype)
            obj_feats = (masks @ tar_feat_spatial) / masks.sum(dim=-1, keepdim=True)
            obj_feats = F.normalize(obj_feats, p=2, dim=-1)

        merged_scores = sim_global
        
        # Top-k selection
        if self.cls_num_per_mask == -1:
            self.cls_num_per_mask = self.memory_bank.n_classes
        top_scores, labels = torch.topk(merged_scores, k=self.cls_num_per_mask)
        
        if self.cls_num_per_mask == self.memory_bank.n_classes:
            max_scores = top_scores[:, 0:1]
            top_scores = top_scores * (top_scores > (max_scores * 0.6))

        labels = labels.flatten()
        scores_all_class = top_scores.flatten()

        lr_bboxes = batched_mask_to_box(lr_masks > 0)
        lr_bboxes_expand = (
            lr_bboxes.unsqueeze(dim=1)
            .expand(-1, self.cls_num_per_mask, -1)
            .reshape(lr_masks.shape[0] * self.cls_num_per_mask, 4)
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

        # Filter positive scores
        pos_inds = scores_out > 0.0
        scores_out = scores_out[pos_inds]
        lr_masks_out = lr_masks_out[pos_inds]
        obj_feats_out = obj_feats_out[pos_inds]
        labels_out = labels_out[pos_inds]

        # Resizing and output
        ori_h = input_dicts[0]["target_img_info"]["ori_height"]
        ori_w = input_dicts[0]["target_img_info"]["ori_width"]

        if lr_masks_out.shape[0] == 0:
            self._reset()
            return [dict(
                binary_masks=torch.zeros((0, ori_h, ori_w), device=device, dtype=torch.bool),
                bboxes=torch.zeros((0, 4), device=device, dtype=torch.float32),
                scores=torch.zeros((0,), device=device, dtype=torch.float32),
                labels=torch.zeros((0,), device=device, dtype=torch.long),
                image_info=input_dicts[0]["target_img_info"],
            )]

        masks_out_binary = F.interpolate(
            lr_masks_out.unsqueeze(dim=1),
            size=(ori_h, ori_w),
            mode="bilinear",
            align_corners=False,
            antialias=True
        ).squeeze(dim=1) > 0

        bboxes = batched_mask_to_box(masks_out_binary)

        # Semantic IoU decay
        obj_sim = obj_feats_out @ obj_feats_out.t()
        obj_sim = obj_sim.clamp(min=0.0)
        ios = compute_semantic_ios(masks_out_binary, labels_out, obj_sim, self.memory_bank.n_classes, use_semantic=True, rank_score=True)
        score_decay = 1 - ios
        scores_out = scores_out * torch.pow(score_decay, 0.5)

        final_out_num = min(self.num_out_instance, scores_out.shape[0])
        final_out_inds = torch.argsort(scores_out, descending=True)[:final_out_num]

        output_dict = dict(
            binary_masks=masks_out_binary[final_out_inds],
            bboxes=bboxes[final_out_inds],
            scores=scores_out[final_out_inds],
            labels=labels_out[final_out_inds],
            image_info=input_dicts[0]["target_img_info"],
        )

        if self.online_vis:
            vis_results_online(
                output_dict, 
                input_dicts[0]["tar_anns_by_cat"], 
                self.sam_img_size,
                score_thr=self.vis_thr, 
                show_scores=True,
                dataset_name=self.dataset_name, 
                dataset_imgs_path=self.dataset_imgs_path, 
                class_names=self.class_names
            )
        
        self._reset()
        return [output_dict]

    def postprocess_memory(self):
        """
        Post-process the memory bank.
        """
        self.memory_bank.postprocess()

    def postprocess_memory_negative(self):
        """
        Post-process the negative memory bank.
        """
        self.memory_bank_neg.postprocess()

    def forward(self, input_dicts):
        """
        Main forward method dispatching to specific modes.

        Args:
            input_dicts (list): List of input dictionaries.

        Returns:
            Any: Result depending on the data_mode (fill_memory, vis_memory, test, etc.).
        """
        data_mode = input_dicts[0].pop("data_mode", None)
        assert data_mode is not None
        assert not self.training

        if data_mode == "fill_memory":
            return self.forward_fill_memory(input_dicts, is_positive=True)
        elif data_mode == "fill_memory_neg":
            assert self.with_negative_refs
            assert not self.memory_bank_neg.postprocessed[0].item()
            return self.forward_fill_memory(input_dicts, is_positive=False)
        elif data_mode == "vis_memory":
            return self.forward_vis_memory(input_dicts)
        elif data_mode == "test":
            if self.with_negative_refs:
                if not self.memory_bank.ready:
                    if self.memory_bank.postprocessed[0].item():
                        self.memory_bank.ready = True
                    else:
                        raise RuntimeError("Memory is not ready!")
                if not self.memory_bank_neg.ready:
                    if self.memory_bank_neg.postprocessed[0].item():
                        self.memory_bank_neg.ready = True
                    else:
                        raise RuntimeError("Negative memory is not ready!")
                return self.forward_test(input_dicts, with_negative=True)
            else:
                if not self.memory_bank.ready:
                    if self.memory_bank.postprocessed[0].item():
                        self.memory_bank.ready = True
                    else:
                        raise RuntimeError("Memory is not ready!")
                return self.forward_test(input_dicts, with_negative=False)
        elif data_mode == "test_support":
            assert self.with_negative_refs
            if not self.memory_bank.ready:
                if self.memory_bank.postprocessed[0].item():
                    self.memory_bank.ready = True
                else:
                    raise RuntimeError("Memory is not ready!")
            assert not self.memory_bank_neg.ready
            assert not self.memory_bank_neg.postprocessed[0].item()
            return self.forward_test(input_dicts, with_negative=False)
        else:
            raise NotImplementedError(f"Unrecognized data mode during inference: {data_mode}")