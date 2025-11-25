import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from copy import deepcopy

# import faiss
# import faiss.contrib.torch_utils

from sam2.automatic_mask_generator import (
    SAM2AutomaticMaskGenerator,
    generate_crop_boxes,
    MaskData,
    batch_iterator,
    calculate_stability_score,
    batched_mask_to_box,
    is_box_near_crop_edge,
    uncrop_masks,
    mask_to_rle_pytorch,
    batched_nms,
    uncrop_boxes_xyxy,
    uncrop_points,
    coco_encode_rle,
    rle_to_mask,
    area_from_rle
)

def fast_l2(x, y, sqrt=True):
    """
    Compute L2 distances between x and y in the last dimension.

    Args:
        x (torch.Tensor): Tensor of shape (..., D).
        y (torch.Tensor): Tensor of shape (..., D).
        sqrt (bool, optional): Whether to take the square root of the distance. Defaults to True.

    Returns:
        torch.Tensor: L2 distances.
    """
    # compute L2 distances between x, y in dim -2

    assert len(x.shape) == len(y.shape)
    assert x.shape[-1] == y.shape[-1]

    x_norms = torch.pow(x, 2.0).sum(dim=-1, keepdim=True)
    y_norms = torch.pow(y, 2.0).sum(dim=-1, keepdim=True)

    dist = x_norms + y_norms.transpose(-1, -2) - 2 * (x @ y.transpose(-1, -2))
    if sqrt:
        dist = torch.sqrt(torch.clamp(dist, min=0.0))
    return dist


def kmeans(feats, k, n_iter=100):
    """
    Perform K-means clustering on features.

    Args:
        feats (torch.Tensor): Input features of shape (N, C).
        k (int): Number of clusters.
        n_iter (int, optional): Number of iterations. Defaults to 100.

    Returns:
        torch.Tensor: Cluster centers of shape (k, C).
    """
    assert len(feats.shape) == 2

    device = feats.device
    n, c = feats.shape

    centers = feats[torch.randperm(n)[:k].to(device=device)]  # [k, c]
    for i in range(n_iter):
        sim = F.normalize(feats, p=2, dim=-1) @ F.normalize(centers, p=2, dim=-1).t()  # [n, k]
        new_center_inds = torch.argmax(sim, dim=1)  # [n]

        new_centers = []
        for j in range(k):
            new_centers.append(feats[new_center_inds==j].mean(dim=0, keepdim=True))
        centers = torch.cat(new_centers, dim=0)  # [k, c]
    centers = F.normalize(centers, dim=-1)
    return centers


def kmeans_decouple(feats, feats_fore, k, n_iter=100):
    """
    Perform decoupled K-means clustering.

    Args:
        feats (torch.Tensor): All features.
        feats_fore (torch.Tensor): Foreground features used for initial clustering.
        k (int): Number of clusters.
        n_iter (int, optional): Number of iterations. Defaults to 100.

    Returns:
        torch.Tensor: Cluster centers.
    """
    assert len(feats.shape) == 2

    device = feats.device
    n, c = feats.shape

    centers = feats_fore[torch.randperm(n)[:k].to(device=device)]  # [k, c]
    for i in range(n_iter):
        sim = F.normalize(feats, p=2, dim=-1) @ F.normalize(centers, p=2, dim=-1).t()  # [n, k]
        new_center_inds = torch.argmax(sim, dim=1)  # [n]

        new_centers = []
        for j in range(k):
            new_centers.append(feats_fore[new_center_inds==j].mean(dim=0, keepdim=True))
        centers = torch.cat(new_centers, dim=0)  # [k, c]

    sim_fore = F.normalize(feats_fore, p=2, dim=-1) @ F.normalize(centers, p=2, dim=-1).t()
    assign_inds = torch.argmax(sim_fore, dim=-1)

    new_centers = []
    for i in range(k):
        new_centers.append(
            feats[assign_inds == i].mean(dim=0, keepdim=True)
        )
    new_centers = torch.cat(new_centers, dim=0)
    centers = F.normalize(new_centers, dim=-1)
    return centers


def compute_foundpose(feats: torch.Tensor, masks: torch.Tensor, k_kmeans: int, n_pca: int):
    """
    Compute FoundPose features using PCA and K-means.

    Args:
        feats (torch.Tensor): Input features.
        masks (torch.Tensor): Masks indicating foreground.
        k_kmeans (int): Number of K-means clusters.
        n_pca (int): Number of PCA components.
    """
    # Reference: https://github.com/facebookresearch/foundpose/

    device = feats.device
    n_class, n_shot = feats.shape[:2]
    mem_dim = feats.shape[-1]

    feats = feats.reshape(n_class * n_shot, -1, mem_dim)
    masks = feats.reshape(n_class * n_shot, -1)

    fore_feats = []
    for i in range(feats.shape[0]):
        fore_feats.append(feats[i][masks[i] > 0])
    fore_feats = torch.cat(fore_feats, dim=0)

    # PCA
    fore_feats_np = fore_feats.cpu().numpy()
    pca = PCA(n_components=n_pca)
    pca.fit(fore_feats_np)
    pca_mean = torch.from_numpy(pca.mean_).to(device=device)
    pca_components = torch.from_numpy(pca.components_).to(device=device)
    fore_feats = (fore_feats - pca_mean.reshape(1, -1)) @ pca_components.t()

    # K-means
    fore_feats_cpu = fore_feats.cpu()
    kmeans = faiss.Kmeans(
        n_pca,
        k_kmeans,
        niter=100,
        gpu=False,
        verbose=True,
        seed=0,
        spherical=False,
    )
    kmeans.train(fore_feats_cpu)

    centers = array_to_tensor(kmeans.centroids).to(device)

    centroid_distances, cluster_ids = kmeans.index.search(fore_feats_cpu, 1)
    centroid_distances = centroid_distances.squeeze(axis=-1).to(device=device)
    cluster_ids = cluster_ids.squeeze(axis=-1).to(device=device)

    # TF-IDF







def vis_kmeans(ref_imgs, ref_masks_ori, ref_cat_ind, ref_feats, ref_masks, feats_centers, encoder_shape_info, device, transparency):
    """
    Visualize K-means clustering results on reference images.

    Args:
        ref_imgs (torch.Tensor): Reference images.
        ref_masks_ori (torch.Tensor): Original reference masks.
        ref_cat_ind (int): Category index.
        ref_feats (torch.Tensor): Reference features.
        ref_masks (torch.Tensor): Reference masks.
        feats_centers (torch.Tensor): Feature centers from K-means.
        encoder_shape_info (dict): Encoder shape information (height, width, patch_size).
        device (torch.device): Device.
        transparency (float): Transparency for visualization.

    Returns:
        torch.Tensor: Visualized image.
    """
    encoder_h = encoder_shape_info.get("height")
    encoder_w = encoder_shape_info.get("width")
    encoder_patch_size = encoder_shape_info.get("patch_size")

    assert encoder_h == encoder_w
    encoder_img_size = encoder_h * encoder_patch_size

    color_template = torch.tensor([
        [255, 0, 0],
        [148, 33, 146],
        [255, 251, 0],
        [170, 121, 66],
        [4, 51, 255],
        [0, 249, 0],
        [255, 64, 255],
        [0, 253, 255]
    ]).to(device=device, dtype=torch.float32)

    cat_centers = feats_centers[ref_cat_ind]
    n_centers = cat_centers.shape[0]
    assert n_centers <= len(color_template)

    center_assign = (
        F.normalize(ref_feats, p=2, dim=-1)
        @ cat_centers.t()
    ).max(dim=-1)[-1]

    center_assign[ref_masks == 0] = -1
    center_assign = center_assign.to(dtype=torch.long)

    canvas = torch.zeros(
        (encoder_h * encoder_w, encoder_patch_size, encoder_patch_size, 3), device=device
    )
    for i in range(len(center_assign)):
        if center_assign[i].item() != -1:
            canvas[i, :, :, :] += color_template[center_assign[i]]
    canvas = canvas.reshape(encoder_h, encoder_w, encoder_patch_size, encoder_patch_size, 3)
    canvas = canvas.permute(0, 2, 1, 3, 4).reshape((encoder_img_size, encoder_img_size, 3))

    vis_img = ref_imgs[0].permute(1, 2, 0) * 255.0

    color_ws = ref_masks_ori.reshape(encoder_img_size, encoder_img_size, 1) > 0
    color_ws = color_ws.to(dtype=torch.float32) * transparency
    out_vis = vis_img * (1 - color_ws) + canvas * color_ws
    return out_vis


def vis_pca(ref_imgs, ref_masks_ori, ref_cat_ind, ref_feats, ref_masks, pca_means, pca_components, encoder_shape_info, device, transparency):
    """
    Visualize PCA results on reference images.

    Args:
        ref_imgs (torch.Tensor): Reference images.
        ref_masks_ori (torch.Tensor): Original reference masks.
        ref_cat_ind (int): Category index.
        ref_feats (torch.Tensor): Reference features.
        ref_masks (torch.Tensor): Reference masks.
        pca_means (torch.Tensor): PCA means.
        pca_components (torch.Tensor): PCA components.
        encoder_shape_info (dict): Encoder shape information.
        device (torch.device): Device.
        transparency (float): Transparency.

    Returns:
        torch.Tensor: Visualized image.
    """
    encoder_h = encoder_shape_info.get("height")
    encoder_w = encoder_shape_info.get("width")
    encoder_patch_size = encoder_shape_info.get("patch_size")

    assert encoder_h == encoder_w
    encoder_img_size = encoder_h * encoder_patch_size

    foreground_inds = ref_masks > 0

    foreground_feats = ref_feats[foreground_inds]
    pca_mean = pca_means[ref_cat_ind].unsqueeze(dim=0)
    pca_components = pca_components[ref_cat_ind]
    pca_weights = (foreground_feats - pca_mean) @ pca_components.t()
    _max_w = pca_weights.max()
    _min_w = pca_weights.min()
    pca_weights = (pca_weights - _min_w) / (_max_w - _min_w)

    rgb = torch.zeros((encoder_h * encoder_w, 3), device=device)
    rgb[foreground_inds] += pca_weights * 255.0

    canvas = torch.zeros(
        (encoder_h * encoder_w, encoder_patch_size, encoder_patch_size, 3), device=device
    )
    for i in range(encoder_h * encoder_w):
        canvas[i, :, :, :] += rgb[i]
    canvas = canvas.reshape(encoder_h, encoder_w, encoder_patch_size, encoder_patch_size, 3)
    canvas = canvas.permute(0, 2, 1, 3, 4).reshape((encoder_img_size, encoder_img_size, 3))

    vis_img = ref_imgs[0].permute(1, 2, 0) * 255.0

    color_ws = ref_masks_ori.reshape(encoder_img_size, encoder_img_size, 1) > 0
    color_ws = color_ws.to(dtype=torch.float32) * transparency
    out_vis = vis_img * (1 - color_ws) + canvas * color_ws
    return out_vis







class SAM2AutomaticMaskGenerator_MatchingBaseline(SAM2AutomaticMaskGenerator):
    """
    SAM2 Automatic Mask Generator adapted for Matching Baseline.
    """
    @torch.no_grad()
    def generate(
        self,
        image,
        select_point_coords=None,
        select_point_labels=None,
        select_box=None,
        select_mask_input=None
    ):
        """
        Generate masks for the given image.

        Args:
            image (np.ndarray): Input image.
            select_point_coords (np.ndarray, optional): Point coordinates. Defaults to None.
            select_point_labels (np.ndarray, optional): Point labels. Defaults to None.
            select_box (np.ndarray, optional): Bounding box. Defaults to None.
            select_mask_input (np.ndarray, optional): Mask input. Defaults to None.

        Returns:
            tuple: (masks, ious, low_res_masks)
        """
        mask_data = self._generate_masks(
            image, select_point_coords, select_point_labels, select_box, select_mask_input
        )
        masks = mask_data["masks"]
        ious = mask_data["iou_preds"]
        return masks, ious, mask_data["low_res_masks"]

    def _generate_masks(
        self,
        image,
        select_point_coords,
        select_point_labels,
        select_box,
        select_mask_input
    ):
        """
        Internal method to generate masks.

        Args:
            image (np.ndarray): Input image.
            select_point_coords (np.ndarray): Point coordinates.
            select_point_labels (np.ndarray): Point labels.
            select_box (np.ndarray): Bounding box.
            select_mask_input (np.ndarray): Mask input.

        Returns:
            MaskData: Generated mask data.
        """
        orig_size = image.shape[:2]
        crop_box = [0, 0, orig_size[-1], orig_size[-2]]

        data = self._process_crop(
            image,
            crop_box,
            0,
            orig_size,
            select_point_coords,
            select_point_labels,
            select_box,
            select_mask_input
        )
        return data

    def _process_crop(
        self,
        image,
        crop_box,
        crop_layer_idx,
        orig_size,
        select_point_coords,
        select_point_labels,
        select_box,
        select_mask_input
    ):
        """
        Process a single crop of the image.

        Args:
            image (np.ndarray): Image crop.
            crop_box (list): Crop box coordinates [x0, y0, x1, y1].
            crop_layer_idx (int): Index of the crop layer.
            orig_size (tuple): Original image size.
            select_point_coords (np.ndarray): Point coordinates.
            select_point_labels (np.ndarray): Point labels.
            select_box (np.ndarray): Bounding box.
            select_mask_input (np.ndarray): Mask input.

        Returns:
            MaskData: Mask data for the crop.
        """
        x0, y0, x1, y1 = crop_box
        cropped_im_size = (int(y1-y0), int(x1-x0))

        points_scale = np.array(cropped_im_size)[None, ::-1]
        points_for_image = self.point_grids[crop_layer_idx] * points_scale

        data = MaskData()

        for (points,) in batch_iterator(self.points_per_batch, points_for_image):
            batch_data = self._process_batch(points, cropped_im_size, crop_box, orig_size, normalize=True)
            data.cat(batch_data)
            del batch_data

        # Remove duplicates within this crop.
        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros_like(data["boxes"][:, 0]),  # categories
            iou_threshold=self.box_nms_thresh,
        )
        data.filter(keep_by_nms)

        # Return to the original image frame
        data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
        return data

    def _process_batch(
        self,
        points,
        im_size,
        crop_box,
        orig_size,
        normalize=False,
    ) -> MaskData:
        """
        Process a batch of points.

        Args:
            points (np.ndarray): Points to process.
            im_size (tuple): Image size.
            crop_box (list): Crop box.
            orig_size (tuple): Original image size.
            normalize (bool, optional): Whether to normalize coordinates. Defaults to False.

        Returns:
            MaskData: Mask data for the batch.
        """
        orig_h, orig_w = orig_size

        # Run model on this batch
        points = torch.as_tensor(
            points, dtype=torch.float32, device=self.predictor.device
        )
        in_points = self.predictor._transforms.transform_coords(
            points, normalize=normalize, orig_hw=im_size
        )
        in_labels = torch.ones(
            in_points.shape[0], dtype=torch.int, device=in_points.device
        )
        masks, iou_preds, low_res_masks = self.predictor._predict(
            in_points[:, None, :],
            in_labels[:, None],
            multimask_output=self.multimask_output,
            return_logits=True,
        )

        # Serialize predictions and store in MaskData
        data = MaskData(
            masks=masks.flatten(0, 1),
            iou_preds=iou_preds.flatten(0, 1),
            low_res_masks=low_res_masks.flatten(0, 1),
        )
        del masks

        if not self.use_m2m:
            # Filter by predicted IoU
            if self.pred_iou_thresh > 0.0:
                keep_mask = data["iou_preds"] > self.pred_iou_thresh
                data.filter(keep_mask)

            # Calculate and filter by stability score
            data["stability_score"] = calculate_stability_score(
                data["masks"], self.mask_threshold, self.stability_score_offset
            )
            if self.stability_score_thresh > 0.0:
                keep_mask = data["stability_score"] >= self.stability_score_thresh
                data.filter(keep_mask)
        else:
            in_points = self.predictor._transforms.transform_coords(
                points.repeat_interleave(masks.shape[1], dim=0), normalize=normalize, orig_hw=im_size
            )
            labels = torch.ones(
                in_points.shape[0], dtype=torch.int, device=in_points.device
            )
            masks, ious = self.refine_with_m2m(
                in_points, labels, data["low_res_masks"], self.points_per_batch
            )
            data["masks"] = masks.squeeze(1)
            data["iou_preds"] = ious.squeeze(1)

            if self.pred_iou_thresh > 0.0:
                keep_mask = data["iou_preds"] > self.pred_iou_thresh
                data.filter(keep_mask)

            data["stability_score"] = calculate_stability_score(
                data["masks"], self.mask_threshold, self.stability_score_offset
            )
            if self.stability_score_thresh > 0.0:
                keep_mask = data["stability_score"] >= self.stability_score_thresh
                data.filter(keep_mask)

        # Threshold masks and calculate boxes
        data["masks"] = data["masks"] > self.mask_threshold
        data["boxes"] = batched_mask_to_box(data["masks"])

        # Filter boxes that touch crop boundaries
        keep_mask = ~is_box_near_crop_edge(
            data["boxes"], crop_box, [0, 0, orig_w, orig_h]
        )
        if not torch.all(keep_mask):
            data.filter(keep_mask)

        # Compress to RLE
        data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
        return data


import torch.nn as nn

class MemoryBank(nn.Module):
    """
    Memory Bank for storing and retrieving features.
    """
    def __init__(self, config, kmeans_k, n_pca_components):
        """
        Initialize the MemoryBank.

        Args:
            config (dict): Configuration dictionary.
            kmeans_k (int): Number of K-means clusters.
            n_pca_components (int): Number of PCA components.
        """
        super().__init__()
        self.n_classes = config.get("category_num")
        self.length = config.get("length")
        self.feat_shape = config.get("feat_shape")
        self.kmeans_k = kmeans_k
        self.n_pca_components = n_pca_components
        
        assert len(self.feat_shape) == 2
        _mem_n, _mem_c = self.feat_shape

        self.register_buffer("fill_counts", torch.zeros((self.n_classes,), dtype=torch.long))
        self.register_buffer("feats", torch.zeros((self.n_classes, self.length, _mem_n, _mem_c)))
        self.register_buffer("masks", torch.zeros((self.n_classes, self.length, _mem_n)))
        self.register_buffer("feats_avg", torch.zeros((self.n_classes, _mem_c)))
        self.register_buffer("feats_ins_avg", torch.zeros((self.n_classes, self.length, _mem_c)))
        self.register_buffer("feats_covariances", torch.zeros((self.n_classes, _mem_c, _mem_c)))
        self.register_buffer("feats_centers", torch.zeros((self.n_classes, self.kmeans_k, _mem_c)))
        self.register_buffer("ins_sim_avg", torch.zeros((self.n_classes,)))
        self.register_buffer("pca_mean", torch.zeros((self.n_classes, _mem_c)))
        self.register_buffer("pca_components", torch.zeros((self.n_classes, self.n_pca_components, _mem_c)))
        self.register_buffer("postprocessed", torch.zeros((1,), dtype=torch.bool))
        self.ready = False

    def postprocess(self):
        """
        Post-process the memory bank features (compute averages, covariances, K-means, PCA).
        """
        # Compute class-wise average features
        c = self.feats.shape[-1]

        self.feats_avg *= 0.0
        mem_feats_avg = (
            torch.sum(self.feats * self.masks.unsqueeze(dim=-1), dim=(1, 2))
            / self.masks.sum(dim=(1, 2)).unsqueeze(dim=1)
        )
        self.feats_avg += mem_feats_avg

        mem_feats_ins_avg = (
            torch.sum(self.feats * self.masks.unsqueeze(dim=-1), dim=2)
            / self.masks.sum(dim=2).unsqueeze(dim=2)
        )
        self.feats_ins_avg += mem_feats_ins_avg

        sigmas = []
        for i in range(self.n_classes):
            feats = self.feats[i]
            masks = self.masks[i]
            feats = feats[masks > 0]
            if feats.shape[0] == 0:
                sigmas.append(torch.eye(c, device=self.feats.device).unsqueeze(dim=0))
                continue
            feats = feats - self.feats_avg[i]
            sigma = (feats.t() @ feats) / feats.shape[0]
            sigmas.append(sigma.unsqueeze(dim=0))
        sigmas = torch.cat(sigmas, dim=0)
        self.feats_covariances += sigmas

        ins_sims = []
        for i in range(self.n_classes):
            feats = self.feats_ins_avg[i]
            if self.fill_counts[i] == 0:
                ins_sims.append(torch.tensor(0.0, device=self.feats.device))
                continue
            feats = feats[:self.fill_counts[i]]
            feats = F.normalize(feats, p=2, dim=-1)
            sim = feats @ feats.t()
            # remove diagonal
            sim = sim[~torch.eye(sim.shape[0], dtype=torch.bool, device=self.feats.device)].reshape(sim.shape[0], -1)
            ins_sims.append(sim.mean())
        ins_sims = torch.stack(ins_sims).reshape(self.n_classes)
        self.ins_sim_avg += ins_sims

        # K-means
        for i in range(self.n_classes):
            feats = self.feats[i]
            masks = self.masks[i]
            feats = feats[masks > 0]
            if feats.shape[0] < self.kmeans_k:
                continue
            centers = kmeans(feats, self.kmeans_k)
            self.feats_centers[i] = centers

        # PCA
        for i in range(self.n_classes):
            feats = self.feats[i]
            masks = self.masks[i]
            feats = feats[masks > 0]
            if feats.shape[0] < self.n_pca_components:
                continue
            
            # sklearn PCA on CPU
            feats_np = feats.cpu().numpy()
            pca = PCA(n_components=self.n_pca_components)
            pca.fit(feats_np)
            
            self.pca_mean[i] = torch.from_numpy(pca.mean_).to(self.feats.device)
            self.pca_components[i] = torch.from_numpy(pca.components_).to(self.feats.device)

        self.postprocessed[0] = True


import os
from PIL import Image
from no_time_to_train.dataset.visualization import vis_coco

def vis_memory(input_dicts, memory_bank, encoder, encoder_transform, encoder_img_size, encoder_h, encoder_w, encoder_patch_size, device):
    """
    Visualize the memory bank contents.

    Args:
        input_dicts (list): Input dictionaries.
        memory_bank (MemoryBank): Memory bank instance.
        encoder (nn.Module): Encoder model.
        encoder_transform (callable): Encoder transform.
        encoder_img_size (int): Encoder image size.
        encoder_h (int): Encoder height.
        encoder_w (int): Encoder width.
        encoder_patch_size (int): Encoder patch size.
        device (torch.device): Device.

    Returns:
        dict: Empty dictionary.
    """
    assert len(input_dicts) == 1
    assert memory_bank.fill_counts[0].item() > 0
    assert memory_bank.n_pca_components == 3  # RGB

    output_dir = "./results_analysis/memory_vis"
    os.makedirs(output_dir, exist_ok=True)

    ref_cat_ind = list(input_dicts[0]["refs_by_cat"].keys())[0]

    ref_imgs = input_dicts[0]["refs_by_cat"][ref_cat_ind]["imgs"].to(device=device)
    ref_imgs = F.interpolate(
        ref_imgs,
        size=(encoder_img_size, encoder_img_size),
        mode="bicubic"
    )
    ref_imgs_normed = encoder_transform(ref_imgs)
    
    # Forward encoder
    n_skip_tokens = 1 + getattr(encoder.config, "num_register_tokens", 0)
    outputs = encoder(pixel_values=ref_imgs_normed, output_hidden_states=False)
    seq_feats = outputs.last_hidden_state
    feats = seq_feats[:, n_skip_tokens:, :]
    ref_feats = feats.reshape(ref_imgs.shape[0], -1, encoder.config.hidden_size)
    ref_feats = ref_feats.reshape(-1, encoder.config.hidden_size)

    ref_masks_ori = input_dicts[0]["refs_by_cat"][ref_cat_ind]["masks"].to(dtype=ref_feats.dtype, device=device)
    ref_masks = F.interpolate(
        ref_masks_ori.unsqueeze(dim=0),
        size=(encoder_h, encoder_w),
        mode="nearest"
    ).reshape(-1)

    encoder_shape_info = dict(
        height=encoder_h,
        width=encoder_w,
        patch_size=encoder_patch_size
    )

    pca_vis_result = vis_pca(
        ref_imgs,
        ref_masks_ori,
        ref_cat_ind,
        ref_feats,
        ref_masks,
        memory_bank.pca_mean,
        memory_bank.pca_components,
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
        memory_bank.feats_centers,
        encoder_shape_info,
        device,
        transparency=1.0
    )
    ori_img = ref_imgs[0].permute(1, 2, 0) * 255.0
    margin = torch.zeros((ori_img.shape[0], 5, 3), dtype=ori_img.dtype, device=device) + 255
    output_final = torch.cat((
        ori_img, margin, kmeans_vis_result, margin, pca_vis_result
    ), dim=1)

    out_vis_img = Image.fromarray(output_final.cpu().numpy().astype(np.uint8))
    img_id = int(input_dicts[0]["refs_by_cat"][ref_cat_ind]["img_info"][0]['id'])
    out_vis_img.save(os.path.join(output_dir, "%d_%d.png" % (ref_cat_ind, img_id)))
    return {}

def vis_results_online(output_dict, tar_anns_by_cat, sam_img_size, score_thr=0.65, show_scores=False, dataset_name=None, dataset_imgs_path=None, class_names=None):
    """
    Visualize inference results online.

    Args:
        output_dict (dict): Output dictionary from the model.
        tar_anns_by_cat (dict): Target annotations by category.
        sam_img_size (int): SAM image size.
        score_thr (float, optional): Score threshold. Defaults to 0.65.
        show_scores (bool, optional): Whether to show scores. Defaults to False.
        dataset_name (str, optional): Dataset name. Defaults to None.
        dataset_imgs_path (str, optional): Dataset images path. Defaults to None.
        class_names (list, optional): Class names. Defaults to None.
    """
    scores = output_dict["scores"].cpu().numpy()
    masks_pred = output_dict["binary_masks"].cpu().numpy()
    bboxes = output_dict["bboxes"].cpu().numpy()
    labels = output_dict["labels"].cpu().numpy()

    image_info = output_dict["image_info"]
    
    # Determine output path
    if dataset_name == "coco" or dataset_name == "few_shot_classes":
        img_path = os.path.join(dataset_imgs_path, image_info["file_name"])
    elif dataset_name == "lvis":
        img_path = os.path.join(dataset_imgs_path, image_info["file_name"])
    else:
        img_path = os.path.join(dataset_imgs_path, image_info["file_name"])
        
    out_dir = f"./results_analysis/{dataset_name}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(image_info["file_name"]))

    gt_masks = []
    gt_bboxes = []
    gt_labels = []

    for cat_ind in tar_anns_by_cat.keys():
        if "masks" in tar_anns_by_cat[cat_ind]:
            gt_masks.append(tar_anns_by_cat[cat_ind]["masks"].cpu().numpy())
        if "bboxes" in tar_anns_by_cat[cat_ind]:
            gt_bboxes.append(tar_anns_by_cat[cat_ind]["bboxes"].cpu().numpy())
        
        n_items = len(tar_anns_by_cat[cat_ind]["bboxes"])
        gt_labels.extend([cat_ind] * n_items)

    if len(gt_bboxes) > 0:
        gt_bboxes = np.concatenate(gt_bboxes)
        # Scaling logic
        gt_bboxes[:, 0] = gt_bboxes[:, 0] * image_info["ori_width"] / sam_img_size
        gt_bboxes[:, 1] = gt_bboxes[:, 1] * image_info["ori_height"] / sam_img_size
        gt_bboxes[:, 2] = gt_bboxes[:, 2] * image_info["ori_width"] / sam_img_size
        gt_bboxes[:, 3] = gt_bboxes[:, 3] * image_info["ori_height"] / sam_img_size
    
    if len(gt_masks) > 0:
        gt_masks = np.concatenate(gt_masks)
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

def compute_semantic_ios(masks_binary, labels, obj_sim, mem_n_classes, use_semantic=True, rank_score=True):
    """
    Compute semantic Intersection over Self (IoS) scores.

    Args:
        masks_binary (torch.Tensor): Binary masks.
        labels (torch.Tensor): Labels.
        obj_sim (torch.Tensor): Object similarity matrix.
        mem_n_classes (int): Number of memory classes.
        use_semantic (bool, optional): Whether to use semantic similarity. Defaults to True.
        rank_score (bool, optional): Whether to rank scores. Defaults to True.

    Returns:
        torch.Tensor: IoS scores.
    """
    n_masks = masks_binary.shape[0]
    masks = masks_binary.reshape(n_masks, -1).to(dtype=torch.float32)
    ios = torch.zeros((n_masks,), device=masks_binary.device, dtype=torch.float32)

    for cat_ind in range(mem_n_classes):
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
            inter_num = inter_num * _obj_sim
        _ios = (inter_num / pos_num[:, None])
        if use_semantic:
            _ios = _ios * _obj_sim
        _ios = _ios.max(dim=-1)[0]
        ios[select_idxs] += _ios
    return ios

def compute_sim_global_avg(tar_feat, masks_feat_size_bool, mem_feats_ins_avg, softmax=False, temp=1.0, ret_feats=False):
    """
    Compute global average similarity between target features and memory features.

    Args:
        tar_feat (torch.Tensor): Target features.
        masks_feat_size_bool (torch.Tensor): Mask booleans resized to feature size.
        mem_feats_ins_avg (torch.Tensor): Memory features instance average.
        softmax (bool, optional): Whether to apply softmax. Defaults to False.
        temp (float, optional): Temperature for softmax. Defaults to 1.0.
        ret_feats (bool, optional): Whether to return target average features. Defaults to False.

    Returns:
        torch.Tensor or tuple: Similarity scores, or (similarity scores, target average features).
    """
    masks = masks_feat_size_bool.to(dtype=tar_feat.dtype)
    tar_avg_feats = (masks @ tar_feat) / masks.sum(dim=-1, keepdim=True)
    tar_avg_feats = F.normalize(tar_avg_feats, p=2, dim=-1)

    mem_feats_avg = mem_feats_ins_avg.mean(dim=1)
    mem_feats_avg = F.normalize(mem_feats_avg, p=2, dim=-1)

    sim_avg = tar_avg_feats @ mem_feats_avg.t()
    if softmax:
        sim_avg = F.softmax(sim_avg / temp, dim=-1)
    else:
        sim_avg = sim_avg / temp
    if not ret_feats:
        return sim_avg
    else:
        return sim_avg, tar_avg_feats

def compute_sim_global_avg_with_neg(tar_feat, masks_feat_size_bool, mem_feats_avg, mem_feats_ins_avg_neg, mem_n_classes, sigma=1.0):
    """
    Compute global average similarity with negative references.

    Args:
        tar_feat (torch.Tensor): Target features.
        masks_feat_size_bool (torch.Tensor): Mask booleans.
        mem_feats_avg (torch.Tensor): Memory features average.
        mem_feats_ins_avg_neg (torch.Tensor): Negative memory features instance average.
        mem_n_classes (int): Number of memory classes.
        sigma (float, optional): Sigma parameter. Defaults to 1.0.

    Returns:
        torch.Tensor: Final similarity scores.
    """
    n_masks = masks_feat_size_bool.shape[0]
    c = tar_feat.shape[-1]

    masks = masks_feat_size_bool.to(dtype=tar_feat.dtype)
    tar_avg_feats = (masks @ tar_feat) / masks.sum(dim=-1, keepdim=True)
    tar_avg_feats = F.normalize(tar_avg_feats, p=2, dim=-1)

    mem_feats_avg = F.normalize(mem_feats_avg, p=2, dim=-1)

    mem_feats_ins_avg_neg = F.normalize(mem_feats_ins_avg_neg, p=2, dim=-1).reshape(-1, c)

    sim_pos = tar_avg_feats @ mem_feats_avg.t()
    sim_pos = sim_pos.clamp(min=0.0)

    sim_neg = tar_avg_feats @ mem_feats_ins_avg_neg.t()
    sim_neg = sim_neg.clamp(min=0.0)
    sim_neg = sim_neg.reshape(n_masks, mem_n_classes, -1)
    sim_neg, max_inds = sim_neg.max(dim=-1)

    sim_final = sim_pos * torch.exp(-1.0 * (sim_neg - sim_pos).clamp(min=0.0) / sigma)
    return sim_final

