
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
import matplotlib
#matplotlib.use("TkAgg")   # oder "QtAgg"

import os
# ============================================================
# Shared low-level utilities
# ============================================================

def safe_div(num, den, eps=1e-8):
    if num is None or den is None:
        return np.nan
    return float(num) / float(den + eps)


def is_finite_scalar(x):
    try:
        return np.isfinite(x)
    except Exception:
        return False


def sanitize_image(img):
    img = np.asarray(img, dtype=np.float32)
    return np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)


def sanitize_mask(mask):
    mask = np.asarray(mask, dtype=np.float32)
    mask = np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
    return mask > 0.0


def mask_area(mask):
    return float(sanitize_mask(mask).sum())


def is_valid_mask(mask, min_area=3):
    return np.isfinite(mask_area(mask)) and mask_area(mask) >= min_area


def abs_change_map(img_a, img_b):
    return np.abs(sanitize_image(img_b) - sanitize_image(img_a))


def binary_change_mask(img_a, img_b, quantile=0.90):
    diff = abs_change_map(img_a, img_b)
    if diff.size == 0 or np.allclose(diff, 0):
        return np.zeros_like(diff, dtype=bool)
    return diff > np.quantile(diff, quantile)


def dilate_mask(mask, iterations=2):
    mask = sanitize_mask(mask)
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    return ndimage.binary_dilation(mask, iterations=iterations)


def erode_mask(mask, iterations=1):
    mask = sanitize_mask(mask)
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    return ndimage.binary_erosion(mask, iterations=iterations)


def boundary(mask):
    mask = sanitize_mask(mask)
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    return mask ^ erode_mask(mask, iterations=1)


def epsilon_ring(mask, inner_iter=1, outer_iter=3):
    """
    Ring-shaped epsilon region with a hole in the center.
    Contains both an inner and an outer boundary band.
    """
    mask = sanitize_mask(mask)
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    outer = ndimage.binary_dilation(mask, iterations=outer_iter)
    inner = ndimage.binary_erosion(mask, iterations=inner_iter) if inner_iter > 0 else mask
    return outer ^ inner


def inner_ring(mask, inner_iter=1):
    mask = sanitize_mask(mask)
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    return mask ^ erode_mask(mask, iterations=inner_iter)


def outer_ring(mask, outer_iter=3):
    mask = sanitize_mask(mask)
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    return dilate_mask(mask, iterations=outer_iter) ^ mask


def build_anatomical_regions(
    mask_org,
    roi_base_pad_iter=0,
    roi_relax_pad_iter=2,
    ring_inner_iter=1,
    ring_outer_iter=3,
):
    """
    Fair, method-independent regions built from the ORIGINAL mask only.

    roi_strict:
        Original mask, optionally slightly enlarged by roi_base_pad_iter.
    roi_relaxed:
        roi_strict enlarged further by roi_relax_pad_iter.
    epsilon_ring_*:
        Ring around the corresponding ROI.
    """
    roi_strict = sanitize_mask(mask_org)
    if roi_base_pad_iter > 0:
        roi_strict = dilate_mask(roi_strict, iterations=roi_base_pad_iter)

    roi_relaxed = roi_strict.copy()
    if roi_relax_pad_iter > 0:
        roi_relaxed = dilate_mask(roi_relaxed, iterations=roi_relax_pad_iter)

    return {
        "roi_strict": roi_strict,
        "roi_relaxed": roi_relaxed,
        "boundary_strict": boundary(roi_strict),
        "boundary_relaxed": boundary(roi_relaxed),
        "epsilon_ring_strict": epsilon_ring(roi_strict, inner_iter=ring_inner_iter, outer_iter=ring_outer_iter),
        "epsilon_ring_relaxed": epsilon_ring(roi_relaxed, inner_iter=ring_inner_iter, outer_iter=ring_outer_iter),
        "inner_ring_strict": inner_ring(roi_strict, inner_iter=ring_inner_iter),
        "outer_ring_strict": outer_ring(roi_strict, outer_iter=ring_outer_iter),
        "inner_ring_relaxed": inner_ring(roi_relaxed, inner_iter=ring_inner_iter),
        "outer_ring_relaxed": outer_ring(roi_relaxed, outer_iter=ring_outer_iter),
    }


def weighted_inside_ratio(img_a, img_b, roi_mask):
    diff = abs_change_map(img_a, img_b)
    roi_mask = sanitize_mask(roi_mask)
    total = diff.sum()
    if total <= 0:
        return np.nan
    return safe_div(diff[roi_mask].sum(), total)


def weighted_outside_ratio(img_a, img_b, roi_mask):
    inside = weighted_inside_ratio(img_a, img_b, roi_mask)
    return np.nan if not is_finite_scalar(inside) else float(1.0 - inside)


def binary_inside_ratio(change_mask, roi_mask):
    cm = np.asarray(change_mask, dtype=bool)
    roi = sanitize_mask(roi_mask)
    if cm.sum() == 0:
        return np.nan
    return safe_div(np.logical_and(cm, roi).sum(), cm.sum())


def roi_precision_recall_f1(change_mask, roi_mask):
    cm = np.asarray(change_mask, dtype=bool)
    roi = sanitize_mask(roi_mask)
    tp = np.logical_and(cm, roi).sum()
    fp = np.logical_and(cm, ~roi).sum()
    fn = np.logical_and(~cm, roi).sum()
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {"roi_precision": precision, "roi_recall": recall, "roi_f1": f1}


def distance_of_changes_to_boundary(change_mask, roi_mask):
    cm = np.asarray(change_mask, dtype=bool)
    bnd = boundary(roi_mask)
    if cm.sum() == 0 or bnd.sum() == 0:
        return {"mean_distance_to_boundary": np.nan, "median_distance_to_boundary": np.nan, "max_distance_to_boundary": np.nan, "dist_map": None}
    dist_map = ndimage.distance_transform_edt(~bnd)
    vals = dist_map[cm]
    if vals.size == 0:
        return {"mean_distance_to_boundary": np.nan, "median_distance_to_boundary": np.nan, "max_distance_to_boundary": np.nan, "dist_map": dist_map}
    return {
        "mean_distance_to_boundary": float(np.mean(vals)),
        "median_distance_to_boundary": float(np.median(vals)),
        "max_distance_to_boundary": float(np.max(vals)),
        "dist_map": dist_map,
    }


def region_change_density(img_a, img_b, region_mask):
    """
    Mean absolute change per pixel inside a region.
    """
    diff = abs_change_map(img_a, img_b)
    region_mask = sanitize_mask(region_mask)
    area = region_mask.sum()
    if area == 0:
        return np.nan
    return safe_div(diff[region_mask].sum(), area)


def outside_region_change_density(img_a, img_b, region_mask):
    """
    Mean absolute change per pixel outside a region.
    """
    diff = abs_change_map(img_a, img_b)
    region_mask = sanitize_mask(region_mask)
    outside = ~region_mask
    area = outside.sum()
    if area == 0:
        return np.nan
    return safe_div(diff[outside].sum(), area)


def inside_outside_density_ratio(img_a, img_b, region_mask):
    inside = region_change_density(img_a, img_b, region_mask)
    outside = outside_region_change_density(img_a, img_b, region_mask)
    if not is_finite_scalar(inside) or not is_finite_scalar(outside):
        return np.nan
    return safe_div(inside, outside)


def boundary_contrast(img_a, img_b, boundary_mask):
    """
    Sum of absolute change inside the boundary band divided by the
    sum of absolute change outside the boundary band. Values > 1 indicate
    boundary-focused changes.
    """
    diff = abs_change_map(img_a, img_b)
    boundary_mask = sanitize_mask(boundary_mask)
    inside = diff[boundary_mask].sum()
    outside = diff[~boundary_mask].sum()
    if inside <= 0 and outside <= 0:
        return np.nan
    return safe_div(inside, outside)


def pixel_l1_minimality(img_a, img_b):
    """Mean absolute pixel change over the full two-frame image/video. Lower is more minimal."""
    return float(np.mean(abs_change_map(img_a, img_b)))


def spearman_corr_safe(y_true, y_pred):
    """Small dependency-light Spearman correlation implementation with average ranks for ties."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return np.nan
    x = y_true[mask]
    y = y_pred[mask]

    def rankdata_average_ties(a):
        order = np.argsort(a)
        ranks = np.empty(len(a), dtype=np.float64)
        i = 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
                j += 1
            avg_rank = 0.5 * (i + j) + 1.0
            ranks[order[i:j + 1]] = avg_rank
            i = j + 1
        return ranks

    rx = rankdata_average_ties(x)
    ry = rankdata_average_ties(y)
    if np.std(rx) <= 1e-12 or np.std(ry) <= 1e-12:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def dice(mask_a, mask_b):
    a, b = sanitize_mask(mask_a), sanitize_mask(mask_b)
    denom = a.sum() + b.sum()
    if denom == 0:
        return np.nan
    return safe_div(2 * np.logical_and(a, b).sum(), denom)


def jaccard(mask_a, mask_b):
    a, b = sanitize_mask(mask_a), sanitize_mask(mask_b)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return np.nan
    return safe_div(np.logical_and(a, b).sum(), union)


def extract_shape_features(mask):
    mask = sanitize_mask(mask)
    if not is_valid_mask(mask):
        return {
            "area": np.nan, "perimeter": np.nan, "circularity": np.nan,
            "bbox_height": np.nan, "bbox_width": np.nan, "aspect_ratio": np.nan,
            "solidity": np.nan, "centroid_y": np.nan, "centroid_x": np.nan,
        }
    area = float(mask.sum())
    bnd = boundary(mask)
    perimeter = float(bnd.sum()) if bnd.sum() > 0 else np.nan
    coords = np.argwhere(mask)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    bbox_h = float(y_max - y_min + 1)
    bbox_w = float(x_max - x_min + 1)
    aspect_ratio = safe_div(max(bbox_h, bbox_w), min(bbox_h, bbox_w))
    solidity = safe_div(area, bbox_h * bbox_w)
    circularity = safe_div(4.0 * np.pi * area, perimeter ** 2) if is_finite_scalar(perimeter) and perimeter > 0 else np.nan
    centroid_y, centroid_x = coords.mean(axis=0)
    return {
        "area": area, "perimeter": perimeter, "circularity": circularity,
        "bbox_height": bbox_h, "bbox_width": bbox_w, "aspect_ratio": aspect_ratio,
        "solidity": solidity, "centroid_y": float(centroid_y), "centroid_x": float(centroid_x),
    }


def compare_shape_features(mask_org, mask_cf):
    f_org = extract_shape_features(mask_org)
    f_cf = extract_shape_features(mask_cf)
    out = {}
    for k in f_org:
        out[f"{k}_org"] = f_org[k]
        out[f"{k}_cf"] = f_cf[k]
        out[f"delta_{k}"] = float(f_cf[k] - f_org[k]) if is_finite_scalar(f_org[k]) and is_finite_scalar(f_cf[k]) else np.nan
    out["dice_org_cf"] = dice(mask_org, mask_cf)
    out["jaccard_org_cf"] = jaccard(mask_org, mask_cf)
    return out


def infer_ed_es_indices_from_masks(mask_pair):
    if mask_pair.shape[-1] != 2:
        return {"ed_idx": None, "es_idx": None, "area_frame0": np.nan, "area_frame1": np.nan, "valid_pair": False}
    a0 = mask_area(mask_pair[..., 0])
    a1 = mask_area(mask_pair[..., 1])
    valid0 = is_valid_mask(mask_pair[..., 0])
    valid1 = is_valid_mask(mask_pair[..., 1])
    if (not valid0) or (not valid1):
        return {"ed_idx": None, "es_idx": None, "area_frame0": a0, "area_frame1": a1, "valid_pair": False}
    ed_idx, es_idx = (0, 1) if a0 >= a1 else (1, 0)
    return {"ed_idx": ed_idx, "es_idx": es_idx, "area_frame0": a0, "area_frame1": a1, "valid_pair": True}


def pair_functional_features_from_mask_pair(mask_pair):
    info = infer_ed_es_indices_from_masks(mask_pair)
    out = {
        "ed_idx": info["ed_idx"], "es_idx": info["es_idx"],
        "area_frame0": info["area_frame0"], "area_frame1": info["area_frame1"],
        "valid_pair": info["valid_pair"], "area_ed": np.nan, "area_es": np.nan,
        "area_change": np.nan, "fac": np.nan,
    }
    if not info["valid_pair"]:
        return out
    area_ed = extract_shape_features(mask_pair[..., info["ed_idx"]])["area"]
    area_es = extract_shape_features(mask_pair[..., info["es_idx"]])["area"]
    if not is_finite_scalar(area_ed) or not is_finite_scalar(area_es) or area_ed <= 0:
        return out
    out["area_ed"] = area_ed
    out["area_es"] = area_es
    out["area_change"] = float(area_ed - area_es)
    out["fac"] = safe_div(area_ed - area_es, area_ed)
    return out


def temporal_mask_l1(mask_org_pair, mask_cf_pair):
    o0 = sanitize_mask(mask_org_pair[..., 0]).astype(float)
    o1 = sanitize_mask(mask_org_pair[..., 1]).astype(float)
    c0 = sanitize_mask(mask_cf_pair[..., 0]).astype(float)
    c1 = sanitize_mask(mask_cf_pair[..., 1]).astype(float)
    return float(np.mean(np.abs((o0 - o1) - (c0 - c1))))


def temporal_mask_xor(mask_org_pair, mask_cf_pair):
    xo = np.logical_xor(sanitize_mask(mask_org_pair[..., 0]), sanitize_mask(mask_org_pair[..., 1]))
    xc = np.logical_xor(sanitize_mask(mask_cf_pair[..., 0]), sanitize_mask(mask_cf_pair[..., 1]))
    denom = np.logical_or(xo, xc).sum()
    val = 0.0 if denom == 0 else float(np.logical_xor(xo, xc).sum() / (denom + 1e-8))
    return val, xo, xc


# ============================================================
# Main metrics computation
# ============================================================

def compute_advanced_cf_metrics(
    x_data,
    y_data,
    x_masks,
    y_cf_pred,
    reference_masks=None,
    change_quantile=0.90,
    roi_base_pad_iter=0,
    roi_relax_pad_iter=2,
    boundary_inner_iter=1,
    boundary_outer_iter=3,
    cf_validity_mae=None,
    cf_validity_rmse=None,
    cf_validity_spearman=None,
    cf_validity_mae_ef=None,
    cf_validity_rmse_ef=None,
    cf_validity_spearman_ef=None,
):
    x_org, x_rec, x_cf = x_data
    y_org, y_pred, y_cf_goal = y_data
    x_masks_org, x_masks_rec, x_masks_cf = x_masks

    x_org = np.asarray(x_org)
    x_cf = np.asarray(x_cf)
    x_masks_org = np.asarray(x_masks_org)
    x_masks_cf = np.asarray(x_masks_cf)

    y_org = np.asarray(y_org).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    y_cf_goal = np.asarray(y_cf_goal).reshape(-1)
    y_cf_pred = np.asarray(y_cf_pred).reshape(-1)

    if x_org.shape != x_cf.shape:
        raise ValueError(f"x_org and x_cf must have same shape, got {x_org.shape} vs {x_cf.shape}")
    if x_org.shape != x_masks_org.shape:
        raise ValueError(f"x_org and x_masks_org must have same shape, got {x_org.shape} vs {x_masks_org.shape}")
    if x_masks_org.shape != x_masks_cf.shape:
        raise ValueError(f"x_masks_org and x_masks_cf must have same shape, got {x_masks_org.shape} vs {x_masks_cf.shape}")
    if x_org.ndim != 4 or x_org.shape[-1] != 2:
        raise ValueError(f"Expected shape (N,H,W,2), got {x_org.shape}")

    if reference_masks is not None:
        reference_masks = np.asarray(reference_masks)
        if reference_masks.ndim != 4 or reference_masks.shape[-1] != 2:
            raise ValueError(f"reference_masks must have shape (N_ref,H,W,2), got {reference_masks.shape}")

    def cf_regression_success_metrics(y_pred_i, y_goal_i, y_cf_pred_i):
        if not (is_finite_scalar(y_pred_i) and is_finite_scalar(y_goal_i) and is_finite_scalar(y_cf_pred_i)):
            return {
                "error_org_to_goal": np.nan, "error_cf_to_goal": np.nan,
                "improvement_to_goal": np.nan, "relative_improvement_to_goal": np.nan,
                "direction_match": np.nan, "target_success_soft": np.nan,
                "target_success_hard": np.nan,
            }
        err_org = abs(float(y_pred_i) - float(y_goal_i))
        err_cf = abs(float(y_cf_pred_i) - float(y_goal_i))
        improvement = err_org - err_cf
        rel_improvement = improvement / err_org if err_org > 1e-8 else (1.0 if err_cf <= 1e-8 else 0.0)
        desired_direction = np.sign(float(y_goal_i) - float(y_pred_i))
        actual_direction = np.sign(float(y_cf_pred_i) - float(y_pred_i))
        direction_match = 1.0 if (desired_direction == 0 and err_cf <= err_org + 1e-8) or (desired_direction != 0 and actual_direction == desired_direction) else 0.0
        return {
            "error_org_to_goal": float(err_org),
            "error_cf_to_goal": float(err_cf),
            "improvement_to_goal": float(improvement),
            "relative_improvement_to_goal": float(rel_improvement),
            "direction_match": float(direction_match),
            "target_success_soft": float(np.clip(rel_improvement, 0.0, 1.0)),
            "target_success_hard": float(err_cf < err_org),
        }

    def aligned_pair_features(mask_pair_org, mask_pair_cf):
        org_info = infer_ed_es_indices_from_masks(mask_pair_org)
        out = {
            "org_ed_idx": org_info["ed_idx"], "org_es_idx": org_info["es_idx"], "org_valid_pair": org_info["valid_pair"],
            "aligned_org_area_ed": np.nan, "aligned_org_area_es": np.nan,
            "aligned_cf_area_at_org_ed_pos": np.nan, "aligned_cf_area_at_org_es_pos": np.nan,
            "aligned_org_area_change": np.nan, "aligned_cf_area_change_wrt_org_pos": np.nan,
            "aligned_org_fac": np.nan, "aligned_cf_fac_wrt_org_pos": np.nan,
            "aligned_delta_area_change": np.nan, "aligned_delta_fac": np.nan,
            "order_preserved_wrt_org": np.nan, "aligned_ed_dice": np.nan, "aligned_es_dice": np.nan,
        }
        if not org_info["valid_pair"]:
            return out
        org_ed_idx, org_es_idx = org_info["ed_idx"], org_info["es_idx"]
        org_ed_area = mask_area(mask_pair_org[..., org_ed_idx]) if is_valid_mask(mask_pair_org[..., org_ed_idx]) else np.nan
        org_es_area = mask_area(mask_pair_org[..., org_es_idx]) if is_valid_mask(mask_pair_org[..., org_es_idx]) else np.nan
        cf_ed_area = mask_area(mask_pair_cf[..., org_ed_idx]) if is_valid_mask(mask_pair_cf[..., org_ed_idx]) else np.nan
        cf_es_area = mask_area(mask_pair_cf[..., org_es_idx]) if is_valid_mask(mask_pair_cf[..., org_es_idx]) else np.nan
        out["aligned_org_area_ed"] = org_ed_area
        out["aligned_org_area_es"] = org_es_area
        out["aligned_cf_area_at_org_ed_pos"] = cf_ed_area
        out["aligned_cf_area_at_org_es_pos"] = cf_es_area
        if is_finite_scalar(org_ed_area) and is_finite_scalar(org_es_area) and org_ed_area > 0:
            out["aligned_org_area_change"] = float(org_ed_area - org_es_area)
            out["aligned_org_fac"] = safe_div(org_ed_area - org_es_area, org_ed_area)
        if is_finite_scalar(cf_ed_area) and is_finite_scalar(cf_es_area) and cf_ed_area > 0:
            out["aligned_cf_area_change_wrt_org_pos"] = float(cf_ed_area - cf_es_area)
            out["aligned_cf_fac_wrt_org_pos"] = safe_div(cf_ed_area - cf_es_area, cf_ed_area)
        if is_finite_scalar(out["aligned_org_area_change"]) and is_finite_scalar(out["aligned_cf_area_change_wrt_org_pos"]):
            out["aligned_delta_area_change"] = float(out["aligned_cf_area_change_wrt_org_pos"] - out["aligned_org_area_change"])
        if is_finite_scalar(out["aligned_org_fac"]) and is_finite_scalar(out["aligned_cf_fac_wrt_org_pos"]):
            out["aligned_delta_fac"] = float(out["aligned_cf_fac_wrt_org_pos"] - out["aligned_org_fac"])
        if is_finite_scalar(cf_ed_area) and is_finite_scalar(cf_es_area):
            out["order_preserved_wrt_org"] = float(cf_ed_area >= cf_es_area)
        out["aligned_ed_dice"] = dice(mask_pair_org[..., org_ed_idx], mask_pair_cf[..., org_ed_idx])
        out["aligned_es_dice"] = dice(mask_pair_org[..., org_es_idx], mask_pair_cf[..., org_es_idx])
        return out

    def fit_reference_stats_from_masks(reference_masks):
        if reference_masks is None:
            return None
        feats = []
        for i in range(reference_masks.shape[0]):
            pair_feats = pair_functional_features_from_mask_pair(reference_masks[i])
            if pair_feats["valid_pair"]:
                vec = [pair_feats["area_ed"], pair_feats["area_es"], pair_feats["area_change"], pair_feats["fac"]]
                if np.all(np.isfinite(vec)):
                    feats.append(vec)
        if len(feats) < 3:
            return None
        feats = np.asarray(feats, dtype=np.float64)
        return {
            "feature_names": ["area_ed", "area_es", "area_change", "fac"],
            "mean": np.mean(feats, axis=0),
            "std": np.std(feats, axis=0) + 1e-8,
            "cov": np.cov(feats, rowvar=False),
        }

    def mahalanobis_distance(x, mean, cov):
        diff = np.asarray(x, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
        cov_inv = np.linalg.pinv(np.asarray(cov, dtype=np.float64))
        return float(np.sqrt(diff.T @ cov_inv @ diff))

    def plausibility_against_reference(feature_dict, reference_stats):
        if reference_stats is None:
            return {
                "reference_mahalanobis_distance": None, "reference_mean_abs_z": None,
                "reference_max_abs_z": None, "reference_z_area_ed": None,
                "reference_z_area_es": None, "reference_z_area_change": None,
                "reference_z_fac": None,
            }
        names = reference_stats["feature_names"]
        vec = np.array([feature_dict.get(n, np.nan) for n in names], dtype=np.float64)
        if not np.all(np.isfinite(vec)):
            return {
                "reference_mahalanobis_distance": None, "reference_mean_abs_z": None,
                "reference_max_abs_z": None, "reference_z_area_ed": None,
                "reference_z_area_es": None, "reference_z_area_change": None,
                "reference_z_fac": None,
            }
        z = (vec - reference_stats["mean"]) / reference_stats["std"]
        md = mahalanobis_distance(vec, reference_stats["mean"], reference_stats["cov"])
        return {
            "reference_mahalanobis_distance": float(md),
            "reference_mean_abs_z": float(np.mean(np.abs(z))),
            "reference_max_abs_z": float(np.max(np.abs(z))),
            "reference_z_area_ed": float(z[0]),
            "reference_z_area_es": float(z[1]),
            "reference_z_area_change": float(z[2]),
            "reference_z_fac": float(z[3]),
        }

    def counterfactual_specificity_score(target_success_soft, inside_lv_ratio, outside_change_ratio, boundary_ratio, temporal_penalty=None, plausibility_penalty=None, w_target=0.30, w_inside=0.20, w_outside=0.15, w_boundary=0.10, w_temporal=0.10, w_plaus=0.15):
        score = 0.0
        if is_finite_scalar(target_success_soft):
            score += w_target * target_success_soft
        if is_finite_scalar(inside_lv_ratio):
            score += w_inside * inside_lv_ratio
        if is_finite_scalar(outside_change_ratio):
            score -= w_outside * outside_change_ratio
        if is_finite_scalar(boundary_ratio):
            score += w_boundary * boundary_ratio
        if is_finite_scalar(temporal_penalty):
            score -= w_temporal * temporal_penalty
        if is_finite_scalar(plausibility_penalty):
            score -= w_plaus * plausibility_penalty
        return float(score)

    reference_stats = fit_reference_stats_from_masks(reference_masks)
    per_instance_results = []

    # Difficulty is the required target shift. It is used to relax Dice penalties
    # for harder counterfactual targets where larger anatomical changes are expected.
    target_difficulty = np.abs(y_org.astype(np.float64) - y_cf_goal.astype(np.float64))
    finite_difficulty = target_difficulty[np.isfinite(target_difficulty)]
    max_target_difficulty = float(np.max(finite_difficulty)) if finite_difficulty.size > 0 else 0.0

    roi_metric_names = [
        "inside_ratio_binary", "inside_ratio_weighted", "outside_ratio_weighted",
        "boundary_concentration", "mean_distance_to_boundary", "median_distance_to_boundary",
        "max_distance_to_boundary", "roi_precision", "roi_recall", "roi_f1",
        "inside_density", "outside_density", "density_ratio", "ring_density", "boundary_contrast",
    ]
    suffixes = ["strict", "relaxed"]

    for i in range(x_org.shape[0]):
        inst = {
            "instance_idx": int(i),
            "y_org": float(y_org[i]) if is_finite_scalar(y_org[i]) else np.nan,
            "y_pred": float(y_pred[i]) if is_finite_scalar(y_pred[i]) else np.nan,
            "y_cf_goal": float(y_cf_goal[i]) if is_finite_scalar(y_cf_goal[i]) else np.nan,
            "y_cf_pred": float(y_cf_pred[i]) if is_finite_scalar(y_cf_pred[i]) else np.nan,
        }
        inst.update(cf_regression_success_metrics(y_pred[i], y_cf_goal[i], y_cf_pred[i]))
        inst["abs_error_cf_goal_pred"] = float(abs(y_cf_pred[i] - y_cf_goal[i])) if is_finite_scalar(y_cf_pred[i]) and is_finite_scalar(y_cf_goal[i]) else np.nan
        inst["target_difficulty"] = float(target_difficulty[i]) if is_finite_scalar(target_difficulty[i]) else np.nan
        inst["target_difficulty_norm"] = safe_div(inst["target_difficulty"], max_target_difficulty) if max_target_difficulty > 1e-8 and is_finite_scalar(inst["target_difficulty"]) else 0.0
        inst["pixel_l1_minimality"] = pixel_l1_minimality(x_org[i], x_cf[i])

        org_pair_sorted = pair_functional_features_from_mask_pair(x_masks_org[i])
        cf_pair_sorted = pair_functional_features_from_mask_pair(x_masks_cf[i])
        for prefix, pair in [("org", org_pair_sorted), ("cf", cf_pair_sorted)]:
            inst[f"{prefix}_ed_idx_sorted"] = pair["ed_idx"]
            inst[f"{prefix}_es_idx_sorted"] = pair["es_idx"]
            inst[f"{prefix}_valid_pair_sorted"] = bool(pair["valid_pair"])
            inst[f"{prefix}_area_ed_sorted"] = pair["area_ed"]
            inst[f"{prefix}_area_es_sorted"] = pair["area_es"]
            inst[f"{prefix}_area_change_sorted"] = pair["area_change"]
            inst[f"{prefix}_fac_sorted"] = pair["fac"]

        inst["delta_area_ed_sorted"] = float(cf_pair_sorted["area_ed"] - org_pair_sorted["area_ed"]) if is_finite_scalar(org_pair_sorted["area_ed"]) and is_finite_scalar(cf_pair_sorted["area_ed"]) else np.nan
        inst["delta_area_es_sorted"] = float(cf_pair_sorted["area_es"] - org_pair_sorted["area_es"]) if is_finite_scalar(org_pair_sorted["area_es"]) and is_finite_scalar(cf_pair_sorted["area_es"]) else np.nan
        inst["delta_area_change_sorted"] = float(cf_pair_sorted["area_change"] - org_pair_sorted["area_change"]) if is_finite_scalar(org_pair_sorted["area_change"]) and is_finite_scalar(cf_pair_sorted["area_change"]) else np.nan
        inst["delta_fac_sorted"] = float(cf_pair_sorted["fac"] - org_pair_sorted["fac"]) if is_finite_scalar(org_pair_sorted["fac"]) and is_finite_scalar(cf_pair_sorted["fac"]) else np.nan

        if is_finite_scalar(inst["delta_fac_sorted"]):
            desired_delta_y = float(y_cf_goal[i] - y_pred[i])
            inst["fac_direction_match_to_goal_sorted"] = np.nan if np.sign(desired_delta_y) == 0 else float(np.sign(inst["delta_fac_sorted"]) == np.sign(desired_delta_y))
        else:
            inst["fac_direction_match_to_goal_sorted"] = np.nan

        inst.update(aligned_pair_features(x_masks_org[i], x_masks_cf[i]))
        if is_finite_scalar(inst["aligned_delta_fac"]):
            desired_delta_y = float(y_cf_goal[i] - y_pred[i])
            inst["fac_direction_match_to_goal_aligned"] = np.nan if np.sign(desired_delta_y) == 0 else float(np.sign(inst["aligned_delta_fac"]) == np.sign(desired_delta_y))
        else:
            inst["fac_direction_match_to_goal_aligned"] = np.nan

        inst["temporal_image_l1"] = float(np.mean(np.abs((sanitize_image(x_org[i, :, :, 0]) - sanitize_image(x_org[i, :, :, 1])) - (sanitize_image(x_cf[i, :, :, 0]) - sanitize_image(x_cf[i, :, :, 1])))))
        inst["temporal_mask_l1"] = temporal_mask_l1(x_masks_org[i], x_masks_cf[i])
        inst["temporal_mask_xor_change_ratio"] = temporal_mask_xor(x_masks_org[i], x_masks_cf[i])[0]

        frame_metric_lists = {f"{name}_{sfx}": [] for name in roi_metric_names for sfx in suffixes}
        frame_metric_lists.update({k: [] for k in ["dice_org_cf", "jaccard_org_cf", "delta_area", "delta_perimeter", "delta_circularity", "delta_aspect_ratio", "delta_solidity"]})

        for f in range(2):
            frame_name = f"frame{f}"
            img_org_f, img_cf_f = x_org[i, :, :, f], x_cf[i, :, :, f]
            mask_org_f, mask_cf_f = x_masks_org[i, :, :, f], x_masks_cf[i, :, :, f]
            regions = build_anatomical_regions(
                mask_org_f,
                roi_base_pad_iter=roi_base_pad_iter,
                roi_relax_pad_iter=roi_relax_pad_iter,
                ring_inner_iter=boundary_inner_iter,
                ring_outer_iter=boundary_outer_iter,
            )
            c_mask = binary_change_mask(img_org_f, img_cf_f, quantile=change_quantile)

            for label, roi, ring in [
                ("strict", regions["roi_strict"], regions["epsilon_ring_strict"]),
                ("relaxed", regions["roi_relaxed"], regions["epsilon_ring_relaxed"]),
            ]:
                basic = {
                    f"inside_ratio_binary_{label}": binary_inside_ratio(c_mask, roi),
                    f"inside_ratio_weighted_{label}": weighted_inside_ratio(img_org_f, img_cf_f, roi),
                    f"outside_ratio_weighted_{label}": weighted_outside_ratio(img_org_f, img_cf_f, roi),
                    f"boundary_concentration_{label}": weighted_inside_ratio(img_org_f, img_cf_f, ring),
                    f"inside_density_{label}": region_change_density(img_org_f, img_cf_f, roi),
                    f"outside_density_{label}": outside_region_change_density(img_org_f, img_cf_f, roi),
                    f"density_ratio_{label}": inside_outside_density_ratio(img_org_f, img_cf_f, roi),
                    f"ring_density_{label}": region_change_density(img_org_f, img_cf_f, ring),
                    f"boundary_contrast_{label}": boundary_contrast(img_org_f, img_cf_f, ring),
                }
                dist_stats = distance_of_changes_to_boundary(c_mask, roi)
                basic[f"mean_distance_to_boundary_{label}"] = dist_stats["mean_distance_to_boundary"]
                basic[f"median_distance_to_boundary_{label}"] = dist_stats["median_distance_to_boundary"]
                basic[f"max_distance_to_boundary_{label}"] = dist_stats["max_distance_to_boundary"]

                prf = roi_precision_recall_f1(c_mask, roi)
                basic[f"roi_precision_{label}"] = prf["roi_precision"]
                basic[f"roi_recall_{label}"] = prf["roi_recall"]
                basic[f"roi_f1_{label}"] = prf["roi_f1"]

                for k, v in basic.items():
                    inst[f"{frame_name}_{k}"] = v
                    frame_metric_lists[k].append(v)

            shape_stats = compare_shape_features(mask_org_f, mask_cf_f)
            shape_basic = {
                "dice_org_cf": shape_stats["dice_org_cf"],
                "jaccard_org_cf": shape_stats["jaccard_org_cf"],
                "delta_area": shape_stats["delta_area"],
                "delta_perimeter": shape_stats["delta_perimeter"],
                "delta_circularity": shape_stats["delta_circularity"],
                "delta_aspect_ratio": shape_stats["delta_aspect_ratio"],
                "delta_solidity": shape_stats["delta_solidity"],
            }
            for k, v in shape_basic.items():
                inst[f"{frame_name}_{k}"] = v
                frame_metric_lists[k].append(v)

        for metric_name, vals in frame_metric_lists.items():
            arr = np.asarray(vals, dtype=np.float64)
            inst[f"pair_mean_{metric_name}"] = float(np.nanmean(arr)) if np.any(np.isfinite(arr)) else np.nan

        # Difficulty-weighted Dice: high is better. For hard targets, the Dice penalty
        # is relaxed because larger anatomical changes are expected.
        if is_finite_scalar(inst.get("pair_mean_dice_org_cf")) and is_finite_scalar(inst.get("target_difficulty_norm")):
            w = float(np.clip(inst["target_difficulty_norm"], 0.0, 1.0))
            inst["difficulty_weighted_dice"] = float(1.0 - (1.0 - inst["pair_mean_dice_org_cf"]) * (1.0 - w))
        else:
            inst["difficulty_weighted_dice"] = np.nan

        inst.update(plausibility_against_reference(cf_pair_sorted, reference_stats))

        plaus_penalty = inst["reference_mean_abs_z"] if inst["reference_mean_abs_z"] is not None else None
        temporal_penalty = float(inst["temporal_mask_l1"] + inst["temporal_mask_xor_change_ratio"]) if is_finite_scalar(inst["temporal_mask_l1"]) and is_finite_scalar(inst["temporal_mask_xor_change_ratio"]) else None
        inst["cf_specificity_score"] = counterfactual_specificity_score(
            target_success_soft=inst["target_success_soft"],
            inside_lv_ratio=inst["pair_mean_inside_ratio_weighted_strict"],
            outside_change_ratio=inst["pair_mean_outside_ratio_weighted_strict"],
            boundary_ratio=inst["pair_mean_boundary_concentration_strict"],
            temporal_penalty=temporal_penalty,
            plausibility_penalty=plaus_penalty,
        )
        per_instance_results.append(inst)

    all_keys = set().union(*(r.keys() for r in per_instance_results))
    aggregate_mean, aggregate_std = {}, {}
    for key in sorted(all_keys):
        vals = [float(r[key]) for r in per_instance_results if key in r and isinstance(r[key], (int, float, np.integer, np.floating)) and np.isfinite(r[key])]
        aggregate_mean[key] = float(np.mean(vals)) if vals else None
        aggregate_std[key] = float(np.std(vals)) if vals else None

    # Additional aggregate-only success metrics used for paper plots/tables.
    aggregate_mean["mae_cf_goal_pred"] = float(np.mean(np.abs(y_cf_pred - y_cf_goal)))
    aggregate_std["mae_cf_goal_pred"] = None
    aggregate_mean["spearman_goal_pred"] = spearman_corr_safe(y_cf_goal, y_cf_pred)
    aggregate_std["spearman_goal_pred"] = None

    # Optional externally computed validity metrics, preserved in the JSON if supplied.
    optional_metrics = {
        "cf_validity_mae": cf_validity_mae,
        "cf_validity_rmse": cf_validity_rmse,
        "cf_validity_spearman": cf_validity_spearman,
        "cf_validity_mae_ef": cf_validity_mae_ef,
        "cf_validity_rmse_ef": cf_validity_rmse_ef,
        "cf_validity_spearman_ef": cf_validity_spearman_ef,
    }
    for k, v in optional_metrics.items():
        aggregate_mean[k] = float(v) if is_finite_scalar(v) else None
        aggregate_std[k] = None

    return {"per_instance": per_instance_results, "aggregate_mean": aggregate_mean, "aggregate_std": aggregate_std}


# ============================================================
# Reporting / serialization helpers
# ============================================================

def print_cf_metrics_summary(metrics_out, max_instances_to_show=5, key_groups=None, decimals=4):
    per_instance = metrics_out.get("per_instance", [])
    aggregate_mean = metrics_out.get("aggregate_mean", {})
    aggregate_std = metrics_out.get("aggregate_std", {})
    if key_groups is None:
        key_groups = {
            "Target achievement": ["target_success_soft", "target_success_hard", "relative_improvement_to_goal", "direction_match"],
            "Spatial strict": [
                "pair_mean_inside_ratio_weighted_strict", "pair_mean_outside_ratio_weighted_strict",
                "pair_mean_boundary_concentration_strict", "pair_mean_roi_precision_strict",
                "pair_mean_mean_distance_to_boundary_strict", "pair_mean_boundary_contrast_strict",
                "difficulty_weighted_dice", "pixel_l1_minimality", "mae_cf_goal_pred",
            ],
            "Spatial relaxed": [
                "pair_mean_inside_ratio_weighted_relaxed", "pair_mean_outside_ratio_weighted_relaxed",
                "pair_mean_boundary_concentration_relaxed", "pair_mean_roi_precision_relaxed",
                "pair_mean_mean_distance_to_boundary_relaxed", "pair_mean_boundary_contrast_relaxed",
                "pair_mean_ring_density_relaxed",
            ],
            "Temporal consistency": ["order_preserved_wrt_org", "aligned_ed_dice", "aligned_es_dice", "temporal_image_l1", "temporal_mask_l1", "temporal_mask_xor_change_ratio"],
            "Reference plausibility": ["reference_mean_abs_z", "reference_max_abs_z", "reference_mahalanobis_distance"],
            "Overall": ["cf_specificity_score"],
        }

    def _fmt(v):
        if v is None:
            return "None"
        if isinstance(v, (int, float, np.integer, np.floating)):
            return f"{float(v):.{decimals}f}" if np.isfinite(v) else "nan"
        return str(v)

    print("=" * 80)
    print("COUNTERFACTUAL METRICS SUMMARY")
    print("=" * 80)
    print(f"Number of instances: {len(per_instance)}\n")
    print("AGGREGATE RESULTS (mean ± std)")
    print("-" * 80)
    for group_name, keys in key_groups.items():
        print(f"[{group_name}]")
        for key in keys:
            print(f"  {key:40s}: {_fmt(aggregate_mean.get(key))} ± {_fmt(aggregate_std.get(key))}")
        print()


def make_cf_metrics_latex_table(metrics_out, selected_metrics=None, metric_display_names=None, caption="Counterfactual evaluation results (mean $\\pm$ standard deviation).", label="tab:cf_metrics", decimals=3):
    aggregate_mean = metrics_out.get("aggregate_mean", {})
    aggregate_std = metrics_out.get("aggregate_std", {})
    if selected_metrics is None:
        selected_metrics = [
            "target_success_soft",
            "pair_mean_inside_ratio_weighted_strict",
            "pair_mean_outside_ratio_weighted_strict",
            "pair_mean_boundary_concentration_strict",
            "pair_mean_roi_precision_strict",
            "pair_mean_density_ratio_strict",
            "pair_mean_ring_density_strict",
            "order_preserved_wrt_org",
            "temporal_mask_l1",
            "temporal_mask_xor_change_ratio",
            "reference_mean_abs_z",
            "cf_specificity_score",
        ]
    if metric_display_names is None:
        metric_display_names = {
            "target_success_soft": "Target success (soft)",
            "pair_mean_inside_ratio_weighted_strict": "Inside-ROI change ratio (strict)",
            "pair_mean_outside_ratio_weighted_strict": "Outside-ROI change ratio (strict)",
            "pair_mean_boundary_concentration_strict": "Boundary concentration (strict)",
            "pair_mean_roi_precision_strict": "ROI precision (strict)",
            "pair_mean_density_ratio_strict": "Inside/outside density ratio (strict)",
            "pair_mean_ring_density_strict": "Ring density (strict)",
            "order_preserved_wrt_org": "Order preserved",
            "temporal_mask_l1": "Temporal mask $L_1$",
            "temporal_mask_xor_change_ratio": "Temporal XOR change ratio",
            "reference_mean_abs_z": "Reference mean $|z|$",
            "cf_specificity_score": "CF specificity score",
        }

    def fmt(v):
        if v is None:
            return "--"
        if isinstance(v, (int, float, np.integer, np.floating)):
            return f"{float(v):.{decimals}f}" if np.isfinite(v) else "--"
        return "--"

    def latex_escape(s):
        return s.replace('_', '\\_').replace('%', '\\%').replace('&', '\\&')

    lines = ["\\begin{table}[t]", "\\centering", "\\begin{tabular}{l c}", "\\hline", "Metric & Mean $\\pm$ Std.\\\\", "\\hline"]
    for key in selected_metrics:
        display_name = latex_escape(metric_display_names.get(key, key))
        lines.append(f"{display_name} & {fmt(aggregate_mean.get(key))} $\\pm$ {fmt(aggregate_std.get(key))}\\\\")
    lines += ["\\hline", "\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table}"]
    return "\n".join(lines)


def save_cf_metrics(metrics_out, save_dir, file_prefix="cf_metrics"):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    json_path = save_dir / f"{file_prefix}.json"
    per_instance_csv_path = save_dir / f"{file_prefix}_per_instance.csv"
    aggregate_csv_path = save_dir / f"{file_prefix}_aggregate.csv"

    def to_serializable(obj):
        if isinstance(obj, dict):
            return {str(k): to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        if isinstance(obj, tuple):
            return [to_serializable(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.floating, float)):
            return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
        return obj

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(metrics_out), f, indent=2, ensure_ascii=False)

    pd.DataFrame(metrics_out.get("per_instance", [])).to_csv(per_instance_csv_path, index=False)
    metric_names = sorted(set(metrics_out.get("aggregate_mean", {}).keys()) | set(metrics_out.get("aggregate_std", {}).keys()))
    pd.DataFrame([{ "metric": m, "mean": metrics_out["aggregate_mean"].get(m), "std": metrics_out["aggregate_std"].get(m)} for m in metric_names]).to_csv(aggregate_csv_path, index=False)
    return {"json_path": str(json_path), "per_instance_csv_path": str(per_instance_csv_path), "aggregate_csv_path": str(aggregate_csv_path)}


def load_cf_metrics(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def debug_plot_org_rec_cf_pairs(
    x_data,
    y_data,
    x_masks,
    y_cf_pred,
    instance_indices=None,
    save_dir=None,
    show=True,
    cmap_img="gray",
    cmap_diff="bwr",
    diff_abs=False,
    contour_colors=None,
    figsize_per_instance=(18, 7),
    dpi=150,
):
    """
    Debug plot for all instances (or a subset), showing:
    rows   = 2 frames (EDV / ESV positions in the array)
    cols   = [Org, Rec, CF, Rec-CF, Org-CF]

    Parameters
    ----------
    x_data : tuple
        (x_org, x_rec, x_cf), each shape (N, H, W, 2)

    y_data : tuple
        (y_org, y_pred, y_cf_goal), each shape (N,)

    x_masks : tuple
        (x_masks_org, x_masks_rec, x_masks_cf), each shape (N, H, W, 2)

    y_cf_pred : np.ndarray
        Shape (N,), model predictions on counterfactuals

    instance_indices : list[int] or None
        If None, iterate over all N instances.

    save_dir : str or None
        If given, saves one figure per instance into this directory.

    show : bool
        If True, displays each figure.
        If False, only saves figures (if save_dir is not None).

    cmap_img : str
        Colormap for grayscale ultrasound images.

    cmap_diff : str
        Colormap for difference images.

    diff_abs : bool
        If True, plot absolute differences |rec-cf| and |org-cf|.
        If False, plot signed differences (rec-cf) and (org-cf).

    contour_colors : dict or None
        Optional mapping for contour colors.
        Default:
            {
                "org": "lime",
                "rec": "cyan",
                "cf": "yellow",
                "org_on_diff": "lime",
            }

    figsize_per_instance : tuple
        Figure size for each instance.

    dpi : int
        DPI used when saving.
    """

    def _mask_volume(mask):
        mask = np.asarray(mask)
        mask = np.nan_to_num(mask, nan=0.0)
        return int((mask > 0).sum())

    x_org, x_rec, x_cf = x_data
    y_org, y_pred, y_cf_goal = y_data
    x_masks_org, x_masks_rec, x_masks_cf = x_masks

    x_org = np.asarray(x_org)
    x_rec = np.asarray(x_rec)
    x_cf = np.asarray(x_cf)

    x_masks_org = np.asarray(x_masks_org)
    x_masks_rec = np.asarray(x_masks_rec)
    x_masks_cf = np.asarray(x_masks_cf)

    y_org = np.asarray(y_org)
    y_pred = np.asarray(y_pred)
    y_cf_goal = np.asarray(y_cf_goal)
    y_cf_pred = np.asarray(y_cf_pred)

    if contour_colors is None:
        contour_colors = {
            "org": "lime",
            "rec": "cyan",
            "cf": "yellow",
            "org_on_diff": "lime",
        }

    if x_org.ndim != 4 or x_org.shape[-1] != 2:
        raise ValueError(f"Expected x_org shape (N,H,W,2), got {x_org.shape}")
    if x_rec.shape != x_org.shape or x_cf.shape != x_org.shape:
        raise ValueError("x_org, x_rec, x_cf must have the same shape.")
    if x_masks_org.shape != x_org.shape or x_masks_rec.shape != x_org.shape or x_masks_cf.shape != x_org.shape:
        raise ValueError("All mask arrays must have the same shape as the image arrays.")
    if len(y_org) != x_org.shape[0] or len(y_pred) != x_org.shape[0] or len(y_cf_goal) != x_org.shape[0] or len(y_cf_pred) != x_org.shape[0]:
        raise ValueError("All y arrays must have length N.")

    N = x_org.shape[0]

    if instance_indices is None:
        instance_indices = list(range(N))
    else:
        instance_indices = list(instance_indices)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    def _sanitize_image(img):
        img = np.asarray(img, dtype=np.float32)
        return np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    def _sanitize_mask(mask):
        mask = np.asarray(mask, dtype=np.float32)
        mask = np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
        return mask > 0.0

    def _boundary(mask):
        mask = _sanitize_mask(mask)
        if mask.sum() == 0:
            return np.zeros_like(mask, dtype=bool)
        eroded = ndimage.binary_erosion(mask)
        return mask ^ eroded

    def _overlay_contour(ax, mask, color="lime", s=2):
        b = _boundary(mask)
        ys, xs = np.where(b)
        if len(xs) > 0:
            ax.scatter(xs, ys, s=s, c=color, linewidths=0)

    def _plot_img(ax, img, title, cmap, vmin=None, vmax=None):
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    for i in instance_indices:
        fig, axes = plt.subplots(2, 5, figsize=figsize_per_instance, squeeze=False)

        # global symmetric scaling for difference maps within this instance
        rec_cf_diff = _sanitize_image(x_rec[i]) - _sanitize_image(x_cf[i])
        org_cf_diff = _sanitize_image(x_org[i]) - _sanitize_image(x_cf[i])

        if diff_abs:
            rec_cf_vis = np.abs(rec_cf_diff)
            org_cf_vis = np.abs(org_cf_diff)
            diff_vmin, diff_vmax = None, None
        else:
            rec_cf_vis = rec_cf_diff
            org_cf_vis = org_cf_diff
            max_abs = max(
                float(np.max(np.abs(rec_cf_diff))),
                float(np.max(np.abs(org_cf_diff))),
                1e-8,
            )
            diff_vmin, diff_vmax = -max_abs, max_abs

        for f in range(2):
            img_org_f = _sanitize_image(x_org[i, :, :, f])
            img_rec_f = _sanitize_image(x_rec[i, :, :, f])
            img_cf_f = _sanitize_image(x_cf[i, :, :, f])

            mask_org_f = x_masks_org[i, :, :, f]
            mask_rec_f = x_masks_rec[i, :, :, f]
            mask_cf_f = x_masks_cf[i, :, :, f]

            # 1) Original
            vol_org = _mask_volume(mask_org_f)

            _plot_img(
                axes[f, 0],
                img_org_f,
                title=f"Org | EF={float(y_org[i]):.3f} | V={vol_org}",
                cmap=cmap_img,
            )
            _overlay_contour(axes[f, 0], mask_org_f, color=contour_colors["org"])

            # 2) Reconstruction
            vol_rec = _mask_volume(mask_rec_f)

            _plot_img(
                axes[f, 1],
                img_rec_f,
                title=f"Rec | pred={float(y_pred[i]):.3f} | V={vol_rec}",
                cmap=cmap_img,
            )
            _overlay_contour(axes[f, 1], mask_rec_f, color=contour_colors["rec"])

            # 3) Counterfactual
            vol_cf = _mask_volume(mask_cf_f)

            _plot_img(
                axes[f, 2],
                img_cf_f,
                title=f"CF | goal={float(y_cf_goal[i]):.3f}, pred={float(y_cf_pred[i]):.3f} | V={vol_cf}",
                cmap=cmap_img,
            )
            _overlay_contour(axes[f, 2], mask_cf_f, color=contour_colors["cf"])

            # 4) Reconstruction - Counterfactual
            _plot_img(
                axes[f, 3],
                rec_cf_vis[:, :, f],
                title="rec-CF",
                cmap=cmap_diff,
                vmin=diff_vmin,
                vmax=diff_vmax,
            )
            _overlay_contour(axes[f, 3], mask_org_f, color=contour_colors["org_on_diff"])

            # 5) Original - Counterfactual
            _plot_img(
                axes[f, 4],
                org_cf_vis[:, :, f],
                title="org-CF",
                cmap=cmap_diff,
                vmin=diff_vmin,
                vmax=diff_vmax,
            )
            _overlay_contour(axes[f, 4], mask_org_f, color=contour_colors["org_on_diff"])

            # row labels
            axes[f, 0].set_ylabel(f"frame {f}", fontsize=11)

        fig.suptitle(f"Instance {i}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save_dir is not None:
            out_path = save_dir+f"_instance_{i:04d}.png"
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)
# ============================================================
# Main metrics computation
# ============================================================

ANATOMY_CORE_METRICS = [
    "pair_mean_inside_ratio_weighted_strict",     # higher = better
    "pair_mean_inside_ratio_weighted_relaxed",    # higher = better
    "pair_mean_outside_ratio_weighted_strict",    # lower = better
    "pair_mean_outside_ratio_weighted_relaxed",   # lower = better
    "pair_mean_boundary_concentration_strict",    # higher = better
    "pair_mean_boundary_concentration_relaxed",   # higher = better
    "pair_mean_mean_distance_to_boundary_strict", # lower = better
    "pair_mean_mean_distance_to_boundary_relaxed",# lower = better
    "pair_mean_roi_precision_strict",             # higher = better
    "pair_mean_roi_precision_relaxed",            # higher = better
    "pair_mean_density_ratio_strict",             # higher = better
    "pair_mean_ring_density_strict",              # higher = better
    "pair_mean_boundary_contrast_strict",          # higher = better
    "pair_mean_dice_org_cf",                      # higher = better
    "difficulty_weighted_dice",                   # higher = better
    "pixel_l1_minimality",                        # lower = better
    "pair_mean_delta_area",                       # closer to 0 = better
    "order_preserved_wrt_org",                    # higher = better
    "temporal_mask_l1",                           # lower = better
    "temporal_mask_xor_change_ratio",             # lower = better
]


# ============================================================
# Debug visualization helpers
# ============================================================

def _overlay_mask(ax, mask, color="lime", alpha=0.35):
    m = sanitize_mask(mask)
    if m.sum() == 0:
        return
    overlay = np.zeros((*m.shape, 4), dtype=np.float32)
    color_map = {
        "lime": (0.0, 1.0, 0.0), "red": (1.0, 0.0, 0.0), "blue": (0.0, 0.0, 1.0),
        "yellow": (1.0, 1.0, 0.0), "cyan": (0.0, 1.0, 1.0), "magenta": (1.0, 0.0, 1.0),
        "orange": (1.0, 0.5, 0.0),
    }
    rgb = color_map.get(color, (0.0, 1.0, 0.0))
    overlay[..., 0], overlay[..., 1], overlay[..., 2], overlay[..., 3] = rgb[0], rgb[1], rgb[2], m.astype(np.float32) * alpha
    ax.imshow(overlay)


def _overlay_contour(ax, mask, color="lime"):
    b = boundary(mask)
    ys, xs = np.where(b)
    if len(xs):
        ax.scatter(xs, ys, s=2, c=color, linewidths=0)


def _plot_base_image(ax, img, title="", cmap="gray"):
    ax.imshow(sanitize_image(img), cmap=cmap)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def debug_show_roi_and_ring(
    x_data,
    x_masks,
    instance_indices=None,
    n_examples=4,
    roi_base_pad_iter=0,
    roi_relax_pad_iter=2,
    ring_inner_iter=1,
    ring_outer_iter=3,
):
    x_org, _, x_cf = x_data
    x_masks_org, _, x_masks_cf = x_masks
    N = x_org.shape[0]
    instance_indices = list(range(min(n_examples, N))) if instance_indices is None else list(instance_indices)[:n_examples]
    for inst_idx in instance_indices:
        fig, axes = plt.subplots(2, 7, figsize=(21, 6))
        fig.suptitle(
            f"Instance {inst_idx} | roi_base_pad={roi_base_pad_iter}, roi_relax_pad={roi_relax_pad_iter}, "
            f"ring_inner={ring_inner_iter}, ring_outer={ring_outer_iter}",
            fontsize=14
        )
        for f in range(2):
            regions = build_anatomical_regions(
                x_masks_org[inst_idx, :, :, f],
                roi_base_pad_iter=roi_base_pad_iter,
                roi_relax_pad_iter=roi_relax_pad_iter,
                ring_inner_iter=ring_inner_iter,
                ring_outer_iter=ring_outer_iter,
            )
            _plot_base_image(axes[f, 0], x_org[inst_idx, :, :, f], f"Org f{f} + ROI strict")
            _overlay_mask(axes[f, 0], regions["roi_strict"], color="lime", alpha=0.25)
            _plot_base_image(axes[f, 1], x_cf[inst_idx, :, :, f], f"CF f{f} + ROI strict")
            _overlay_mask(axes[f, 1], regions["roi_strict"], color="lime", alpha=0.25)
            _plot_base_image(axes[f, 2], x_org[inst_idx, :, :, f], f"Org f{f} + ROI relaxed")
            _overlay_mask(axes[f, 2], regions["roi_relaxed"], color="cyan", alpha=0.25)
            _plot_base_image(axes[f, 3], x_org[inst_idx, :, :, f], f"Org f{f} + Eps Ring strict")
            _overlay_mask(axes[f, 3], regions["epsilon_ring_strict"], color="yellow", alpha=0.35)
            _plot_base_image(axes[f, 4], x_org[inst_idx, :, :, f], f"Org f{f} + Eps Ring relaxed")
            _overlay_mask(axes[f, 4], regions["epsilon_ring_relaxed"], color="orange", alpha=0.35)
            _plot_base_image(axes[f, 5], x_org[inst_idx, :, :, f], f"Org f{f} + Inner Ring strict")
            _overlay_mask(axes[f, 5], regions["inner_ring_strict"], color="magenta", alpha=0.35)
            _plot_base_image(axes[f, 6], x_org[inst_idx, :, :, f], f"Org f{f} + Outer Ring strict")
            _overlay_mask(axes[f, 6], regions["outer_ring_strict"], color="red", alpha=0.35)
        plt.tight_layout()
        plt.show()


def debug_anatomy_core_metrics(
    x_data,
    x_masks,
    metrics_out,
    instance_indices=None,
    n_examples=4,
    metric_names=None,
    change_quantile=0.90,
    roi_base_pad_iter=0,
    roi_relax_pad_iter=2,
    boundary_inner_iter=1,
    boundary_outer_iter=3,
    figsize=(14, 8),
    cmap_img="gray",
):
    x_org, _, x_cf = x_data
    x_masks_org, _, x_masks_cf = x_masks
    per_instance = metrics_out["per_instance"]
    metric_names = ANATOMY_CORE_METRICS if metric_names is None else metric_names
    instance_indices = list(range(min(n_examples, len(per_instance)))) if instance_indices is None else list(instance_indices)[:n_examples]

    def draw_img(ax, img, title="", contour=None, overlay=None, overlay_color="lime", cmap=None):
        ax.imshow(sanitize_image(img), cmap=cmap or cmap_img)
        if contour is not None:
            _overlay_contour(ax, contour)
        if overlay is not None:
            _overlay_mask(ax, overlay, color=overlay_color)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    def prepare_frame_data(inst_idx):
        out = []
        for f in range(2):
            org = x_org[inst_idx, :, :, f]
            cf = x_cf[inst_idx, :, :, f]
            morg = x_masks_org[inst_idx, :, :, f]
            mcf = x_masks_cf[inst_idx, :, :, f]
            regions = build_anatomical_regions(
                morg,
                roi_base_pad_iter=roi_base_pad_iter,
                roi_relax_pad_iter=roi_relax_pad_iter,
                ring_inner_iter=boundary_inner_iter,
                ring_outer_iter=boundary_outer_iter,
            )
            cm = binary_change_mask(org, cf, quantile=change_quantile)
            strict_dist = distance_of_changes_to_boundary(cm, regions["roi_strict"])
            relaxed_dist = distance_of_changes_to_boundary(cm, regions["roi_relaxed"])
            out.append({
                "org": org, "cf": cf, "morg": morg, "mcf": mcf,
                "regions": regions,
                "cmask": cm, "cmap": abs_change_map(org, cf),
                "inside_strict": weighted_inside_ratio(org, cf, regions["roi_strict"]),
                "inside_relaxed": weighted_inside_ratio(org, cf, regions["roi_relaxed"]),
                "outside_strict": weighted_outside_ratio(org, cf, regions["roi_strict"]),
                "outside_relaxed": weighted_outside_ratio(org, cf, regions["roi_relaxed"]),
                "precision_strict": roi_precision_recall_f1(cm, regions["roi_strict"])["roi_precision"],
                "precision_relaxed": roi_precision_recall_f1(cm, regions["roi_relaxed"])["roi_precision"],
                "boundary_conc_strict": weighted_inside_ratio(org, cf, regions["epsilon_ring_strict"]),
                "boundary_conc_relaxed": weighted_inside_ratio(org, cf, regions["epsilon_ring_relaxed"]),
                "mean_dist_strict": strict_dist["mean_distance_to_boundary"],
                "mean_dist_relaxed": relaxed_dist["mean_distance_to_boundary"],
                "inside_density_strict": region_change_density(org, cf, regions["roi_strict"]),
                "inside_density_relaxed": region_change_density(org, cf, regions["roi_relaxed"]),
                "outside_density_strict": outside_region_change_density(org, cf, regions["roi_strict"]),
                "outside_density_relaxed": outside_region_change_density(org, cf, regions["roi_relaxed"]),
                "density_ratio_strict": inside_outside_density_ratio(org, cf, regions["roi_strict"]),
                "density_ratio_relaxed": inside_outside_density_ratio(org, cf, regions["roi_relaxed"]),
                "ring_density_strict": region_change_density(org, cf, regions["epsilon_ring_strict"]),
                "ring_density_relaxed": region_change_density(org, cf, regions["epsilon_ring_relaxed"]),
                "boundary_contrast_strict": boundary_contrast(org, cf, regions["epsilon_ring_strict"]),
                "boundary_contrast_relaxed": boundary_contrast(org, cf, regions["epsilon_ring_relaxed"]),
                "dice": dice(morg, mcf),
                "delta_area": mask_area(mcf) - mask_area(morg),
                "strict_dist_map": strict_dist["dist_map"],
                "relaxed_dist_map": relaxed_dist["dist_map"],
            })
        return out

    layout_map = {
        "pair_mean_inside_ratio_weighted_strict": "inside_strict",
        "pair_mean_inside_ratio_weighted_relaxed": "inside_relaxed",
        "pair_mean_outside_ratio_weighted_strict": "outside_strict",
        "pair_mean_outside_ratio_weighted_relaxed": "outside_relaxed",
        "pair_mean_boundary_concentration_strict": "boundary_strict",
        "pair_mean_boundary_concentration_relaxed": "boundary_relaxed",
        "pair_mean_mean_distance_to_boundary_strict": "distance_strict",
        "pair_mean_mean_distance_to_boundary_relaxed": "distance_relaxed",
        "pair_mean_roi_precision_strict": "precision_strict",
        "pair_mean_roi_precision_relaxed": "precision_relaxed",
        "pair_mean_density_ratio_strict": "density_strict",
        "pair_mean_density_ratio_relaxed": "density_relaxed",
        "pair_mean_ring_density_strict": "ring_density_strict",
        "pair_mean_ring_density_relaxed": "ring_density_relaxed",
        "pair_mean_boundary_contrast_strict": "boundary_contrast_strict",
        "pair_mean_boundary_contrast_relaxed": "boundary_contrast_relaxed",
        "pair_mean_dice_org_cf": "dice",
        "pair_mean_delta_area": "delta_area",
        "order_preserved_wrt_org": "order",
        "temporal_mask_l1": "temporal_l1",
        "temporal_mask_xor_change_ratio": "temporal_xor",
    }

    for metric_name in metric_names:
        fig, axes = plt.subplots(len(instance_indices), 6, figsize=(figsize[0], max(3, 2.8 * len(instance_indices))), squeeze=False)
        fig.suptitle(f"Debugging metric: {metric_name}", fontsize=15, y=0.995)
        layout = layout_map.get(metric_name, "inside_strict")

        for r, inst_idx in enumerate(instance_indices):
            frame_data = prepare_frame_data(inst_idx)
            row_metrics = per_instance[inst_idx]
            order_info = infer_ed_es_indices_from_masks(x_masks_org[inst_idx])
            org_ed_idx, org_es_idx = order_info["ed_idx"], order_info["es_idx"]
            order_preserved = row_metrics.get("order_preserved_wrt_org", np.nan)
            t_l1 = temporal_mask_l1(x_masks_org[inst_idx], x_masks_cf[inst_idx])
            xor_val, xor_org, xor_cf = temporal_mask_xor(x_masks_org[inst_idx], x_masks_cf[inst_idx])
            ax_row = axes[r]

            if layout in {"inside_strict", "inside_relaxed", "outside_strict", "outside_relaxed", "precision_strict", "precision_relaxed", "boundary_strict", "boundary_relaxed", "distance_strict", "distance_relaxed", "density_strict", "density_relaxed", "ring_density_strict", "ring_density_relaxed", "boundary_contrast_strict", "boundary_contrast_relaxed"}:
                suffix = "strict" if "strict" in layout else "relaxed"
                for f in range(2):
                    d = frame_data[f]
                    roi = d["regions"][f"roi_{suffix}"]
                    ring = d["regions"][f"epsilon_ring_{suffix}"]
                    draw_img(ax_row[f*3+0], d["org"], f"Org f{f}", contour=d["morg"])
                    if layout.startswith("inside"):
                        draw_img(ax_row[f*3+1], d["cf"], f"CF f{f}\ninside={d[f'inside_{suffix}']:.3f}", contour=d["mcf"])
                        draw_img(ax_row[f*3+2], d["cmap"], f"Change+ROI f{f}", overlay=roi, overlay_color="lime", cmap="hot")
                    elif layout.startswith("outside"):
                        outside_mask = np.logical_and(d["cmask"], ~sanitize_mask(roi))
                        draw_img(ax_row[f*3+1], d["cf"], f"CF f{f}\noutside={d[f'outside_{suffix}']:.3f}", contour=d["mcf"])
                        draw_img(ax_row[f*3+2], d["cmap"], f"Outside changes f{f}", overlay=outside_mask, overlay_color="red", cmap="hot")
                        _overlay_mask(ax_row[f*3+2], roi, color="lime", alpha=0.12)
                    elif layout.startswith("precision"):
                        draw_img(ax_row[f*3+1], d["cf"], f"CF f{f}\nprecision={d[f'precision_{suffix}']:.3f}", overlay=roi, overlay_color="lime")
                        draw_img(ax_row[f*3+2], d["cf"], f"ROI+changes f{f}", overlay=roi, overlay_color="lime")
                        _overlay_mask(ax_row[f*3+2], d["cmask"], color="red", alpha=0.5)
                    elif layout.startswith("boundary"):
                        draw_img(ax_row[f*3+1], d["cf"], f"CF f{f}\nboundary={d[f'boundary_conc_{suffix}']:.3f}", contour=d["mcf"])
                        draw_img(ax_row[f*3+2], d["cmap"], f"Change+ring f{f}", overlay=ring, overlay_color="yellow", cmap="hot")
                    elif layout.startswith("distance"):
                        draw_img(ax_row[f*3+1], d["cf"], f"CF f{f}\nmean_dist={d[f'mean_dist_{suffix}']:.3f}", contour=d["mcf"])
                        dist_map = d[f"{suffix}_dist_map"]
                        draw_img(ax_row[f*3+2], np.zeros_like(d["org"]) if dist_map is None else dist_map, f"Distance map f{f}", overlay=d["cmask"] if dist_map is not None else None, overlay_color="red", cmap="viridis")
                    elif layout.startswith("density"):
                        draw_img(ax_row[f*3+1], d["cf"], f"CF f{f}\ndensity ratio={d[f'density_ratio_{suffix}']:.3f}", contour=d["mcf"])
                        draw_img(ax_row[f*3+2], d["cmap"], f"in={d[f'inside_density_{suffix}']:.3f}\nout={d[f'outside_density_{suffix}']:.3f}", overlay=roi, overlay_color="lime", cmap="hot")
                    elif layout.startswith("ring_density"):
                        draw_img(ax_row[f*3+1], d["cf"], f"CF f{f}\nring dens={d[f'ring_density_{suffix}']:.3f}", contour=d["mcf"])
                        draw_img(ax_row[f*3+2], d["cmap"], f"Ring density f{f}", overlay=ring, overlay_color="yellow", cmap="hot")
                    elif layout.startswith("boundary_contrast"):
                        draw_img(ax_row[f*3+1], d["cf"], f"CF f{f}\nboundary contrast={d[f'boundary_contrast_{suffix}']:.3f}", contour=d["mcf"])
                        draw_img(ax_row[f*3+2], d["cmap"], f"Boundary contrast f{f}", overlay=ring, overlay_color="yellow", cmap="hot")

            elif layout == "dice":
                for f in range(2):
                    d = frame_data[f]
                    draw_img(ax_row[f*3+0], d["org"], f"Org f{f}", contour=d["morg"])
                    draw_img(ax_row[f*3+1], d["cf"], f"CF f{f}\ndice={d['dice']:.3f}", contour=d["mcf"])
                    draw_img(ax_row[f*3+2], d["cf"], f"Mask overlap f{f}")
                    _overlay_mask(ax_row[f*3+2], d["morg"], color="lime", alpha=0.25)
                    _overlay_mask(ax_row[f*3+2], d["mcf"], color="cyan", alpha=0.25)

            elif layout == "delta_area":
                for f in range(2):
                    d = frame_data[f]
                    draw_img(ax_row[f*3+0], d["org"], f"Org f{f}\narea={mask_area(d['morg']):.1f}", contour=d["morg"])
                    draw_img(ax_row[f*3+1], d["cf"], f"CF f{f}\narea={mask_area(d['mcf']):.1f}", contour=d["mcf"])
                    draw_img(ax_row[f*3+2], d["cf"], f"Δarea f{f}={d['delta_area']:.1f}")
                    _overlay_mask(ax_row[f*3+2], d["morg"], color="lime", alpha=0.25)
                    _overlay_mask(ax_row[f*3+2], d["mcf"], color="red", alpha=0.25)

            elif layout == "order":
                for f in range(2):
                    draw_img(ax_row[f*3+0], x_org[inst_idx, :, :, f], f"Org f{f}\narea={mask_area(x_masks_org[inst_idx, :, :, f]):.1f}", contour=x_masks_org[inst_idx, :, :, f])
                    draw_img(ax_row[f*3+1], x_cf[inst_idx, :, :, f], f"CF f{f}\narea={mask_area(x_masks_cf[inst_idx, :, :, f]):.1f}", contour=x_masks_cf[inst_idx, :, :, f])
                    title = f"ED idx={org_ed_idx}, ES idx={org_es_idx}\norder={order_preserved:.0f}" if (f == 0 and org_ed_idx is not None) else "Original temporal order"
                    draw_img(ax_row[f*3+2], x_cf[inst_idx, :, :, f], title, contour=x_masks_cf[inst_idx, :, :, f])

            elif layout == "temporal_l1":
                delta_org = sanitize_mask(x_masks_org[inst_idx, :, :, 0]).astype(float) - sanitize_mask(x_masks_org[inst_idx, :, :, 1]).astype(float)
                delta_cf = sanitize_mask(x_masks_cf[inst_idx, :, :, 0]).astype(float) - sanitize_mask(x_masks_cf[inst_idx, :, :, 1]).astype(float)
                diff = np.abs(delta_org - delta_cf)
                for f in range(2):
                    draw_img(ax_row[f*3+0], x_org[inst_idx, :, :, f], f"Org f{f}", contour=x_masks_org[inst_idx, :, :, f])
                    draw_img(ax_row[f*3+1], x_cf[inst_idx, :, :, f], f"CF f{f}", contour=x_masks_cf[inst_idx, :, :, f])
                    draw_img(ax_row[f*3+2], diff, f"Temporal mask diff\nL1={t_l1:.3f}", cmap="magma")

            elif layout == "temporal_xor":
                for f in range(2):
                    draw_img(ax_row[f*3+0], x_org[inst_idx, :, :, f], f"Org f{f}", contour=x_masks_org[inst_idx, :, :, f])
                    draw_img(ax_row[f*3+1], x_cf[inst_idx, :, :, f], f"CF f{f}", contour=x_masks_cf[inst_idx, :, :, f])
                    draw_img(ax_row[f*3+2], (xor_org if f == 0 else xor_cf).astype(float), "XOR org" if f == 0 else f"XOR cf\nratio={xor_val:.3f}", cmap="gray")

            val = row_metrics.get(metric_name)
            label = f"Instance {inst_idx}" if not isinstance(val, (int, float, np.integer, np.floating)) or not np.isfinite(val) else f"Instance {inst_idx} | {metric_name}={val:.3f}"
            axes[r, 0].set_ylabel(label, fontsize=9)

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.show()
