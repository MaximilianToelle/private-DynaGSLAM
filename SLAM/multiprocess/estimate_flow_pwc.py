import torch
import numpy as np
from torch.nn import functional as F
import sys
import os
import cv2
import time
from PIL import Image
import matplotlib.pyplot as plt
from typing import Optional

from .motion_models.pwc.pwc_net import Network
from .motion_models.utils import flow_viz

def writeFlowFile(filename, uv):
    TAG_STRING = np.array(202021.25, dtype=np.float32)
    if uv.shape[2] != 2:
        sys.exit("writeFlowFile: flow must have two bands!")
    H = np.array(uv.shape[0], dtype=np.int32)
    W = np.array(uv.shape[1], dtype=np.int32)
    with open(filename, 'wb') as f:
        f.write(TAG_STRING.tobytes())
        f.write(W.tobytes())
        f.write(H.tobytes())
        f.write(uv.tobytes())

def add_gaussian_noise(flow: np.ndarray, sigma: float = 2.0, seed: Optional[int] = None) -> np.ndarray:
    """
    Add per-component Gaussian noise N(0, sigma^2) to a (H, W, 2) optical flow array.
    """
    assert flow.ndim == 3 and flow.shape[2] == 2, "flow must be (H, W, 2)"
    rng = np.random.default_rng(seed)
    flow_f = flow.astype(np.float32, copy=True)
    noise = rng.normal(loc=0.0, scale=sigma, size=flow_f.shape).astype(np.float32)
    return flow_f + noise
        
def add_sparse_gaussian_noise(flow: np.ndarray, sigma: float = 2.0, p: float = 0.3, seed: Optional[int] = None) -> np.ndarray:
    """
    Add Gaussian noise N(0, sigma^2) to a random p-fraction of *pixels*.
    If a pixel is chosen, both (u,v) components are noised.
    """
    assert flow.ndim == 3 and flow.shape[2] == 2, f"flow must be (H,W,2), got {flow.shape}"
    rng = np.random.default_rng(seed)
    flow_f = flow.astype(np.float32, copy=True)

    H, W, C = flow_f.shape  # C==2
    noise = rng.normal(0.0, sigma, size=(H, W, C)).astype(np.float32)
    mask = (rng.random((H, W)) < p)[..., None]           # (H, W, 1)
    flow_f += noise * mask.astype(np.float32)            # broadcast to (H, W, 2)
    return flow_f


# -------- RAFT caching (so we don't reload every call) --------
_RAFT_MODEL = None
def _get_raft():
    global _RAFT_MODEL
    if _RAFT_MODEL is None:
        model = torch.nn.DataParallel(RAFT())
        ckpt = '//home/blark/Desktop/2024/qcomm/dynagslam/SLAM/multiprocess/motion_models/ckpts/raft-things.pth'
        state = torch.load(ckpt, map_location='cuda')
        model.load_state_dict(state)
        _RAFT_MODEL = model.module.to('cuda').eval()
        for p in _RAFT_MODEL.parameters():
            p.requires_grad = False
    return _RAFT_MODEL


#-----------PWC
pwc_net = Network().cuda().train(False)


# -------- small utilities --------
def _remap(flow, x, y):
    """Sample 2-channel flow at (x,y) using bilinear interpolation (OpenCV)."""
    fx = cv2.remap(flow[..., 0], x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    fy = cv2.remap(flow[..., 1], x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return np.stack([fx, fy], axis=-1)

def _forward_backward_consistency(f12, f21, tau=1.0):
    """Return reliability mask where || f12(x) + f21(x + f12(x)) || < tau."""
    H, W = f12.shape[:2]
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    x2 = xs + f12[..., 0]
    y2 = ys + f12[..., 1]
    valid = (x2 >= 0) & (x2 <= (W - 1)) & (y2 >= 0) & (y2 <= (H - 1))
    f21_w = _remap(f21, x2, y2)
    fb = np.sqrt((f12[..., 0] + f21_w[..., 0])**2 + (f12[..., 1] + f21_w[..., 1])**2)
    rel = (fb < tau) & valid
    return rel.astype(np.uint8)  # {0,1}

def _fit_affine_flow_ransac(u, v, rel_mask, max_iters=1000, eps=1.0, min_inliers_frac=0.3, rng=np.random):
    """
    Fit affine flow field:
        u(x,y) = a*x + b*y + c
        v(x,y) = d*x + e*y + f
    using RANSAC on reliable pixels. Returns (coeffs 6,), inlier_mask.
    """
    H, W = u.shape
    ys, xs = np.nonzero(rel_mask > 0)  # int arrays
    if len(xs) < 50:  # not enough reliable pixels
        return None, np.zeros_like(u, dtype=np.uint8)

    # Build LS from a set of indices; keep int for indexing and float for A
    def solve_from_indices(idx):
        xi = xs[idx].astype(np.int64)  # int for indexing
        yi = ys[idx].astype(np.int64)
        x = xi.astype(np.float32)      # float for model matrix
        y = yi.astype(np.float32)
        # [a b c  0 0 0] [x y 1]^T -> u
        # [0 0 0  d e f] [x y 1]^T -> v
        A = np.stack([
            np.stack([x, y, np.ones_like(x), np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)], axis=1),
            np.stack([np.zeros_like(x), np.zeros_like(x), np.zeros_like(x), x, y, np.ones_like(x)], axis=1)
        ], axis=1).reshape(-1, 6)
        b = np.stack([u[yi, xi], v[yi, xi]], axis=1).reshape(-1)  # integer indexing here
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            return coeffs
        except np.linalg.LinAlgError:
            return None

    best_inliers = None
    best_coeffs = None
    N = len(xs)
    sample_size = min(6, N)  # use >=3 points; 6 is stable

    for _ in range(max_iters):
        idx = rng.choice(N, size=sample_size, replace=False)
        coeffs = solve_from_indices(idx)
        if coeffs is None:
            continue
        a, b, c, d, e, f = coeffs

        # Predict on ALL reliable pixels
        xi_all = xs.astype(np.int64)
        yi_all = ys.astype(np.int64)
        x_all = xi_all.astype(np.float32)
        y_all = yi_all.astype(np.float32)
        u_hat = a * x_all + b * y_all + c
        v_hat = d * x_all + e * y_all + f

        res = np.sqrt((u[yi_all, xi_all] - u_hat)**2 + (v[yi_all, xi_all] - v_hat)**2)
        inliers = res < eps

        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_coeffs = coeffs
            if inliers.mean() > max(min_inliers_frac, 0.8):
                break

    if best_inliers is None or best_inliers.sum() < 20:
        return None, np.zeros_like(u, dtype=np.uint8)

    # Refit on inliers
    in_idx = np.where(best_inliers)[0]
    final_coeffs = solve_from_indices(in_idx)

    inlier_map = np.zeros_like(u, dtype=np.uint8)
    inlier_map[ys[in_idx], xs[in_idx]] = 1
    return final_coeffs, inlier_map


def _affine_predict(coeffs, H, W):
    a, b, c, d, e, f = coeffs
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    u_hat = a * xs + b * ys + c
    v_hat = d * xs + e * ys + f
    return u_hat, v_hat


def estimate_flow(frame1, frame2, frame_id):  # returns (flow_up, mask_bool)
    # -------------------- RAFT inference --------------------
    image1 = frame1.to('cuda').permute(2, 0, 1).unsqueeze(0) #* 255.0
    image2 = frame2.to('cuda').permute(2, 0, 1).unsqueeze(0) #* 255.0
    with torch.no_grad():
        image1 = torch.nn.functional.interpolate(input=image1, size=(512, 640), mode='bilinear', align_corners=False)
        image2 = torch.nn.functional.interpolate(input=image2, size=(512, 640), mode='bilinear', align_corners=False)
        f12 = pwc_net(image1, image2)  # 1->2
        f21 = pwc_net(image2, image1)  # 2->1

    #flow_up = f12  # keep your original return
    flo12 = f12[0].permute(1, 2, 0)#.detach().cpu().numpy().astype(np.float32)
    flo21 = f21[0].permute(1, 2, 0)#.detach().cpu().numpy().astype(np.float32)
    flo12 = torch.nn.functional.interpolate(input=flo12.permute(2,0,1).unsqueeze(0), size=(480,640), mode='bilinear')
    flo21 = torch.nn.functional.interpolate(input=flo21.permute(2,0,1).unsqueeze(0), size=(480,640), mode='bilinear')
    flow_up = flo12
    flo12 = (flo12.squeeze(0)).permute(1,2,0).detach().cpu().numpy().astype(np.float32)
    flo21 = (flo21.squeeze(0)).permute(1,2,0).detach().cpu().numpy().astype(np.float32)
    H, W = flo12.shape[:2]
    #flo12 = add_gaussian_noise(flo12, sigma=1.0, seed=42)
    #flo21 = add_gaussian_noise(flo21, sigma=1.0, seed=42)
    #flo12 = add_sparse_gaussian_noise(flo12, sigma=2.0, p=0.3, seed=42)
    #flo21 = add_sparse_gaussian_noise(flo21, sigma=2.0, p=0.3, seed=42)
    flo_rgb = flow_viz.flow_to_image(flo12)  # uint8 RGB
    flo_save = flow_viz.flow_to_image(flo21)

    # -------------------- Robust dynamic segmentation --------------------
    # 0) knobs
    TAU_FB = 30.0           # stricter fwd-back consistency
    RANSAC_EPS = 0.8        # px residual for inliers
    MAD_FACTOR = 1.7        # residual > median + k*MAD
    CLOSE_K, OPEN_K = 7, 3
    ERODE_K = 2
    MIN_AREA = 300
    BORDER_BAND = 50        # px band to suppress around image edges
    REMOVE_BORDER_TOUCH = True
    EXPECTED_NUM_OBJECTS = 4
    SUPPRESS_BG_BY_INLIERS = True
    INLIER_OVERLAP_MAX = 30.0  # drop comps if >50% overlap with affine inliers

    # 1) forward-backward reliability
    rel = _forward_backward_consistency(flo12, flo21, tau=TAU_FB)  # {0,1}

    # 2) fit affine background on reliable pixels
    u, v = flo12[..., 0], flo12[..., 1]
    coeffs, inlier_map = _fit_affine_flow_ransac(
        u, v, rel, max_iters=1000, eps=RANSAC_EPS, min_inliers_frac=0.3
    )

    # 3) residual magnitude
    if coeffs is None:
        residual = np.sqrt(u*u + v*v)
    else:
        u_hat, v_hat = _affine_predict(coeffs, H, W)
        du, dv = u - u_hat, v - v_hat
        residual = np.sqrt(du*du + dv*dv)

    # 4) robust threshold on residuals (MAD) + reliability gate
    med = np.median(residual[rel > 0]) if rel.any() else np.median(residual)
    mad = np.median(np.abs(residual - med))
    scale = 1.4826 * mad if mad > 1e-6 else 1.0
    thr = med + MAD_FACTOR * scale
    dyn0 = ((residual > thr) & (rel > 0)).astype(np.uint8) * 255

    # 5) suppress a thin border band (kills corner leaks)
    if BORDER_BAND > 0:
        dyn0[:BORDER_BAND, :] = 0
        dyn0[-BORDER_BAND:, :] = 0
        dyn0[:, :BORDER_BAND] = 0
        dyn0[:, -BORDER_BAND:] = 0

    # 6) light morphology
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_K, CLOSE_K))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_K, OPEN_K))
    k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ERODE_K, ERODE_K))
    mask_u8 = cv2.morphologyEx(dyn0, cv2.MORPH_CLOSE, k_close, iterations=1)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN,  k_open,  iterations=1)

    # 7) component filtering: area, border-touch, background-inlier overlap, top-K
    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask_u8 > 0).astype(np.uint8), connectivity=8)
    cand = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < MIN_AREA:
            continue
        if REMOVE_BORDER_TOUCH and (x == 0 or y == 0 or x + w >= W - 1 or y + h >= H - 1):
            continue
        if SUPPRESS_BG_BY_INLIERS and coeffs is not None:
            comp = (labels == i)
            overlap = (comp & (inlier_map > 0)).sum() / float(comp.sum())
            if overlap > INLIER_OVERLAP_MAX:
                continue
        cand.append((area, i))

    # keep largest K components (boxes) — prevents any stray background blob
    cand.sort(reverse=True)
    if EXPECTED_NUM_OBJECTS is not None:
        cand = cand[:EXPECTED_NUM_OBJECTS]

    keep = np.zeros_like(mask_u8)
    for _, i in cand:
        keep[labels == i] = 255

    mask_tight = cv2.erode(keep, k_erode, iterations=1)

    # 8) diagnostic edges for saving (filename unchanged)
    resid_u8 = cv2.normalize(residual, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    edges = cv2.Canny(resid_u8, 60, 180)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    segmented = cv2.bitwise_and(flo_rgb, flo_rgb, mask=mask_tight)

    # -------- NEW: overlay exactly the selected components' centroids (≤ EXPECTED_NUM_OBJECTS) --------
    # Use `keep` (pre-erosion selection) to avoid splits; sort by area and label in that order.
    label_src = (keep > 0).astype(np.uint8)
    numK, labK, statsK, centsK = cv2.connectedComponentsWithStats(label_src, connectivity=8)

    # indices of components sorted by area (desc), skipping background 0
    idxs = list(range(1, numK))
    idxs.sort(key=lambda i: int(statsK[i, cv2.CC_STAT_AREA]), reverse=True)

    for disp_idx, i in enumerate(idxs, start=1):
        cx, cy = centsK[i]
        cxi, cyi = int(round(cx)), int(round(cy))
        cxi = max(0, min(W - 1, cxi))
        cyi = max(0, min(H - 1, cyi))
        cv2.circle(segmented, (cxi, cyi), 6, (0, 0, 255), 2)
        cv2.putText(segmented, f"{disp_idx}", (cxi + 8, cyi - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    # -----------------------------------------------------------------------------------------------

    # -------------------- Save (same paths/names) --------------------
    outdir = '//hdd2/output/omd/swinging_4_unconstrained/flowmap/frame_%04d' % (frame_id)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    cv2.imwrite(os.path.join(outdir, 'flow.png'), flo_save[:, :, [2, 1, 0]])
    cv2.imwrite(os.path.join(outdir, 'closed_edge.png'), closed_edges)
    cv2.imwrite(os.path.join(outdir, 'mask.png'), mask_tight)
    cv2.imwrite(os.path.join(outdir, 'segmented.png'), segmented)

    # -------------------- Return (same IO spec) --------------------
    mask_bool = torch.from_numpy(mask_tight > 0).to(flow_up.device).bool()
    return flow_up, mask_bool
