"""
visualize_rangeview_3d.py — Interactive 3D point cloud viewer for RangeView DiT

Runs on a headless remote server and serves an interactive 3D viewer at
http://localhost:<PORT> via Open3D's WebRTC backend.  No display / GLFW window
is required.

Usage
-----
python scripts/visualize_rangeview_3d.py \\
    --config   configs/dit_config_rangeview.py \\
    --ckpt     /DATA2/shuhul/exp/ckpt/<exp>/rangeview_dit_NNNNN.pkl \\
    --vae_ckpt /DATA2/shuhul/exp/ckpt/<exp>/vae_pre_disc_stepNNNNN.pth  \\  # optional
    --split    val      \\   # 'val' or 'test'
    --sample   0        \\   # sample index in the chosen split
    --future   1        \\   # which forecast step to show (1 = t+1, …, N = t+N)
    --n_future 5        \\   # how many AR steps to run (max = config forward_iter)
    --all_steps         \\   # show all AR-chain steps at once (offset in Y)
    --port     8888     \\   # WebRTC HTTP server port
    --device   cuda

VAE checkpoint loading priority
--------------------------------
1. The main --ckpt file is always loaded first via load_parameters().
   If it contains 'vae_tokenizer.*' keys (e.g. from a stage='all' or
   stage='1' run), those weights win — they are the most up-to-date.
2. If --vae_ckpt is supplied, it is passed to RangeLDMVAE during model
   construction (before the main checkpoint is applied).  This ensures
   the VAE is never left at random weights even if the main checkpoint
   was saved without 'vae_tokenizer.*' keys (e.g. a stage='2' run where
   the VAE tokenizer keys were absent from the saved state dict).
3. If neither source provides VAE weights the script prints a loud warning
   so you know inference will produce garbage depth predictions.

SSH port-forwarding (to open from your laptop browser):
    ssh -L 8888:localhost:8888 user@remote-server
Then navigate to:
    http://localhost:8888

Color coding (matches bev_utils.py / train_rangeview.py):
    Ground truth   →  red   (220, 50,  50)  / 255
    Predicted      →  blue  (50, 100, 220)  / 255
    Overlap region →  purple shown only in BEV 2-D; in 3-D both clouds are present

When --all_steps is given every future step is rendered in its own GT+Pred pair,
offset by STEP_OFFSET_M metres along the Y axis so the AR chain can be inspected
as a spatial sequence in the same scene.
"""

import os
import sys
import argparse
import numpy as np
import torch

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.config_utils import Config
from utils.running import load_parameters
from utils.preprocess import get_rel_pose
from models.model_rangeview import RangeViewDiT
from dataset.dataset_kitti_rangeview import (
    KITTIRangeViewValDataset,
    KITTIRangeViewTestDataset,
)
from dataset.projection import RangeProjection

# ── Color coding — matches bev_utils.py render_bev_comparison ─────────────────
#
# Single-step default (dark background):
#   GT   → vivid crimson red   (220, 30,  30)  / 255
#   Pred → vivid cobalt blue   (30,  100, 230) / 255
#
# These are saturated, high-contrast colours chosen to pop clearly against
# the dark background.  They deliberately match the warm/cool convention
# used in bev_utils.py (GT=red, Pred=blue) while being much more vivid
# than the previous pastel-gradient approach.
GT_COLOR_FLAT   = (220 / 255, 30  / 255, 30  / 255)   # vivid crimson
PRED_COLOR_FLAT = (30  / 255, 100 / 255, 230 / 255)   # vivid cobalt

# Per-point depth colormaps used in --all_steps mode only.
# 'hot'    → black→red→orange→yellow  (all vivid on dark bg, warm = GT)
# 'winter' → blue→cyan                (all vivid on dark bg, cool = Pred)
GT_CMAP   = "hot"
PRED_CMAP = "winter"

# Depth range used to normalise the colormap (metres)
DEPTH_CMAP_MIN = 0.5
DEPTH_CMAP_MAX = 80.0

# Spacing between AR-chain steps when all steps are shown.
# Each step is displaced along the X axis (vehicle-forward direction) so the
# temporal sequence reads left-to-right in the scene like a spatial trajectory.
STEP_OFFSET_X = 120.0   # metres between consecutive step panels


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _unnorm_depth(norm_chw: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Return (H, W) depth map in metres from normalised channel-0 feature."""
    return norm_chw[0] * std + mean


def _backproject(depth_hw: np.ndarray, projector: RangeProjection,
                 min_depth: float = 0.5) -> np.ndarray:
    """Back-project a depth map (H, W) to an (N, 3) point cloud."""
    depth = depth_hw.copy()
    depth[depth < min_depth] = 0.0
    pts = projector.back_project_range(depth)
    return pts                      # (N, 3) float32


def _colorize_by_depth(pts: np.ndarray, cmap_name: str) -> np.ndarray:
    """Return per-point RGB colours from a matplotlib colormap keyed on depth.

    Depth (Euclidean distance from origin) is linearly mapped to [0, 1]
    between DEPTH_CMAP_MIN and DEPTH_CMAP_MAX.  Near points receive the
    *low* end of the colormap (brighter for 'Reds'/'Blues'), far points the
    *high* end (darker).  This gives clear depth-perception cues in 3-D.

    Args:
        pts:       (N, 3) float32 XYZ point cloud.
        cmap_name: matplotlib colormap name, e.g. 'Reds' or 'Blues'.

    Returns:
        (N, 3) float64 RGB colours in [0, 1].
    """
    import matplotlib.pyplot as plt
    depths = np.linalg.norm(pts, axis=1)                          # (N,)
    t = np.clip(
        (depths - DEPTH_CMAP_MIN) / (DEPTH_CMAP_MAX - DEPTH_CMAP_MIN),
        0.0, 1.0
    )
    cmap   = plt.get_cmap(cmap_name)
    colors = cmap(t)[:, :3]                                       # (N, 3) RGB
    return colors.astype(np.float64)


def _make_pcd(pts: np.ndarray, color_or_colors):
    """Wrap an (N, 3) array in an Open3D PointCloud.

    Args:
        pts:              (N, 3) float32 XYZ array.
        color_or_colors:  Either a 3-tuple (uniform colour) or an (N, 3)
                          float64 array of per-point colours.
    """
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    if isinstance(color_or_colors, np.ndarray) and color_or_colors.ndim == 2:
        pcd.colors = o3d.utility.Vector3dVector(color_or_colors)
    else:
        pcd.paint_uniform_color(list(color_or_colors))
    return pcd


def _offset_pcd(pcd, dx: float):
    """Return a copy of *pcd* shifted by *dx* metres along the X axis."""
    import open3d as o3d
    pcd2 = o3d.geometry.PointCloud(pcd)
    pcd2.translate([dx, 0, 0])
    return pcd2


def _axes_at(origin, size: float = 5.0):
    """Return a coordinate frame mesh anchored at *origin*."""
    import open3d as o3d
    return o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=size, origin=list(origin)
    )


def _ground_grid(center_x: float, half_extent: float = 60.0,
                 spacing: float = 10.0) -> "open3d.geometry.LineSet":
    """Return a flat ground grid (Z=0) centred at *center_x* on the X axis.

    The grid gives spatial scale reference in the browser viewer.
    Lines are drawn in a light grey so they do not obscure the point clouds.
    """
    import open3d as o3d
    x0 = center_x - half_extent
    x1 = center_x + half_extent
    y0, y1 = -half_extent, half_extent

    points, lines = [], []
    # Lines parallel to Y axis
    xs = np.arange(x0, x1 + spacing * 0.5, spacing)
    for x in xs:
        i = len(points)
        points += [[x, y0, 0.0], [x, y1, 0.0]]
        lines.append([i, i + 1])
    # Lines parallel to X axis
    ys = np.arange(y0, y1 + spacing * 0.5, spacing)
    for y in ys:
        i = len(points)
        points += [[x0, y, 0.0], [x1, y, 0.0]]
        lines.append([i, i + 1])

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    ls.paint_uniform_color([0.75, 0.75, 0.75])   # light grey
    return ls


def _step_sphere(x_offset: float, step_idx: int) -> "open3d.geometry.TriangleMesh":
    """Return a small coloured sphere marking the origin of step *step_idx*."""
    import open3d as o3d
    # Cycle through a set of distinct accent colours for each step
    accent_colors = [
        [0.9, 0.6, 0.0],   # amber   t+1
        [0.0, 0.7, 0.4],   # teal    t+2
        [0.7, 0.0, 0.9],   # violet  t+3
        [0.9, 0.3, 0.0],   # orange  t+4
        [0.0, 0.5, 0.9],   # sky     t+5
    ]
    color = accent_colors[step_idx % len(accent_colors)]
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=2.0)
    sphere.translate([x_offset, 0.0, 0.0])
    sphere.paint_uniform_color(color)
    sphere.compute_vertex_normals()
    return sphere


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(cfg, ckpt_path: str, device: str, vae_ckpt_path: str | None = None) -> RangeViewDiT:
    """Instantiate RangeViewDiT and load weights.

    Loading order
    -------------
    1. If *vae_ckpt_path* is given, ``cfg.vae_ckpt`` is set to that path
       **before** the model is constructed.  ``RangeLDMVAE.__init__`` picks it
       up and loads the pre-trained encoder/decoder weights immediately.
    2. The full model checkpoint at *ckpt_path* is then applied via
       ``load_parameters``.  Any ``vae_tokenizer.*`` keys present in that
       checkpoint will overwrite the weights from step 1 (they represent the
       latest jointly-trained or fine-tuned state).
    3. If the main checkpoint has no ``vae_tokenizer.*`` keys *and* no
       *vae_ckpt_path* was provided, the VAE stays at random weights and a
       loud warning is printed.

    Args:
        cfg:           Config namespace (mutated in-place to set vae_ckpt).
        ckpt_path:     Path to the main model .pkl checkpoint.
        device:        'cuda' or 'cpu'.
        vae_ckpt_path: Optional separate VAE checkpoint (.pth / .ckpt).
    """
    # ── Step 1: inject vae_ckpt into config before model construction ─────────
    if vae_ckpt_path is not None:
        if not os.path.isfile(vae_ckpt_path):
            raise FileNotFoundError(f"--vae_ckpt not found: {vae_ckpt_path}")
        print(f"VAE checkpoint supplied: {vae_ckpt_path}")
        cfg.vae_ckpt = vae_ckpt_path
    else:
        # Keep whatever is in the config (could already be a valid path or None).
        existing = getattr(cfg, "vae_ckpt", None)
        if existing:
            print(f"VAE checkpoint from config: {existing}")
        else:
            print(
                "INFO: No --vae_ckpt argument and config.vae_ckpt is None.\n"
                "      VAE will be initialised at random unless the main\n"
                "      checkpoint contains 'vae_tokenizer.*' keys."
            )

    # Batch-size 1 for inference; DeepSpeed not needed.
    cfg.batch_size = 1

    print(f"Instantiating RangeViewDiT …")
    model = RangeViewDiT(
        cfg,
        local_rank=0 if device.startswith("cuda") else -1,
        condition_frames=cfg.condition_frames // cfg.block_size,
    )

    # ── Step 2: load main checkpoint ──────────────────────────────────────────
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"--ckpt not found: {ckpt_path}")
    print(f"Loading main checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Detect whether VAE tokenizer weights are present in the checkpoint.
    raw_state = ckpt.get("model_state_dict", ckpt)
    vae_keys_in_ckpt = [
        k for k in raw_state
        if k.replace("module.", "", 1).startswith("vae_tokenizer.")
    ]
    if vae_keys_in_ckpt:
        print(
            f"  Main checkpoint contains {len(vae_keys_in_ckpt)} "
            "vae_tokenizer.* keys — VAE weights will be loaded from it."
        )
    else:
        if vae_ckpt_path is not None:
            print(
                "  Main checkpoint has NO vae_tokenizer.* keys.\n"
                f"  VAE weights come exclusively from --vae_ckpt: {vae_ckpt_path}"
            )
        else:
            print(
                "\n" + "!" * 60 + "\n"
                "  WARNING: main checkpoint has NO vae_tokenizer.* keys\n"
                "           AND no --vae_ckpt was provided.\n"
                "           The VAE decoder will use RANDOM weights.\n"
                "           Predicted depth maps will be GARBAGE.\n"
                "           Re-run with --vae_ckpt /path/to/vae.pth\n"
                "!" * 60 + "\n"
            )

    model = load_parameters(model, ckpt)
    model.eval()

    dev = torch.device(device)
    model = model.to(dev)
    print(f"Model ready on {device}.")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(cfg, split: str):
    """Return a KITTI RangeView dataset for the requested split."""
    split = split.lower()
    if split == "val":
        sequences = cfg.val_sequences
        DatasetCls = KITTIRangeViewValDataset
    elif split == "test":
        sequences = cfg.test_sequences
        DatasetCls = KITTIRangeViewTestDataset
    else:
        raise ValueError(f"--split must be 'val' or 'test', got '{split}'")

    dataset = DatasetCls(
        sequences_path=cfg.kitti_sequences_path,
        poses_path=cfg.kitti_poses_path,
        sequences=sequences,
        condition_frames=cfg.condition_frames,
        forward_iter=cfg.forward_iter,
        h=cfg.range_h,
        w=cfg.range_w,
        fov_up=cfg.fov_up,
        fov_down=cfg.fov_down,
        fov_left=cfg.fov_left,
        fov_right=cfg.fov_right,
        proj_img_mean=cfg.proj_img_mean,
        proj_img_stds=cfg.proj_img_stds,
        pc_extension=cfg.pc_extension,
        pc_dtype=getattr(np, cfg.pc_dtype),
        pc_reshape=tuple(cfg.pc_reshape),
    )
    print(f"Dataset [{split}]: {len(dataset)} samples (sequences {sequences})")
    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# Autoregressive inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model: RangeViewDiT, range_views: torch.Tensor,
                  poses: torch.Tensor, n_future: int, device: str):
    """Run autoregressive range-view forecasting.

    Args:
        model:       Loaded RangeViewDiT in eval mode.
        range_views: (T, C, H, W)  — full window from dataset.
        poses:       (T, 4, 4)     — absolute pose matrices.
        n_future:    How many future steps to predict (≤ forward_iter).
        device:      'cuda' / 'cpu'.

    Returns:
        pred_list: list of n_future (C, H, W) numpy arrays (predicted frames).
        gt_list:   list of n_future (C, H, W) numpy arrays (GT frames).
    """
    dev       = torch.device(device)
    cf        = model.condition_frames          # number of conditioning frames
    block_sz  = model.args.block_size

    pred_list = []
    gt_list   = []

    # Initial conditioning window (first CF frames)
    features_cond = range_views[:cf].unsqueeze(0).to(dev)   # [1, CF, C, H, W]

    for j in range(n_future):
        # Rotation matrices for this AR step: CF conditioning + 1 GT
        n_rot = (cf + 1) * block_sz
        rot_start = j * block_sz
        rot_mat_j = poses[rot_start: rot_start + n_rot].unsqueeze(0).to(dev)  # [1, n_rot, 4, 4]

        # GT frame for this step
        gt_frame = range_views[cf + j]       # (C, H, W)
        gt_list.append(gt_frame.numpy())

        # Dummy gt tensor (not used in eval; needed for forward signature)
        features_gt_dummy = range_views[cf + j: cf + j + 1].unsqueeze(0).to(dev)  # [1, 1, C, H, W]

        # Call model in eval mode — returns [1, C, H, W]
        model.eval()
        with torch.cuda.amp.autocast(enabled=False):
            rel_pose, rel_yaw = get_rel_pose(rot_mat_j)
        predicted = model.step_eval(features_cond, rel_pose, rel_yaw)  # [1, C, H, W]

        pred_frame = predicted[0].float().cpu().numpy()   # (C, H, W)
        pred_list.append(pred_frame)

        # Slide conditioning window: drop oldest, append prediction
        features_cond = torch.cat([
            features_cond[:, 1:, ...],                         # [1, CF-1, C, H, W]
            predicted.unsqueeze(1).float(),                    # [1, 1, C, H, W]
        ], dim=1)                                              # [1, CF, C, H, W]

    return pred_list, gt_list


# ─────────────────────────────────────────────────────────────────────────────
# Build Open3D scene
# ─────────────────────────────────────────────────────────────────────────────

def build_scene(pred_list, gt_list, projector: RangeProjection,
                range_mean: float, range_std: float,
                show_step: int, all_steps: bool):
    """Build a list of Open3D geometries from the AR-chain predictions.

    Layout
    ------
    *all_steps* (default):
        All N future steps are shown simultaneously, each pair (GT + Pred)
        placed at a different X-axis offset (STEP_OFFSET_X × step_index).
        The X axis is the vehicle forward direction, so the steps read as a
        temporal trajectory going left→right in the scene.

        Step i origin is at X = i × STEP_OFFSET_X metres.
        GT  (depth-coloured warm: Reds  colormap) and
        Pred (depth-coloured cool: Blues colormap) are drawn at the same
        X offset so they visually overlap — rotate the camera to compare.

    Single step (*all_steps* = False):
        Only step *show_step* (1-indexed) is rendered with no offset.

    Color scheme
    ------------
    *Single step* (all_steps=False, default):
        Flat uniform colours on a dark background for maximum contrast:
            GT   → vivid crimson (220, 30,  30)  — stands out immediately
            Pred → vivid cobalt  (30, 100, 230)  — clearly distinct from GT
        No depth gradient: the solid saturated colour is the clearest way
        to compare two overlapping point clouds in a browser 3-D viewer.

    *All steps* (all_steps=True):
        Per-point depth coloring with dark-friendly colormaps:
            GT   → 'hot'    (black→red→orange→yellow, all vivid on dark bg)
            Pred → 'winter' (blue→cyan, all vivid on dark bg)
        Depth gradient helps distinguish structure when viewing 5 dense
        clouds at once.

    A flat ground grid and a coloured origin sphere are added per step
    for spatial scale reference.

    Returns
    -------
    geometries : list of Open3D geometry objects ready for draw()
    scene_info : dict with 'n_steps' and 'center_x' for camera placement
    """
    geometries = []
    steps_to_show = list(range(len(pred_list))) if all_steps else [show_step - 1]

    x_offsets = []
    for i in steps_to_show:
        pred_chw = pred_list[i]
        gt_chw   = gt_list[i]

        # Unnormalise depth channel
        pred_depth = _unnorm_depth(pred_chw, range_mean, range_std)
        gt_depth   = _unnorm_depth(gt_chw,   range_mean, range_std)

        # Back-project to 3-D (N, 3)
        pts_gt   = _backproject(gt_depth,   projector)
        pts_pred = _backproject(pred_depth, projector)

        # X-axis offset: step i is at X = i × STEP_OFFSET_X
        dx = i * STEP_OFFSET_X if all_steps else 0.0
        x_offsets.append(dx)

        # ── Ground truth cloud ─────────────────────────────────────────────
        if pts_gt.shape[0] > 0:
            # Single-step: flat vivid colour — most readable in browser.
            # All-steps:   per-point depth colormap for structural detail.
            color_gt = (
                _colorize_by_depth(pts_gt, GT_CMAP)
                if all_steps else GT_COLOR_FLAT
            )
            pcd_gt = _make_pcd(pts_gt, color_gt)
            if dx != 0.0:
                pcd_gt = _offset_pcd(pcd_gt, dx)
            geometries.append(pcd_gt)

        # ── Predicted cloud ────────────────────────────────────────────────
        if pts_pred.shape[0] > 0:
            color_pred = (
                _colorize_by_depth(pts_pred, PRED_CMAP)
                if all_steps else PRED_COLOR_FLAT
            )
            pcd_pred = _make_pcd(pts_pred, color_pred)
            if dx != 0.0:
                pcd_pred = _offset_pcd(pcd_pred, dx)
            geometries.append(pcd_pred)

        # ── Step origin marker (coloured sphere) + coordinate frame ───────
        geometries.append(_step_sphere(dx, i))
        geometries.append(_axes_at([dx, 0.0, 0.0], size=4.0))

        # ── Ground grid centred on this step ──────────────────────────────
        geometries.append(_ground_grid(center_x=dx, half_extent=55.0, spacing=10.0))

        print(
            f"  t+{i+1}: GT={pts_gt.shape[0]:,} pts (Reds)  "
            f"Pred={pts_pred.shape[0]:,} pts (Blues)  "
            f"X-offset={dx:.0f} m"
        )

    center_x = float(np.mean(x_offsets)) if x_offsets else 0.0
    return geometries, {"n_steps": len(steps_to_show), "center_x": center_x}


# ─────────────────────────────────────────────────────────────────────────────
# WebRTC server launch
# ─────────────────────────────────────────────────────────────────────────────

def launch_webrtc_viewer(geometries: list, port: int, title: str,
                         point_size: float = 3.0,
                         lookat: list = None,
                         eye: list = None):
    """
    Start Open3D's WebRTC HTTP server and serve *geometries* interactively.

    The server blocks until the user closes the window or presses Ctrl-C.
    Open3D >= 0.13 is required.  Access the viewer at http://localhost:<port>.

    Installation
    ------------
    If Open3D is not installed yet::

        pip install open3d           # CPU-only build
        # or
        pip install open3d-cpu       # explicitly CPU-only

    The WebRTC backend is included in the standard PyPI wheel (>= 0.13).

    Port note
    ---------
    Open3D's built-in WebRTC server listens on port 8888 by default.  Some
    newer builds (>= 0.17) expose ``http_handshake_server_port`` as an argument
    to ``enable_webrtc``; older builds silently ignore it — the actual port is
    always 8888 in those cases regardless of what ``--port`` you pass.  Update
    your SSH tunnel accordingly.
    """
    import open3d as o3d

    # Check Open3D version
    ver_str = o3d.__version__
    ver = tuple(int(x) for x in ver_str.split(".")[:2])
    if ver < (0, 13):
        raise RuntimeError(
            f"Open3D >= 0.13 required for WebRTC support; found {ver_str}"
        )

    # Enable the WebRTC backend — must be called BEFORE draw().
    # Try to pass the port if the current build supports it; fall back
    # gracefully if it does not (the server will use 8888 regardless).
    webrtc_ok = False
    try:
        ws = o3d.visualization.webrtc_server
        try:
            ws.enable_webrtc(http_handshake_server_port=port)
        except TypeError:
            # Older API — no port argument
            ws.enable_webrtc()
        webrtc_ok = True
        actual_port = port
    except AttributeError:
        # Build does not include WebRTC — warn but continue (will fail on
        # truly headless servers; works fine if a display is available).
        actual_port = port
        print(
            "WARNING: open3d.visualization.webrtc_server not found.\n"
            "         Your Open3D build may not include WebRTC support.\n"
            "         Falling back to a standard (blocking) visualizer window.\n"
            "         If you are on a headless server this will likely fail.\n"
            "         Upgrade Open3D:  pip install open3d --upgrade"
        )

    if webrtc_ok:
        print("\n" + "=" * 60)
        print(f"  Open3D WebRTC server started  (Open3D {ver_str})")
        print(f"  Access the 3D viewer at:  http://localhost:{actual_port}")
        print("  If on a remote server, first open an SSH tunnel:")
        print(f"    ssh -L {actual_port}:localhost:{actual_port} <user>@<remote-host>")
        print("  Then open http://localhost:{:d} in your browser.".format(actual_port))
        print("=" * 60 + "\n")

    print(f"Launching viewer: '{title}'")
    print("Legend:")
    print("  Red  (vivid crimson)  = Ground Truth")
    print("  Blue (vivid cobalt)   = Predicted")
    print("  Dark background — saturated colours for maximum contrast")
    print("  Rotate:  left-drag   |  Pan: right-drag   |  Zoom: scroll")
    print("Press Ctrl-C to stop the server.\n")

    draw_kwargs = dict(
        title=title,
        width=1600,
        height=900,
        # show_ui=False gives the 3-D viewport the full browser window
        # (removes the right-hand scene-tree panel — fixes "2 windows" look).
        show_ui=False,
        point_size=point_size,
        # Near-black background: vivid red and blue pop with maximum contrast.
        bg_color=(0.05, 0.05, 0.08, 1.0),
        lookat=list(lookat),
        eye=list(eye),
        up=[0.0, 0.0, 1.0],
    )

    # Older Open3D versions may not accept all kwargs — strip unknowns if needed.
    try:
        o3d.visualization.draw(geometries, **draw_kwargs)
    except TypeError:
        # Minimal fallback
        o3d.visualization.draw(geometries, title=title)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize RangeView DiT point cloud predictions via Open3D WebRTC"
    )
    p.add_argument(
        "--config", required=True,
        help="Path to config file, e.g. configs/dit_config_rangeview.py"
    )
    p.add_argument(
        "--ckpt", required=True,
        help="Path to trained model checkpoint (.pkl), "
             "e.g. /DATA2/shuhul/exp/ckpt/<exp>/rangeview_dit_60000.pkl"
    )
    p.add_argument(
        "--vae_ckpt", default=None,
        help=(
            "Optional path to a separate VAE checkpoint (.pth / .ckpt).\n"
            "Use this when the main --ckpt was saved during Stage-2 training\n"
            "and may not contain 'vae_tokenizer.*' keys, so the VAE would\n"
            "otherwise be left at random weights.\n"
            "Example: /DATA2/shuhul/exp/ckpt/<exp>/vae_pre_disc_step50000.pth\n"
            "Loading priority: (1) --vae_ckpt at model construction, then\n"
            "(2) any vae_tokenizer.* keys found in --ckpt overwrite step 1."
        )
    )
    p.add_argument(
        "--split", default="val", choices=["val", "test"],
        help="Dataset split to use: 'val' (sequences 6-7) or 'test' (sequences 8-10). "
             "Default: val"
    )
    p.add_argument(
        "--sample", type=int, default=0,
        help="Index of the sample to visualise within the chosen split. Default: 0"
    )
    p.add_argument(
        "--future", type=int, default=1,
        help="Which forecast step to display (1 = t+1, 2 = t+2, …). "
             "Ignored when --all_steps is set. Default: 1"
    )
    p.add_argument(
        "--n_future", type=int, default=None,
        help="Number of autoregressive future steps to run. "
             "Defaults to forward_iter from the config."
    )
    p.add_argument(
        "--all_steps", action="store_true", default=False,
        help="Show all AR-chain steps simultaneously, each offset along the X axis. "
             "Off by default (single-step mode is the default). "
             "In all-steps mode depth-based 'hot'/'winter' colormaps replace the "
             "flat vivid colours."
    )
    p.add_argument(
        "--single_step", action="store_true",
        help="Explicitly force single-step mode (overrides --all_steps if both given)."
    )
    p.add_argument(
        "--point_size", type=float, default=3.0,
        help="Rendered point size in the Open3D viewer. Larger = clearer but slower. "
             "Default: 3.0  (try 4.0–6.0 if points still look too small)"
    )
    p.add_argument(
        "--port", type=int, default=8888,
        help="Port for the Open3D WebRTC HTTP server. Default: 8888"
    )
    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device ('cuda' or 'cpu'). Default: cuda if available"
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = Config.fromfile(args.config)

    # Defaults for batch_size / num_sampling_steps if absent
    if not hasattr(cfg, "batch_size"):
        cfg.batch_size = 1
    if not hasattr(cfg, "num_sampling_steps"):
        cfg.num_sampling_steps = 100

    n_future = args.n_future or cfg.forward_iter

    # --single_step overrides --all_steps
    show_all = args.all_steps and not args.single_step

    # Clamp future step index (only matters in single-step mode)
    future_step = max(1, min(args.future, n_future))

    # ── Projector (for back-projecting depth maps to 3-D) ─────────────────────
    projector = RangeProjection(
        fov_up=cfg.fov_up,
        fov_down=cfg.fov_down,
        fov_left=cfg.fov_left,
        fov_right=cfg.fov_right,
        proj_h=cfg.range_h,
        proj_w=cfg.range_w,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = load_model(cfg, args.ckpt, args.device, vae_ckpt_path=args.vae_ckpt)

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = load_dataset(cfg, args.split)

    if args.sample < 0 or args.sample >= len(dataset):
        raise IndexError(
            f"--sample {args.sample} is out of range "
            f"(dataset has {len(dataset)} samples)"
        )

    range_views, poses = dataset[args.sample]
    # range_views: (T, C, H, W)   T = condition_frames + forward_iter
    # poses:       (T, 4, 4)

    print(f"\nSample {args.sample} | split={args.split} | "
          f"range_views shape: {tuple(range_views.shape)} | "
          f"poses shape: {tuple(poses.shape)}")

    # ── Inference ─────────────────────────────────────────────────────────────
    print(f"\nRunning {n_future}-step autoregressive inference on {args.device}…")
    pred_list, gt_list = run_inference(
        model, range_views, poses, n_future, args.device
    )
    print(f"Inference done.  {len(pred_list)} predicted frames available.")

    # ── Build Open3D scene ────────────────────────────────────────────────────
    range_mean = cfg.proj_img_mean[0]
    range_std  = cfg.proj_img_stds[0]

    mode_str = f"all {n_future} AR steps" if show_all else f"step t+{future_step}"
    print(f"\nBuilding 3-D scene ({mode_str})…")

    geometries, scene_info = build_scene(
        pred_list=pred_list,
        gt_list=gt_list,
        projector=projector,
        range_mean=range_mean,
        range_std=range_std,
        show_step=future_step,
        all_steps=show_all,
    )

    # ── Camera placement ──────────────────────────────────────────────────────
    # For all-steps mode: look at the centre of the step row from above-left.
    # For single-step mode: look at origin from a close bird's-eye angle.
    cx = scene_info["center_x"]
    if show_all and n_future > 1:
        # Wide view: camera above-and-to-the-side, looking at the mid-step
        lookat = [cx,   0.0,  0.0]
        eye    = [cx, -160.0, 90.0]
    else:
        # Close view: single-step, more intimate perspective
        lookat = [0.0,   0.0,  0.0]
        eye    = [0.0, -80.0, 50.0]

    # ── Launch WebRTC viewer ──────────────────────────────────────────────────
    title = (
        f"RangeView DiT | {args.split} sample {args.sample} | {mode_str} | "
        "RED=GT  BLUE=Pred"
    )

    launch_webrtc_viewer(
        geometries,
        port=args.port,
        title=title,
        point_size=args.point_size,
        lookat=lookat,
        eye=eye,
    )


if __name__ == "__main__":
    main()
