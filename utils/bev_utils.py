"""
Bird's Eye View (BEV) visualization utilities for LiDAR point cloud evaluation.

Inspired by ATPPNet (https://github.com/kaustabpal/ATPPNet), which visualizes GT and
predicted point clouds side-by-side from a bird's-eye-view perspective.  Unlike
ATPPNet's Open3D-based interactive viewer, this module writes static PNG images so
it works on HPC servers that have no display / GLFW support.

Typical pipeline
----------------
1. The model outputs a *normalised* range-view feature map  ``[C, H, W]``
   (channel 0 = range/depth).
2. ``rangeview_to_pointcloud`` unnormalises channel 0 and calls
   ``RangeProjection.back_project_range`` to recover 3-D (x, y, z) points.
3. ``render_bev_comparison`` renders GT (red) and predicted (blue) point clouds
   from above and saves a 3-panel PNG: GT | Predicted | Overlay.

BEV coordinate convention (KITTI)
----------------------------------
    World X  → forward (up in BEV image)
    World Y  → left    (left in BEV image)
    World Z  → up      (not shown)

Pixel coordinates in the BEV image::

    col  = round( (Y_world + bev_range) / resolution )
    row  = round( (bev_range - X_world) / resolution )   ← flip X so forward=up
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — safe on HPC with no display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from dataset.projection import RangeProjection


# ---------------------------------------------------------------------------
# Unnormalisation helpers
# ---------------------------------------------------------------------------

def unnormalize_range(range_norm: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Unnormalise a range (depth) map from feature space back to metres.

    Args:
        range_norm: ``(H, W)`` normalised range values (channel 0 of feature map).
        mean:       Normalisation mean for the range channel.
        std:        Normalisation std  for the range channel.

    Returns:
        ``(H, W)`` depth values in metres.  Pixels that were empty during
        projection will have values around ``(-1 - mean) / std`` after
        unnormalisation and will be close to or below zero.
    """
    return range_norm * std + mean


# ---------------------------------------------------------------------------
# Back-projection: normalised range view → 3-D point cloud
# ---------------------------------------------------------------------------

def rangeview_to_pointcloud(
    range_view_norm: np.ndarray,
    projector: RangeProjection,
    range_mean: float,
    range_std: float,
    min_depth: float = 0.5,
) -> np.ndarray:
    """Back-project a normalised range-view image to a 3-D point cloud.

    Args:
        range_view_norm: ``(H, W)`` normalised range channel (channel 0).
        projector:       :class:`RangeProjection` instance with FOV parameters.
        range_mean:      Normalisation mean for the range channel.
        range_std:       Normalisation std  for the range channel.
        min_depth:       Minimum valid depth (metres); shallower points dropped.

    Returns:
        ``(N, 3)`` float32 array of valid 3-D points ``(x, y, z)`` in metres.
        May be empty if no valid pixels exist.
    """
    depth = unnormalize_range(range_view_norm, range_mean, range_std)
    # Zero-out pixels below min_depth so back_project_range ignores them.
    depth[depth < min_depth] = 0.0
    return projector.back_project_range(depth)


# ---------------------------------------------------------------------------
# BEV pixel coordinate helpers
# ---------------------------------------------------------------------------

def _world_to_bev_pixels(
    points: np.ndarray,
    bev_range: float,
    resolution: float,
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map 3-D world coordinates to BEV pixel (col, row) indices.

    Args:
        points:     ``(N, 3)`` world points ``(x, y, z)``.
        bev_range:  BEV half-extent in metres.
        resolution: Metres per pixel.
        size:       BEV image side length in pixels.

    Returns:
        ``col_idx, row_idx`` — integer pixel coordinates; filtered to lie
        inside the image bounds.
    """
    if points.shape[0] == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    x = points[:, 0]
    y = points[:, 1]

    col = ((y + bev_range) / resolution).astype(np.int32)   # Y → columns
    row = ((bev_range - x) / resolution).astype(np.int32)   # X (fwd) → rows (flipped)

    valid = (col >= 0) & (col < size) & (row >= 0) & (row < size)
    return col[valid], row[valid]


def _make_bev_canvas(
    points: np.ndarray,
    color: tuple[int, int, int],
    bev_range: float,
    resolution: float,
    size: int,
) -> np.ndarray:
    """Paint one point cloud onto a blank BEV canvas.

    Args:
        points:     ``(N, 3)`` world points.
        color:      RGB colour tuple (0-255).
        bev_range:  BEV half-extent (m).
        resolution: Metres per pixel.
        size:       Canvas side length (pixels).

    Returns:
        ``(size, size, 3)`` uint8 canvas.
    """
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    col, row = _world_to_bev_pixels(points, bev_range, resolution, size)
    if col.size > 0:
        canvas[row, col] = color
    return canvas


# ---------------------------------------------------------------------------
# Axis-tick helpers
# ---------------------------------------------------------------------------

def _apply_bev_ticks(ax, bev_range: float, resolution: float, size: int, tick_step: float = 10.0):
    """Set labelled ticks on a BEV axes object.

    The horizontal axis represents Y (world left/right) and the vertical axis
    represents X (world forward/backward, flipped so forward is up).
    """
    ticks_world = np.arange(-bev_range, bev_range + 1e-6, tick_step)

    # Columns (X-axis of the image) → world Y
    col_ticks = ((ticks_world + bev_range) / resolution).astype(int)
    col_labels = [f'{v:.0f}' for v in ticks_world]

    # Rows (Y-axis of the image) → world X, but flipped
    row_ticks = ((bev_range - ticks_world) / resolution).astype(int)
    row_labels = [f'{v:.0f}' for v in ticks_world]   # X values (not flipped)

    ax.set_xticks(col_ticks)
    ax.set_xticklabels(col_labels, fontsize=7)
    ax.set_yticks(row_ticks)
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_xlabel('Y [m]  (left +)', fontsize=9)
    ax.set_ylabel('X [m]  (forward +)', fontsize=9)


# ---------------------------------------------------------------------------
# Public API — single overlay render
# ---------------------------------------------------------------------------

def render_bev(
    points_gt: np.ndarray,
    points_pred: np.ndarray,
    bev_range: float = 50.0,
    resolution: float = 0.2,
    output_path: str | None = None,
    title: str | None = None,
) -> np.ndarray:
    """Render a Bird's Eye View image overlaying GT (red) and predicted (blue).

    Args:
        points_gt:    ``(N, 3)`` ground-truth 3-D points in metres.
        points_pred:  ``(M, 3)`` predicted 3-D points in metres.
        bev_range:    BEV half-extent in metres; image covers
                      ``[-bev_range, bev_range]`` in both X and Y.
        resolution:   Metres per pixel (smaller → higher resolution).
        output_path:  If given, save the PNG to this path.
        title:        Optional figure title.

    Returns:
        ``(H, W, 3)`` uint8 RGB BEV image.
    """
    size = int(2 * bev_range / resolution)
    gt_canvas   = _make_bev_canvas(points_gt,   (220, 50,  50),  bev_range, resolution, size)
    pred_canvas = _make_bev_canvas(points_pred,  (50, 100, 220),  bev_range, resolution, size)

    # Overlay: GT takes priority where both exist
    overlay = np.zeros((size, size, 3), dtype=np.uint8)
    gt_mask   = np.any(gt_canvas   > 0, axis=2)
    pred_mask = np.any(pred_canvas > 0, axis=2)
    overlay[gt_mask]   = gt_canvas[gt_mask]
    overlay[pred_mask & ~gt_mask] = pred_canvas[pred_mask & ~gt_mask]

    if output_path is not None:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(overlay, origin='upper', aspect='equal')
        _apply_bev_ticks(ax, bev_range, resolution, size)
        gt_patch   = mpatches.Patch(color=(220/255, 50/255, 50/255),
                                    label=f'GT ({points_gt.shape[0]:,} pts)')
        pred_patch = mpatches.Patch(color=(50/255, 100/255, 220/255),
                                    label=f'Pred ({points_pred.shape[0]:,} pts)')
        ax.legend(handles=[gt_patch, pred_patch], loc='upper right', fontsize=10)
        if title:
            ax.set_title(title, fontsize=12)
        plt.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

    return overlay


# ---------------------------------------------------------------------------
# Public API — 3-panel comparison figure (GT | Predicted | Overlay)
# ---------------------------------------------------------------------------

def render_bev_comparison(
    points_gt: np.ndarray,
    points_pred: np.ndarray,
    bev_range: float = 50.0,
    resolution: float = 0.2,
    output_path: str | None = None,
    frame_idx: int = 0,
    metrics: dict | None = None,
) -> None:
    """Save a 3-panel BEV comparison PNG: GT | Predicted | Overlay.

    Inspired by the ATPPNet visualizer which renders GT (red) and predicted
    (blue) point clouds side-by-side using Open3D.  Here we use matplotlib
    so the output is a static PNG that can be produced on HPC without a
    display server.

    Panel layout::

        [ GT (red) ]  |  [ Predicted (blue) ]  |  [ Overlay ]
                                                   red  = GT only
                                                   blue = Pred only
                                                   purple = both agree

    Args:
        points_gt:    ``(N, 3)`` ground-truth 3-D points (metres).
        points_pred:  ``(M, 3)`` predicted 3-D points (metres).
        bev_range:    BEV half-extent (metres).
        resolution:   Metres per pixel.
        output_path:  Path for the saved PNG.  Parent dirs created automatically.
        frame_idx:    Frame number shown in the figure title.
        metrics:      Optional ``{name: value}`` dict of evaluation metrics
                      (e.g. ``{'chamfer_dist': 0.42, 'range_l1': 0.10}``).
    """
    size = int(2 * bev_range / resolution)

    gt_canvas   = _make_bev_canvas(points_gt,   (220, 50,  50),  bev_range, resolution, size)
    pred_canvas = _make_bev_canvas(points_pred,  (50, 100, 220),  bev_range, resolution, size)

    gt_mask   = np.any(gt_canvas   > 0, axis=2)
    pred_mask = np.any(pred_canvas > 0, axis=2)

    # Overlay: GT=red, Pred=blue, both=purple
    overlay = np.zeros((size, size, 3), dtype=np.uint8)
    overlay[gt_mask   & ~pred_mask] = (220, 50,  50)   # GT only  → red
    overlay[pred_mask & ~gt_mask]   = (50, 100, 220)   # Pred only → blue
    overlay[gt_mask   & pred_mask]  = (180, 50, 220)   # both      → purple

    panels = [
        (gt_canvas,   f'Ground Truth  ({points_gt.shape[0]:,} pts)'),
        (pred_canvas, f'Predicted  ({points_pred.shape[0]:,} pts)'),
        (overlay,     'Overlay  (red=GT, blue=Pred, purple=both)'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    for ax, (canvas, subtitle) in zip(axes, panels):
        ax.imshow(canvas, origin='upper', aspect='equal')
        ax.set_title(subtitle, fontsize=11)
        _apply_bev_ticks(ax, bev_range, resolution, size)

    # Build suptitle
    suptitle_parts = [f'Frame {frame_idx}  —  Bird\'s Eye View Comparison  '
                      f'(range {bev_range:.0f} m, res {resolution:.2f} m/px)']
    if metrics:
        suptitle_parts.append('   |   '.join(f'{k}: {v:.4f}' for k, v in metrics.items()))
    fig.suptitle('\n'.join(suptitle_parts), fontsize=12, y=1.02)

    plt.tight_layout()
    if output_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary plot — aggregate metrics across frames
# ---------------------------------------------------------------------------

def plot_metrics_summary(
    metrics_per_frame: list[dict],
    output_path: str,
    title: str = 'Per-frame Evaluation Metrics',
) -> None:
    """Plot per-frame metric curves and save as PNG.

    Args:
        metrics_per_frame: List of ``{metric_name: value}`` dicts, one per frame.
        output_path:       Path for the saved PNG.
        title:             Figure title.
    """
    if not metrics_per_frame:
        return

    keys = list(metrics_per_frame[0].keys())
    frames = list(range(len(metrics_per_frame)))

    n_metrics = len(keys)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4 * n_metrics), squeeze=False)

    for i, key in enumerate(keys):
        values = [m[key] for m in metrics_per_frame]
        ax = axes[i, 0]
        ax.plot(frames, values, marker='o', markersize=3, linewidth=1.2, label=key)
        mean_val = np.mean(values)
        ax.axhline(mean_val, color='red', linestyle='--', linewidth=1,
                   label=f'mean = {mean_val:.4f}')
        ax.set_xlabel('Frame index')
        ax.set_ylabel(key)
        ax.set_title(key)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
