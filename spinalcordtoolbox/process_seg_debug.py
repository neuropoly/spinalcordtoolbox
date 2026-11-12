"""
Debug plotting used by the `process_seg.py` script.

Copyright (c) 2025 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""

import os
import logging
import math
import numpy as np
import skimage

from spinalcordtoolbox.registration.algorithms import compute_pca
from spinalcordtoolbox.utils.sys import LazyLoader

plt = LazyLoader("plt", globals(), "matplotlib.pyplot")
mpl_backend_agg = LazyLoader("mpl_backend_agg", globals(), "matplotlib.backends.backend_agg")
mpl_figure = LazyLoader("mpl_figure", globals(), "matplotlib.figure")
mpl_patches = LazyLoader("mpl_patches", globals(), "matplotlib.patches")


def create_regularized_hog_angle_plot(
        z_indices_array, filter_size,
        angle_hog_array, angle_hog_regularized,
):
    plt.figure(figsize=(10, 6))
    plt.plot(z_indices_array, 180 * angle_hog_array / np.pi, 'ob', label='Original HOG angles')
    plt.plot(z_indices_array, 180 * angle_hog_regularized / np.pi, 'r', linewidth=2, label='Regularized HOG angles')
    plt.grid()
    plt.xlabel('z slice')
    plt.ylabel('Angle (deg)')
    plt.title(f"Regularized HOG angle estimation (filter_size: {filter_size})")
    plt.legend()
    fname_out = os.path.join('process_seg_regularize_hog_rotation.png')
    plt.savefig(fname_out, dpi=300)
    plt.close()
    logging.info(f"Saved regularized HOG angles visualization to: {fname_out}")


def create_quadrant_area_plots(
        # Core inputs (provided directly to `compute_quadrant_areas`)
        image_crop_r, centroid, orientation_deg, dim,
        # Derived inputs (calculated within `compute_quadrant_areas`)
        # Technically these are redundant since they can be derived from the core inputs,
        # but passing them in directly avoids duplicate calculations.
        ant_r_mask, ant_l_mask, post_r_mask, post_l_mask, quadrant_areas,
        # Debug-specific inputs (passed only for these plots)
        diameter_AP, diameter_RL, iz
):
    # simple transformations of the input
    y0, x0 = centroid
    orientation_rad = np.radians(orientation_deg)

    # Calculate AP and RL symmetry
    left_area = quadrant_areas.get('area_quadrant_anterior_left', 0) + quadrant_areas.get('area_quadrant_posterior_left', 0)
    right_area = quadrant_areas.get('area_quadrant_anterior_right', 0) + quadrant_areas.get('area_quadrant_posterior_right', 0)
    anterior_area = quadrant_areas.get('area_quadrant_anterior_left', 0) + quadrant_areas.get('area_quadrant_anterior_right', 0)
    posterior_area = quadrant_areas.get('area_quadrant_posterior_left', 0) + quadrant_areas.get('area_quadrant_posterior_right', 0)

    # Create masks for halves (combining quadrants)
    right_mask = post_r_mask | ant_r_mask  # Right half (posterior + anterior right)
    left_mask = post_l_mask | ant_l_mask   # Left half (posterior + anterior left)
    anterior_mask = ant_r_mask | ant_l_mask  # Anterior half (right + left anterior)
    posterior_mask = post_r_mask | post_l_mask  # Posterior half (right + left posterior)

    # Create figure with 1x3 subplots
    fig = mpl_figure.Figure(figsize=(18, 6))
    mpl_backend_agg.FigureCanvasAgg(fig)

    # ---------------------------------
    # Plot 1: Quadrants
    # ---------------------------------
    ax1 = fig.add_subplot(1, 3, 1)

    # Plot each quadrant mask with a different color
    ax1.imshow(np.where(post_r_mask, image_crop_r, np.nan), cmap='Reds', vmin=0, vmax=1, alpha=1)
    ax1.imshow(np.where(ant_r_mask, image_crop_r, np.nan), cmap='Blues', vmin=0, vmax=1, alpha=1)
    ax1.imshow(np.where(post_l_mask, image_crop_r, np.nan), cmap='Greens', vmin=0, vmax=1, alpha=1)
    ax1.imshow(np.where(ant_l_mask, image_crop_r, np.nan), cmap='Purples', vmin=0, vmax=1, alpha=1)

    ax1.imshow(image_crop_r > 0.5, cmap='gray', interpolation='nearest', vmin=0, vmax=1, alpha=.4)

    _add_diameter_lines(ax1, centroid, diameter_AP, diameter_RL, orientation_rad, dim, )
    _setup_axis(ax1, 'Quadrants')
    offset = 20  # pixel offset from centroid for annotation placement
    ax1.text(x0 - offset, y0 - offset, f"PR:\n{quadrant_areas['area_quadrant_posterior_right']:.2f} mm²", color='red',
             fontsize=10, ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    ax1.text(x0 + offset, y0 - offset, f"AR:\n{quadrant_areas['area_quadrant_anterior_right']:.2f} mm²", color='blue',
             fontsize=10, ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    ax1.text(x0 - offset, y0 + offset, f"PL:\n{quadrant_areas['area_quadrant_posterior_left']:.2f} mm²", color='green',
             fontsize=10, ha='center', va='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    ax1.text(x0 + offset, y0 + offset, f"AL:\n{quadrant_areas['area_quadrant_anterior_left']:.2f} mm²", color='purple',
             fontsize=10, ha='center', va='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    ax1.legend(loc='upper right')

    # ---------------------------------
    # Plot 2: Right-Left Symmetry
    # ---------------------------------
    ax2 = fig.add_subplot(1, 3, 2)

    # Plot each half with a different color
    ax2.imshow(np.where(right_mask, image_crop_r, np.nan), cmap='Reds', vmin=0, vmax=1, alpha=1, label='Right')
    ax2.imshow(np.where(left_mask, image_crop_r, np.nan), cmap='Blues', vmin=0, vmax=1, alpha=1, label='Left')
    ax2.imshow(image_crop_r > 0.5, cmap='gray', interpolation='nearest', vmin=0, vmax=1, alpha=.4)

    _add_diameter_lines(ax2, centroid, diameter_AP, diameter_RL, orientation_rad, dim)
    # _add_ellipse(ax2, centroid, diameter_AP, diameter_RL, orientation_rad, dim, upscale)
    _setup_axis(ax2, 'Right-Left Symmetry')
    ax2.text(x0, y0 - offset, f"Right:\n{right_area:.2f} mm²", color='red', fontsize=10, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    ax2.text(x0, y0 + offset, f"Left:\n{left_area:.2f} mm²", color='blue', fontsize=10, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # ---------------------------------
    # Plot 3: Anterior-Posterior Symmetry
    # ---------------------------------
    ax3 = fig.add_subplot(1, 3, 3)

    # Plot each half with a different color
    ax3.imshow(np.where(anterior_mask, image_crop_r, np.nan), cmap='Greens', vmin=0, vmax=1, alpha=1, label='Anterior')
    ax3.imshow(np.where(posterior_mask, image_crop_r, np.nan), cmap='Purples', vmin=0, vmax=1, alpha=1, label='Posterior')
    ax3.imshow(image_crop_r > 0.5, cmap='gray', interpolation='nearest', vmin=0, vmax=1, alpha=.4)

    _add_diameter_lines(ax3, centroid, diameter_AP, diameter_RL, orientation_rad, dim)
    # _add_ellipse(ax3, centroid, diameter_AP, diameter_RL, orientation_rad, dim, upscale)
    _setup_axis(ax3, 'Anterior-Posterior Symmetry')
    ax3.text(x0 - offset, y0, f"Posterior:\n{posterior_area:.2f} mm²", color='purple',
             fontsize=10, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    ax3.text(x0 + offset, y0, f"Anterior:\n{anterior_area:.2f} mm²", color='green',
             fontsize=10, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # Save figure
    os.makedirs('debug_figures_area_quadrants', exist_ok=True)
    fig.tight_layout()
    fig.savefig(f'debug_figures_area_quadrants/cord_quadrant_tmp_fig_slice_{iz:03d}.png', dpi=150)


def _add_diameter_lines(ax, centroid, diameter_AP, diameter_RL, orientation_rad, dim):
    """
    Helper function to add diameter lines to a matplotlib axis.
    """
    y0, x0 = centroid

    radius_ap = (diameter_AP / dim[0]) * 0.5
    radius_rl = (diameter_RL / dim[1]) * 0.5

    dx_ap = radius_ap * np.cos(orientation_rad)
    dy_ap = radius_ap * np.sin(orientation_rad)
    dx_rl = radius_rl * -np.sin(orientation_rad)
    dy_rl = radius_rl * np.cos(orientation_rad)

    ax.plot([x0 - dx_ap, x0 + dx_ap], [y0 - dy_ap, y0 + dy_ap], 'r--', linewidth=2, label='AP diameter')
    ax.plot([x0 - dx_rl, x0 + dx_rl], [y0 - dy_rl, y0 + dy_rl], 'b--', linewidth=2, label='RL diameter')

    # Add centroid
    ax.plot(x0, y0, '.g', markersize=15)


def _setup_axis(ax, title, xlabel='y\nPosterior-Anterior (PA)', ylabel='x\nLeft-Right (LR)'):
    """
    Helper function to set up common axis properties.
    """
    ax.grid()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def create_ap_diameter_plots(angle_hog, ap0_r, ap_diameter, dim, iz, properties, rl0_r, rl_diameter,
                             rotated_bin, seg_crop_r):
    """
    """
    def _add_labels(ax):
        """Add A, P, R, L labels"""
        bbox_params = dict(facecolor='black', alpha=1)
        ax.text(ap0_r, seg_crop_r.shape[0] * 0.95, 'L', color='white', fontsize=12, ha='center', va='center',
                bbox=bbox_params, alpha=0.8)
        ax.text(seg_crop_r.shape[1] * 0.95, rl0_r, 'A', color='white', fontsize=12, ha='center', va='center',
                bbox=bbox_params, alpha=0.8)
        ax.text(ap0_r, seg_crop_r.shape[0] * 0.05, 'R', color='white', fontsize=12, ha='center', va='center',
                bbox=bbox_params, alpha=0.8)
        ax.text(seg_crop_r.shape[1] * 0.05, rl0_r, 'P', color='white', fontsize=12, ha='center', va='center',
                bbox=bbox_params, alpha=0.8)

    def _add_ellipse(ax, x0, y0):
        """Add an ellipse to the plot."""
        ellipse = mpl_patches.Ellipse(
            (x0, y0),
            width=properties['diameter_AP_ellipse'] / dim[0],
            height=properties['diameter_RL'] / dim[1],
            angle=properties['orientation']*180.0/math.pi,
            edgecolor='orange',
            facecolor='none',
            linewidth=2,
            label=f"Ellipse fitted using skimage.regionprops, angle: {-properties['orientation']*180.0/math.pi:.2f}"
        )
        ax.add_patch(ellipse)

    # Plot the original and rotated segmentation
    fig = plt.figure(figsize=(6, 8))
    ax1 = fig.add_subplot(111)
    # 1. Original segmentation
    seg_crop_r_bin = np.array(seg_crop_r > 0.5, dtype='uint8')  # binarize the original segmentation
    ax1.imshow(seg_crop_r_bin, cmap='gray', alpha=0.6, label='Original Segmentation')
    # ax1.imshow(seg_crop_r_bin, cmap='Reds', alpha=1, label='Original Segmentation', vmin=0, vmax=1.3)
    # Add ellipse fitted using skimage.regionprops
    _, _, [y0, x0] = compute_pca(seg_crop_r)
    # Center of mass in the original segmentation
    ax1.plot(x0, y0, 'ko', markersize=10, label='Original Segmentation Center of Mass')
    _add_ellipse(ax1, x0, y0)
    # Draw AP and RL axes through the center of mass of the original segmentation
    ax1.arrow(ap0_r, rl0_r, np.sin(angle_hog + (90 * math.pi / 180)) * 25,
              np.cos(angle_hog + (90 * math.pi / 180)) * 25, color='black', width=0.1,
              head_width=1, label=f'HOG angle = {angle_hog * 180 / math.pi:.1f}°')  # convert to degrees
    # Add AP and RL diameters from the original segmentation obtained using skimage.regionprops
    radius_ap = (properties['diameter_AP_ellipse'] / dim[0]) * 0.5
    radius_rl = (properties['diameter_RL'] / dim[1]) * 0.5
    dx_ap = radius_ap * np.cos(properties['orientation'])
    dy_ap = radius_ap * np.sin(properties['orientation'])
    dx_rl = radius_rl * -np.sin(properties['orientation'])
    dy_rl = radius_rl * np.cos(properties['orientation'])
    ax1.plot([x0 - dx_ap, x0 + dx_ap], [y0 - dy_ap, y0 + dy_ap], color='blue', linestyle='--', linewidth=2,
             label=f'AP diameter (skimage.regionprops) = {properties["diameter_AP_ellipse"]:.2f} mm')
    ax1.plot([x0 - dx_rl, x0 + dx_rl], [y0 - dy_rl, y0 + dy_rl], color='blue', linestyle='solid', linewidth=2,
             label=f'RL diameter (skimage.regionprops) = {properties["diameter_RL"]:.2f} mm')
    # Add A, P, R, L labels
    _add_labels(ax1)

    # 2. Rotated segmentation by angle_hog
    ax1.imshow(rotated_bin, cmap='Reds', alpha=0.8, label='Rotated Segmentation')
    # Center of mass
    ax1.plot(ap0_r, rl0_r, 'bo', markersize=10, label='Rotated Segmentation Center of Mass')
    rotated_bin_bin = np.array(rotated_bin > 0.5, dtype='uint8')  # binarize the rotated segmentation
    right = np.nonzero(rotated_bin_bin[:, ap0_r])[0][0]
    left = np.nonzero(rotated_bin_bin[:, ap0_r])[0][-1]
    if rotated_bin_bin[rl0_r, :].size > 0 and np.any(rotated_bin_bin[rl0_r, :]):
        anterior = np.nonzero(rotated_bin_bin[rl0_r, :])[0][0]
        posterior = np.nonzero(rotated_bin_bin[rl0_r, :])[0][-1]
    else:
        anterior = posterior = np.nan
    ax1.plot([anterior, posterior], [rl0_r, rl0_r], color='red', linestyle='--', linewidth=2,
             label=f'AP Diameter (rotated segmentation) = {ap_diameter:.2f} mm, rl0_r={rl0_r}')
    ax1.plot([ap0_r, ap0_r], [left, right], color='red', linestyle='solid', linewidth=2,
             label=f'RL Diameter (rotated segmentation) = {rl_diameter:.2f} mm')

    # Plot horizontal and vertical grid lines
    ax1.grid(which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=1, framealpha=1.0, fontsize=8)
    ax1.set_title(f'Slice {iz}\nOriginal segmentation and Segmentation rotated by HOG angle')

    plt.tight_layout()
    # plt.show()
    # Save the figure
    if not os.path.exists('debug_figures_diameters'):
        os.makedirs('debug_figures_diameters')
    fname_out = os.path.join('debug_figures_diameters', f'slice_{iz}.png')
    fig.savefig(fname_out, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free up memory
    print(f'Saved debug figure for slice {iz} with segmentation properties to {fname_out}')


def create_symmetry_plots(
    # Core inputs (provided directly to `_calculate_symmetry`)
    seg_crop_r_rotated, centroid,
    # Derived inputs (calculated within `_calculate_symmetry`)
    # Technically these are redundant since they can be derived from the core inputs,
    # but passing them in directly avoids duplicate calculations.
    seg_crop_r_rotated_cut, seg_crop_r_rotated_cut_RL,
    seg_crop_r_flipped_RL, seg_crop_r_flipped_AP,
    symmetry_metrics,
    # Debug inputs
    iz
):
    # Simple transformations of the input
    y0, x0 = centroid
    y0 = int(round(y0))
    x0 = int(round(x0))

    # Get the AP and RL coordinates
    coords_AP = skimage.metrics.hausdorff_pair(seg_crop_r_rotated_cut > 0.5, seg_crop_r_flipped_AP > 0.5)
    coords_RL = skimage.metrics.hausdorff_pair(seg_crop_r_rotated_cut_RL > 0.5, seg_crop_r_flipped_RL > 0.5)

    plt.figure(figsize=(8, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(seg_crop_r_rotated > 0.5, cmap='gray', vmin=0, vmax=0.1, alpha=1)
    plt.imshow(seg_crop_r_rotated_cut_RL > 0.5, cmap='Reds', vmin=0, vmax=0.1, alpha=0.7)
    plt.imshow(seg_crop_r_flipped_RL > 0.5, cmap='Blues', vmin=0, vmax=0.1, alpha=0.6)
    plt.plot(x0, y0, 'go', markersize=5, label='Centroid')
    if coords_RL is not None and len(coords_RL) == 2:
        (y1, x1), (y2, x2) = coords_RL
        ax2 = plt.gca()
        ax2.plot([x1, x2], [y1, y2], 'y-', linewidth=2, label='Hausdorff distance')
        ax2.plot([x1, x2], [y1, y2], 'yo', markersize=5)
    # plt.title('RL dice')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(seg_crop_r_rotated > 0.5, cmap='gray', vmin=0, vmax=0.1, alpha=1)
    plt.imshow(seg_crop_r_rotated_cut > 0.5, cmap='Reds', vmin=0, vmax=0.1, alpha=0.7)
    plt.imshow(seg_crop_r_flipped_AP > 0.5, cmap='Blues', vmin=0, vmax=0.1, alpha=0.4)
    plt.plot(x0, y0, 'go', markersize=5, label='Centroid')

    # Plot Hausdorff pair points and line for AP dice
    if coords_AP is not None and len(coords_AP) == 2:
        (y1, x1), (y2, x2) = coords_AP
        ax2 = plt.gca()
        ax2.plot([x1, x2], [y1, y2], 'y-', linewidth=2, label='Hausdorff distance')
        ax2.plot([x1, x2], [y1, y2], 'yo', markersize=5)
    # plt.title('AP dice')
    plt.axis('off')
    # Move the legend outside of the subplots
    plt.legend(loc='lower center', bbox_to_anchor=(-0.1, -0.1), ncol=2)
    plt.suptitle(
        f'Symmetry Dice RL: {symmetry_metrics["symmetry_dice_RL"]:.3f}, AP: {symmetry_metrics["symmetry_dice_AP"]:.3f}\n'
        f'Hausdorff RL (mm): {symmetry_metrics["symmetry_hausdorff_RL"]:.3f}, AP: {symmetry_metrics["symmetry_hausdorff_AP"]:.3f}\n'
        f'Symmetric diff RL (mm²): {symmetry_metrics["symmetry_difference_RL"]:.3f}, AP: {symmetry_metrics["symmetry_difference_AP"]:.3f}'
    )
    if not os.path.exists('debug_figures_symmetry'):
        os.makedirs('debug_figures_symmetry')
    fname_out = os.path.join('debug_figures_symmetry', f'process_seg_symmetry_dice_z{iz:03d}.png')
    plt.savefig(fname_out, dpi=300)
    plt.close()
    logging.info(f"Saved symmetry Dice visualization to: {fname_out}")
