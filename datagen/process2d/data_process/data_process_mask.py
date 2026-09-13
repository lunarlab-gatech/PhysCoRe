"""
Clean the segmentation masks and save the processed per-frame masks.
"""

import numpy as np
import open3d as o3d
import json
from tqdm import tqdm
import glob
import cv2
import pickle
from argparse import ArgumentParser

from utils.o3d_window import create_visualizer

parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    required=True,
)
parser.add_argument("--case_name", type=str, required=True)
parser.add_argument("--controller_name", type=str, required=True)
parser.add_argument("--headless", action="store_true",
                    help="Render Open3D offscreen (no display, e.g. ssh/GPU node).")
args = parser.parse_args()

base_path = args.base_path
case_name = args.case_name
CONTROLLER_NAME = args.controller_name
HEADLESS = args.headless

processed_masks = {}


def read_mask(mask_path):
    # Convert the white mask into binary mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = mask > 0
    return mask


def process_pcd_mask(frame_idx, pcd_path, mask_path, mask_info, num_cam):
    global processed_masks
    processed_masks[frame_idx] = {}

    # Load the pcd data
    data = np.load(f"{pcd_path}/{frame_idx}.npz")
    points = data["points"]
    colors = data["colors"]
    masks = data["masks"]

    object_pcd = o3d.geometry.PointCloud()
    controller_pcd = o3d.geometry.PointCloud()

    for i in range(num_cam):
        # Load the object mask (union of all object detections)
        object_mask_raw = np.zeros_like(masks[i])
        for object_idx in mask_info[i]["object"]:
            mask = read_mask(f"{mask_path}/{i}/{object_idx}/{frame_idx}.png")
            object_mask_raw = np.logical_or(object_mask_raw, mask)
        object_mask = np.logical_and(masks[i], object_mask_raw)
        object_points = points[i][object_mask]
        object_colors = colors[i][object_mask]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(object_points)
        pcd.colors = o3d.utility.Vector3dVector(object_colors)
        object_pcd += pcd

        # Load the controller mask
        controller_mask = np.zeros_like(masks[i])
        for controller_idx in mask_info[i]["controller"]:
            mask = read_mask(f"{mask_path}/{i}/{controller_idx}/{frame_idx}.png")
            controller_mask = np.logical_or(controller_mask, mask)
        controller_mask = np.logical_and(masks[i], controller_mask)
        controller_points = points[i][controller_mask]
        controller_colors = colors[i][controller_mask]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(controller_points)
        pcd.colors = o3d.utility.Vector3dVector(controller_colors)
        controller_pcd += pcd

    # Apply the outlier removal
    cl, ind = object_pcd.remove_radius_outlier(nb_points=40, radius=0.01)
    filtered_object_points = np.asarray(
        object_pcd.select_by_index(ind, invert=True).points
    )
    object_pcd = object_pcd.select_by_index(ind)

    cl, ind = controller_pcd.remove_radius_outlier(nb_points=40, radius=0.01)
    filtered_controller_points = np.asarray(
        controller_pcd.select_by_index(ind, invert=True).points
    )
    controller_pcd = controller_pcd.select_by_index(ind)

    object_pcd = o3d.geometry.PointCloud()
    controller_pcd = o3d.geometry.PointCloud()
    for i in range(num_cam):
        processed_masks[frame_idx][i] = {}
        # Load the object mask (union of all object detections)
        object_mask_raw = np.zeros_like(masks[i])
        for object_idx in mask_info[i]["object"]:
            mask = read_mask(f"{mask_path}/{i}/{object_idx}/{frame_idx}.png")
            object_mask_raw = np.logical_or(object_mask_raw, mask)
        object_mask = np.logical_and(masks[i], object_mask_raw)
        object_points = points[i][object_mask]
        indices = np.nonzero(object_mask)
        indices_list = list(zip(indices[0], indices[1]))
        # Locate all the object_points in the filtered points
        object_indices = []
        for j, point in enumerate(object_points):
            if tuple(point) in filtered_object_points:
                object_indices.append(j)
        original_indices = [indices_list[j] for j in object_indices]
        # Update the object mask
        for idx in original_indices:
            object_mask[idx[0], idx[1]] = 0
        processed_masks[frame_idx][i]["object"] = object_mask

        # Load the controller mask
        controller_mask = np.zeros_like(masks[i])
        for controller_idx in mask_info[i]["controller"]:
            mask = read_mask(f"{mask_path}/{i}/{controller_idx}/{frame_idx}.png")
            controller_mask = np.logical_or(controller_mask, mask)
        controller_mask = np.logical_and(masks[i], controller_mask)
        controller_points = points[i][controller_mask]
        indices = np.nonzero(controller_mask)
        indices_list = list(zip(indices[0], indices[1]))
        # Locate all the controller_points in the filtered points
        controller_indices = []
        for j, point in enumerate(controller_points):
            if tuple(point) in filtered_controller_points:
                controller_indices.append(j)
        original_indices = [indices_list[j] for j in controller_indices]
        # Update the controller mask
        for idx in original_indices:
            controller_mask[idx[0], idx[1]] = 0
        processed_masks[frame_idx][i]["controller"] = controller_mask

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[i][object_mask])
        pcd.colors = o3d.utility.Vector3dVector(colors[i][object_mask])

        object_pcd += pcd

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[i][controller_mask])
        pcd.colors = o3d.utility.Vector3dVector(colors[i][controller_mask])

        controller_pcd += pcd

    return object_pcd, controller_pcd


if __name__ == "__main__":
    pcd_path = f"{base_path}/{case_name}/pcd"
    mask_path = f"{base_path}/{case_name}/mask"

    num_cam = len(glob.glob(f"{mask_path}/mask_info_*.json"))
    frame_num = len(glob.glob(f"{pcd_path}/*.npz"))
    # Load the mask metadata
    mask_info = {}
    for i in range(num_cam):
        with open(f"{base_path}/{case_name}/mask/mask_info_{i}.json", "r") as f:
            data = json.load(f)
        mask_info[i] = {"object": [], "controller": []}
        for key, value in data.items():
            if value != CONTROLLER_NAME:
                mask_info[i]["object"].append(int(key))
            else:
                mask_info[i]["controller"].append(int(key))

    vis = create_visualizer(HEADLESS)

    object_pcd = None
    controller_pcd = None
    for i in tqdm(range(frame_num)):
        temp_object_pcd, temp_controller_pcd = process_pcd_mask(
            i, pcd_path, mask_path, mask_info, num_cam
        )
        if i == 0:
            object_pcd = temp_object_pcd
            controller_pcd = temp_controller_pcd
            vis.add_geometry(object_pcd)
            vis.add_geometry(controller_pcd)
            # Adjust the viewpoint
            view_control = vis.get_view_control()
            view_control.set_front([1, 0, -2])
            view_control.set_up([0, 0, -1])
            view_control.set_zoom(1)
        else:
            object_pcd.points = o3d.utility.Vector3dVector(temp_object_pcd.points)
            object_pcd.colors = o3d.utility.Vector3dVector(temp_object_pcd.colors)
            controller_pcd.points = o3d.utility.Vector3dVector(
                temp_controller_pcd.points
            )
            controller_pcd.colors = o3d.utility.Vector3dVector(
                temp_controller_pcd.colors
            )
            vis.update_geometry(object_pcd)
            vis.update_geometry(controller_pcd)
            vis.poll_events()
            vis.update_renderer()

    # Save the processed masks considering both depth filter, semantic filter and outlier filter
    with open(f"{base_path}/{case_name}/mask/processed_masks.pkl", "wb") as f:
        pickle.dump(processed_masks, f)
