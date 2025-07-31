"""
src.post_process.py
Post Processing and Data Output Functions
BoMeyering 2025
"""

import json
import cv2
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
import numpy as np
from typing import Tuple, Union
from ensemble_boxes import weighted_boxes_fusion

# Read in class maps
# Marker Classes
with open('.metadata/obj_det_class_map.json', 'r') as f:
    marker_classes = json.load(f)

# Segmentation Classes
with open('.metadata/seg_class_map.json', 'r') as f:
    seg_classes = json.load(f)


def bbox_fusion(bboxes):
    """ Perform Weighted Box Fusion on the bbox predictions """
    
    bboxes = bboxes.copy()
    # clip bboxes outside of allowed range
    bboxes[:, :4] = np.clip(bboxes[:, :4], a_min=0, a_max=1024)

    # Normalize and convert to lists
    bbox_list = [(bboxes[:, :4]/1024).tolist()]
    conf_list = [bboxes[:, 4].tolist()]
    label_list = [bboxes[:, 5].tolist()]

    # Calculate the weighted fusions
    bbox_list, conf_list, label_list = weighted_boxes_fusion(boxes_list=bbox_list, scores_list=conf_list, labels_list=label_list)

    # Rearrange as numpy array on original scale
    fused_bboxes = np.hstack((np.array(bbox_list)*1024, np.array(conf_list).reshape(-1, 1), np.array(label_list).reshape(-1, 1)))

    return fused_bboxes


def map_preds(preds):
    """preds = integer mask of shape (H, W)"""

    h, w = preds.shape
    color_mask = np.zeros(shape=(h, w, 3)).astype(np.uint8)
    for i in np.unique(preds):
        idx = np.where(preds==i)
        rgb = seg_classes.get(str(i))[0]
        color_mask[idx] = np.array(rgb)
    
    return color_mask

def resize_predictions(output_map: np.ndarray, bboxes: np.ndarray, img: np.ndarray):
    """ Resize all predictions to original image size """

    # Grab the img H and W
    h, w = img.shape[:2]
    shape_vec = np.array([w, h]*2)

    # Resize output map
    output_map = cv2.resize(output_map, img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

    # Resize bounding boxes
    bboxes[:, :4] = bboxes[:, :4] / 1024 * shape_vec

    return output_map, bboxes


def subset_markers(bboxes: np.ndarray, type: str='Both', threshold: Union[float, None]=0.85): 
    """ Subset the detection based on threshold and marker type """
    # Threshold
    if threshold:
        # Get indices for confidence values above threshold
        idx = np.where(bboxes[:, 4] >= threshold)
        bboxes = bboxes[idx]

    if type != "Both":
        class_idx = marker_classes.get(type)
    else:
        # Find top predictions and get the majority class
        # Breaks ties in case of erroneous predictions
        top_four = bboxes[:4, 5].astype(int)
        elements, counts = np.unique(top_four, return_counts=True)
        argmax_idx = np.argmax(counts)
        class_idx = elements[argmax_idx]

    # Subset the top four predictions of class_idx
    idx = np.where(bboxes[:, 5].astype(int) == class_idx)
    bboxes = bboxes[idx]

    # Return top 4 predictions
    return bboxes[:4,:].copy()

def class_proportions(preds: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Calculate the proportions of each PGC class in the predictions

    Args:
        preds (np.ndarray): The integer predictions as a numpy array of shape (H, W)

    Returns:
        Tuple[np.ndarray, dict]: a 1-dimensional numpy array of the proportions of each class, along with a dictionary with keys corresponding to the class name.
    """
    counts = np.bincount(preds.flatten(), minlength=9)
    total_elements = preds.size
    props = counts / total_elements
    
    keys = ['background', 'quadrat', 'pgc_grass', 'pgc_clover', 'broadleaf_weed', 'maize', 'soybean', 'other_vegetation']
    prop_dict = {k: v for k, v in zip(keys, props)}
    
    return props, prop_dict

def match_corner_points(pts: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
    """
    Match a set of 4 midpoints from the ROI to the corners of the image.
    Return them ordered in clockwise fashion starting at top left corner.

    Args:
        pts (np.ndarray): point array with x, y coordinates of shape (3, 2) or (4, 2)
        img_shape (Tuple[int, int]): _description_

    Returns:
        np.ndarray: _description_
    """

    # Unpack the image shape
    height, width = img_shape
    if len(pts) <= 2:
        raise ValueError("Fewer than three points were passed to 'pts'.")
    elif np.any(pts > np.max(img_shape)):
        raise ValueError('One or more of the coordinates in pts is outside the bounds of the image.')
    # elif np.any(np.max(pts, axis=0) > img_shape[::-1]):
    #     raise ValueError('One or more of the coordinates in pts is outside the bounds of the image.')
    elif np.any(pts < 0):
        raise ValueError('One or more of the coordinates in pts is negative.')        

    # Set up the iamge corner array and compute distance matrix. Start at top left origin and work clockwise
    img_corners = np.array([
        [0, 0],
        [width, 0], 
        [width, height],
        [0, height]
    ])

    # Calculate the distance between all of the points
    d_mat = distance_matrix(img_corners, pts)

    # Optimize the linear sums to find the best matching corner using distance a cost
    row_ind, col_ind = linear_sum_assignment(d_mat)

    if len(row_ind) < 4:
        print("interpolating 4th point")
        rect = pts[col_ind].astype('float32')
        temp_rect = np.zeros((4, 2)).astype('float32')
        temp_rect[row_ind] = rect
        if 0 not in row_ind:
            p_vec = temp_rect[2] - temp_rect[1]
            point = temp_rect[3] - p_vec
            print(point)
            temp_rect[0] = point
            return temp_rect
        elif 1 not in row_ind:
            p_vec = temp_rect[3] - temp_rect[0]
            point = temp_rect[2] - p_vec
            print(point)
            temp_rect[1] = point
            return temp_rect
        elif 2 not in row_ind:
            p_vec = temp_rect[0] - temp_rect[3]
            point = temp_rect[1] - p_vec
            print(point)
            temp_rect[2] = point
            return temp_rect
        elif 3 not in row_ind:
            p_vec = temp_rect[1] - temp_rect[2]
            point = temp_rect[0] - p_vec
            print(point)
            temp_rect[3] = point
            return temp_rect
    else:
        # Reorder the points
        rect = pts[col_ind].astype('float32')

        return rect
    

def get_marker_midpoints(bboxes: np.ndarray, clip: bool=False) -> np.ndarray:
    """
    Take a set of marker bounding box coordinates and return the midpoints of each one.

    Args:
        pts (np.ndarray): A numpy array of shape (N, 4), in the format (x1, y1, x2, y2).
        clip (bool, optional): Clip the midpoints between 0 and 1024. Defaults to False.

    Returns:
        np.ndarray: A numpy array of shape (N, 2) with coordinates in the format of (cx, cy)
    """
    if clip:
        bboxes[:, :4] = bboxes[:, :4].clip(min=0, max = 1024)
    
    # Calculate the mid x and y coordinates
    cx = ((bboxes[:, 2] + bboxes[:, 0]) / 2).astype(np.int32)
    cy = ((bboxes[:, 3] + bboxes[:, 1]) / 2).astype(np.int32)
    
    mdpts = np.array([[x, y] for x, y in zip(cx, cy)])
    
    return mdpts

def transform_roi(img: np.ndarray, roi_mdpts: np.ndarray, output_shape = (1024, 1024)):

    h, w = output_shape
    dst = np.array(
        [
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ],
        dtype=np.float32
    )
    M = cv2.getPerspectiveTransform(roi_mdpts, dst)

    # Transform the ROI using nearest interpolation
    transformed = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_NEAREST)

    return transformed