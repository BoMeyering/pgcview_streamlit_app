"""
src.post_process.py
Post Processing and Data Output Functions
BoMeyering 2025
"""

import json
import numpy as np
from typing import Tuple

# Read in class maps
# Marker Classes
with open('.metadata/obj_det_class_map.json', 'r') as f:
    marker_classes = json.load(f)

# Segmentation Classes
with open('.metadata/seg_class_map.json', 'r') as f:
    seg_classes = json.load(f)


def subset_markers(bbox_preds: np.ndarray, type: str='Both', threshold: float=0.85): 

    # Threshold
    idx = np.where(bbox_preds[:, 4] >= threshold)
    bbox_preds = bbox_preds[idx]

    if type != "Both":
        class_idx = marker_classes.get(type)
        print(f'CLASS INDEX: {class_idx}')
        print(bbox_preds[:, 5].astype(int))
        idx = np.where(bbox_preds[:, 5].astype(int) == class_idx)
        bbox_preds = bbox_preds[idx]

    return bbox_preds

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

