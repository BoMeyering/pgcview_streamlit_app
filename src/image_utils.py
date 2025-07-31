"""
src/image_utils.py
Image Utility Functions
BoMeyering 2025
"""

import cv2
import numpy as np

def read_file_to_numpy(file):
    """ Read a file in and output numpy array in OpenCV format """
    file.seek(0)
    try:
          
        buffer = file.read()                            # Read buffer
        image = np.frombuffer(buffer, dtype=np.uint8)   # Create array
        image = cv2.imdecode(image, cv2.IMREAD_COLOR) 

        return image
    
    except Exception as e:
        print(f"Error encoding image {file}: {str(e)}")
        return None

def draw_bounding_boxes(img, pts, pts_range=1024):
    """
    Draw bounding boxes on an image
    """
    if img.shape[:2] != (1024, 1024):
        thickness = 10
    else:
        thickness = 3
    
    img_copy = img.copy()
    for bbox in pts:
        pt_list = bbox[0:4].tolist()
        x1, y1, x2, y2 = (int(i) for i in pt_list) 
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 255), thickness)
        
    return img_copy

def map_preds(preds, mapping):
    """preds = integer mask of shape (H, W)"""

    h, w = preds.shape
    color_mask = np.zeros(shape=(h, w, 3)).astype(np.uint8)
    for i in np.unique(preds):
        idx = np.where(preds==i)
        rgb = mapping.get(i)
        color_mask[idx] = np.array(rgb)
    
    return color_mask

def overlay_preds(img, color_mask, alpha, gamma=0.0):
    beta = 1-alpha
    overlay = cv2.addWeighted(img, alpha, color_mask, beta, gamma)
    return overlay


def clahe_channel(img: np.ndarray, clip_limit=100) -> np.ndarray:
    """
    Performs CLAHE on the l* channel of an image

    Args:
        img (np.ndarray): a three channel image in opencv format (BGR order)

    Returns:
        np.ndarray: The adjusted image
    """
    # Convert to lab
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    
    # Apply CLAHE
    l_channel = lab_img[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=clip_limit)
    new_l = clahe.apply(l_channel)
    lab_img[:, :, 0] = new_l
    
    # Convert back to BGR
    bgr_img = cv2.cvtColor(lab_img, cv2.COLOR_Lab2BGR)

    return bgr_img