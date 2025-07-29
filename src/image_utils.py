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
        print(True)
        h, w = img.shape[:2]
        shape_vec = np.array([w, h]*2)
        thickness = 10
        pts[:, :4] = pts[:, :4] / 1024 * shape_vec

    # if img.shape[:2] != (1024, 1024):
    #     h, w = img.shape[0:2]
    #     thickness = 10
    else:
        h, w = 1024, 1024
        thickness = 3
    
    img_copy = img.copy()
    for bbox in pts:
        print(bbox)
        pt_list = bbox[0:4].tolist()
        print(pt_list)
        x1, y1, x2, y2 = (int(i) for i in pt_list) 
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (255, 100, 100), thickness)
        
    return img_copy

def overlay_preds(img, color_mask, alpha, gamma=0.0):
    beta = 1-alpha
    overlay = cv2.addWeighted(img, alpha, color_mask, beta, gamma)
    return overlay