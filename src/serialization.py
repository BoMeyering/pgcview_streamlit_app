"""
src.img_utils.py
BoMeyering 2025
"""

import base64
from PIL import Image
import numpy as np
import cv2


def encode_image(file):
    """ Encode an uploaded image in Base64 """
    file.seek(0)
    try:
          
        buffer = file.read()                            # Read buffer
        image = np.frombuffer(buffer, dtype=np.uint8)   # Create array
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)   # Decode to OpenCV

        if image is None:
            raise ValueError("Could not decode image")
        
        # Step 3: Resize
        resized = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_AREA) # Resize the image to 1024x1024

        _, png_encoded = cv2.imencode('.png', resized)
        encoded = base64.b64encode(png_encoded).decode('utf-8')

    except Exception as e:
        print(f"Error encoding image {file}: {str(e)}")
        return None

    return encoded

def encode_upload_list(file_uploads):
    """ Encode a whole list of image uploads """
    encoded_strings = []
    file_names = []

    for file in file_uploads:
        encoded = encode_image(file)
        encoded_strings.append(encoded)
        file_names.append(file.name)

    return encoded_strings, file_names

def decode_output_map(b64_map: str) -> np.ndarray:
    """ Decode the output PNG map from the response object """

    arr_bytes = base64.b64decode(b64_map)
    png_encoded = np.frombuffer(arr_bytes, dtype=np.uint8)  # Load bytes into PNG array
    output_map = cv2.imdecode(png_encoded, cv2.IMREAD_UNCHANGED)

    return output_map

def decode_bbox_array(bbox_output: dict) -> np.ndarray:
    """ Decode the Base64 encoded bbox array """
    
    dtype = bbox_output.get('dtype')
    shape = bbox_output.get('shape')
    b64_arr = bbox_output.get('bboxes')

    arr_bytes = base64.b64decode(b64_arr)
    bbox_array = np.frombuffer(arr_bytes, dtype=dtype).reshape(shape)

    return bbox_array