"""
src.img_utils.py
BoMeyering 2025
"""

import base64
from PIL import Image
import numpy as np

def get_shape_and_dtype(file):

    image = Image.open(file)
    image = np.array(image)
    img_shape = list(image.shape)
    dtype = str(image.dtype)

    return img_shape, dtype

def encode_image(file):
    """ Encode an uploaded image in Base64 """
    img_shape, dtype = get_shape_and_dtype(file)
    buffer = file.read()
    encoded = base64.b64encode(buffer).decode()

    return encoded, img_shape, dtype

def encode_upload_list(file_uploads):
    """ Encode a whole list of image uploads """
    encoded_strings = []
    file_names = []
    metadata = []

    for file in file_uploads:
        encoded, img_shape, dtype = encode_image(file)
        encoded_strings.append(encoded)
        file_names.append(file.name)
        metadata.append((img_shape, dtype))

    return encoded_strings, file_names, metadata