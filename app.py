"""
Main Streamlit App
BoMeyering 2025
"""

import streamlit as st
import time
import base64
import io
import cv2
import pandas as pd
import numpy as np
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build


from src.serialization import encode_upload_list, decode_bbox_array, decode_output_map
from src.api import send_request_sync
from src.image_utils import read_file_to_numpy, draw_bounding_boxes, clahe_channel
from src.post_process import subset_markers, class_proportions, resize_predictions, map_preds, match_corner_points, get_marker_midpoints, transform_roi, bbox_fusion

st.set_page_config(layout="wide")

# Google Drive Session Creds
if 'credentials' not in st.session_state:
    st.session_state.credentials = None

st.title("PGCView Image Analysis Pipeline")
st.text("Please select the pipeline version to use")

with st.sidebar:
    st.button('press me')

pipeline_version = st.selectbox("Pipeline Version", ["v1", "v2"])

if pipeline_version == 'v2':
    render_app = False
    st.subheader(f"You selected {pipeline_version}.")
    st.text("Unfortunately version 2 of the PGC View image analysis pipeline is not ready just yet. ")
    st.stop()

st.subheader(f"You have selected {pipeline_version}.")
col1, col2 = st.columns(2)

with col1:
    st.text('Select how you want to upload files')
    upload_type = st.selectbox('File Picker', options=['Local upload', 'Google Drive'])
    if upload_type == 'Local upload':
        files = st.file_uploader(label="Browse", accept_multiple_files=True, type=['jpeg', 'jpg', 'png'], )
        if not files:
            st.stop()

if len(files) > 0:
    st.text(f"You uploaded a total of {len(files)} images")
    with col2:
        return_type = st.selectbox(label='Select the data you want to return', options=['Tabular results', 'Overlay Images', 'Both'])
        marker_type = st.selectbox(label='Do your images contain markers, PVC quadrats or both?', options=['Both', 'Marker', 'Quadrat'])
        conf_restriction = st.checkbox('Do you want to restrict the detections based on the detection confidence?')
        if conf_restriction:
            threshold = st.slider(label='Detection confidence threshold', min_value=0.1, max_value=1.00, value = 0.8)
        else:
            threshold = None
        roi_restriction = st.checkbox('Restrict predictions to ROI if possible?')

st.divider()


st.text("Everything looks good! Ready to run the pipeline?")
run_button = st.button(label='Submit Inference Job', on_click=None)


# Image pipeline event
if run_button:
    with st.status("Preprocessing images and metadata...", expanded=True) as status:
        encoded_images, file_names = encode_upload_list(files)
        status.update(label="Preprocessing Complete", state="complete", expanded=False)
    
    cold_start_placeholder = st.empty()
    
    prog_col1, prog_col2 = st.columns([0.15, 0.85])
    with prog_col1:
        # Placeholder containers
        img_placeholder = st.empty()

    with prog_col2:
        progress_text = ""
        my_bar = st.progress(0, text=progress_text)

    # Send each image to the inference endpoint
    # initialize progress and counters
    i = 0
    progress = i+1 / len(files)

    results = pd.DataFrame(columns=['file_name', 'background', 'quadrat', 'pgc_grass', 'pgc_clover', 'broadleaf_weed', 'maize', 'soybean', 'other_vegetation'])


    # HSV threshold settings
    brown_lo = (0, 0, 45)
    brown_hi = (30, 110, 255)
    green_lo = (32, 45, 45)
    green_hi = (115, 255, 255)



    # Main processing loop
    for b64_img, file_name in zip(encoded_images, file_names):
        print(f"Processing file {file_name}")
        if i==0:
            cold_start_placeholder.warning("The model pipeline is waking up. Inference may be slow on the first image")
        else:
            cold_start_placeholder.empty()
        
        # Update progress bar text
        my_bar.progress(progress, text=f"Processing {file_name} ({i+1}/{len(files)})")

        # Create API payload
        payload = {
            'input': {
				"image": b64_img,
                "img_name": file_name
			}
        }
        
        # Update img_placeholder
        img_placeholder.image(files[i])

        # Send request and get output object
        response = send_request_sync(payload)
        output = response['output']

        # Get each models' output
        seg_data = output['dlv3p']
        det_data = output['effdet']

        # Get and decode all raw model predictions
        output_map = decode_output_map(seg_data['out_map'])
        bboxes = decode_bbox_array(det_data)

        # Perform bbox fusion to get rid of overlapping predictions
        bboxes = bbox_fusion(bboxes)

        # Read in original image
        img = read_file_to_numpy(files[i])

        # Subset marker predictions
        bboxes = subset_markers(bboxes, type=marker_type, threshold=threshold)

        # Resize all predictions to original image size
        output_map, bboxes = resize_predictions(output_map, bboxes, img)

        # Create a color mask for the overlay
        color_mask = map_preds(output_map)


        out_image = draw_bounding_boxes(color_mask, bboxes)
        cv2.imwrite(f"output_images/{file_name.split()[0]}_bboxes.png", out_image)

        # Find ROI Corners and match
        roi_corner_pts = get_marker_midpoints(bboxes)
        roi_corner_pts = match_corner_points(roi_corner_pts, img.shape[:2])

        # Transform images and masks to ROI
        transformed_color_image = transform_roi(img=img, roi_mdpts=roi_corner_pts)
        transformed_color_mask = transform_roi(img=color_mask, roi_mdpts=roi_corner_pts)
        transformed_binary_mask = transform_roi(img=output_map, roi_mdpts=roi_corner_pts)

        # Grab the class proportions from the mask
        proportions, prop_dict = class_proportions(transformed_binary_mask)
        cv2.imwrite(f"output_images/{file_name.split()[0]}_transformed.png", transformed_color_mask)

        # Run CLAHE on the color ROI
        clahe_img = clahe_channel(transformed_color_image, 5)

        # Grab proportions from CLAHE and normal image
        clahe_hsv = cv2.cvtColor(clahe_img, cv2.COLOR_BGR2HSV)
        color_hsv = cv2.cvtColor(transformed_color_image, cv2.COLOR_BGR2HSV)

        # PGC Grass Mask
        grass_mask = np.zeros((transformed_color_image.shape[:2]))
        idx = np.where(transformed_binary_mask==2)
        grass_mask[idx] = 255
        grass_mask = grass_mask.astype(np.uint8)

        # Active/Dormant props for CLAHE
        hsv_gr_mask = cv2.inRange(src=clahe_hsv, lowerb=green_lo, upperb=green_hi) * grass_mask
        hsv_br_mask = cv2.inRange(src=clahe_hsv, lowerb=brown_lo, upperb=brown_hi) * grass_mask
        
        if prop_dict['pgc_grass'] > 0:
            green_fraction = (hsv_gr_mask > 0).sum()
            brown_fraction = (hsv_br_mask > 0).sum()
            total_grass_px = green_fraction + brown_fraction
            green_fraction /= total_grass_px
            brown_fraction /= total_grass_px
            prop_dict['CLAHE_active_grass'] = green_fraction
            prop_dict['CLAHE_dormant_grass'] = brown_fraction
        else:
            prop_dict['CLAHE_active_grass'] = 0
            prop_dict['CLAHE_dormant_grass'] = 0

        # Active/Dormant props for CLAHE
        hsv_gr_mask = cv2.inRange(src=color_hsv, lowerb=green_lo, upperb=green_hi) * grass_mask
        hsv_br_mask = cv2.inRange(src=color_hsv, lowerb=brown_lo, upperb=brown_hi) * grass_mask
        
        if prop_dict['pgc_grass'] > 0:
            green_fraction = (hsv_gr_mask > 0).sum()
            brown_fraction = (hsv_br_mask > 0).sum()
            total_grass_px = green_fraction + brown_fraction
            green_fraction /= total_grass_px
            brown_fraction /= total_grass_px
            prop_dict['COLOR_active_grass'] = green_fraction
            prop_dict['COLOR_dormant_grass'] = brown_fraction
        else:
            prop_dict['COLOR_active_grass'] = 0
            prop_dict['COLOR_dormant_grass'] = 0



        cv2.imwrite(f"output_images/{file_name.split()[0]}_color.png", transformed_color_image)
        cv2.imwrite(f"output_images/{file_name.split()[0]}_clahe.png", clahe_img)

        prop_dict['file_name'] = file_name
        prop_df = pd.DataFrame([prop_dict])
        results = pd.concat([results, prop_df], ignore_index=True)

        

        delay_time = response['delayTime']
        execution_time = response['executionTime']
        run_id = response['id']
        status = response['status']        

        worker_id = response['workerId']

        i+=1
        progress = (i+1) / len(files)

    # Empty the progress containers
    img_placeholder.empty()
    my_bar.empty()

    # Add success message
    st.success("Finished processing all images!")

    if return_type in ['Tabular results', 'Both']:

        dummy_return = pd.read_csv('dummy_data.csv')

        st.dataframe(results)
