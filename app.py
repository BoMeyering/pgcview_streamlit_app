"""
Main Streamlit App
BoMeyering 2025
"""

import streamlit as st
import time
import base64
import io
import pandas as pd
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

from src.image_utils import encode_upload_list
from src.api import send_request_sync

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
    elif upload_type == "Google Drive":
        query_params = st.query_params
        if st.session_state.credentials is None and "code" not in query_params:
            # Not yet authorized
            flow = Flow.from_client_secrets_file(
                '.streamlit/client_secret.json',
                scopes=['https://www.googleapis.com/auth/drive.readonly'],
                redirect_uri='http://localhost:8501/'  # Replace when deployed
            )
            auth_url, _ = flow.authorization_url(prompt='consent')
            st.markdown(f"[🔐 Click here to connect to Google Drive]({auth_url})")
        elif "code" in query_params and st.session_state.credentials is None:
            # Handle redirect and get token
            code = query_params["code"]
            flow = Flow.from_client_secrets_file(
                '.streamlit/client_secret.json',
                scopes=['https://www.googleapis.com/auth/drive.readonly'],
                redirect_uri='http://localhost:8501/',
            )
            flow.fetch_token(code=code)
            creds = flow.credentials
            st.session_state.credentials = creds
            st.success("✅ Google Drive connected!")

        # Once authenticated, list image files
        if st.session_state.credentials:
            drive_service = build('drive', 'v3', credentials=st.session_state.credentials)
            results = drive_service.files().list(
                q="mimeType contains 'image/'",
                pageSize=10,
                fields="files(id, name)"
            ).execute()
            items = results.get("files", [])

            if not items:
                st.warning("No image files found in your Google Drive.")
                st.stop()

            selected_file = st.selectbox("Choose an image file", [f"{f['name']} ({f['id']})" for f in items])
            selected_id = selected_file.split('(')[-1].replace(')', '')

            # Download the file as binary
            request = drive_service.files().get_media(fileId=selected_id)
            fh = io.BytesIO()
            from googleapiclient.http import MediaIoBaseDownload
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                _, done = downloader.next_chunk()

            fh.seek(0)
            files = [fh]
            files[0].name = selected_file.split('(')[0].strip()




if len(files) > 0:
    st.text(f"You uploaded a total of {len(files)} images")
    with col2:
        return_type = st.selectbox(label='Select the data you want to return', options=['Tabular results', 'Overlay Images', 'Both'])
        marker_type = st.selectbox(label='Do your images contain markers, PVC quadrats or both?', options=['Markers', "Quadrat", 'Both'])
        confidence_value = st.slider(label='Detection confidence threshold', min_value=0.1, max_value=1.00, value = 0.8)

st.divider()

st.text("Everything looks good! Ready to run the pipeline?")
run_button = st.button(label='Submit Inference Job', on_click=None)

if run_button:
    with st.status("Preprocessing images and metadata...", expanded=True) as status:
        encoded_images, file_names, metadata = encode_upload_list(files)
        status.update(label="Download complete!", state="complete", expanded=False)

    progress_text = "Running images through pipeline. Please wait..."
    my_bar = st.progress(0, text=progress_text)

    # Send each image to the inference endpoint
    i = 0
    for b64_img, file_name, (img_shape, img_dtype) in zip(encoded_images, file_names, metadata):
        if i==0:
            st.toast("The model pipeline is waking up\nInference may be slow on the first image")
        payload = {
            'input': {
				"image": b64_img,
				"dtype": img_dtype,
				"shape": img_shape,
                "img_name": file_name
			}
        }
        
        response = send_request_sync(payload)

        progress = (i + 1) / len(files)
        my_bar.progress(progress, text=f"Processing {file_name} ({i+1}/{len(files)})")
        
        i+=1

    my_bar.empty()
    st.success("Finished processing all images!")

    if return_type in ['Tabular results', 'Both']:
        dummy_return = pd.read_csv('dummy_data.csv')

        st.dataframe(dummy_return)
