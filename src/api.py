"""
src/api.py
RunPod API Requests
BoMeyering 2025
"""
import streamlit as st
import requests

def send_request_sync(payload):
	""" Send a synchronous request to the API """
	try:
		# Prepare the request
		url = f"https://api.runpod.ai/v2/{st.secrets['v1_endpoint']}/runsync"
		headers = {
			"Authorization": f"Bearer {st.secrets['API_key']}",
			"Content-Type": "application/json"
		}

		# Send request and await response
		response = requests.post(url, headers=headers, json=payload)
		response.raise_for_status()
		result = response.json()

		# Report output
		return result

	except Exception as e:
		print(f"Error sending request for image {payload['img_name']}: {str(e)}")