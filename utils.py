import yaml
import os
import requests
import time
import json

# This class is to simulate a HTTP request, because we get a very tight quota for generating videos via API
class MockResponse:
    def __init__(self, status_code, json_data=None, text_data=None):
        self.status_code = status_code
        self.data = json_data
        self.text_data = text_data

    def json(self):
        return self.data

    def text(self):
        return self.text_data

def create_video(input_text, avatar_id, avatar_style, voice_id, mock=False):
    """
    Create video using HeyGen API

    Args:
        input_text (str): Text to be converted to video (script)
        avatar_id (str): Avatar ID
        avatar_style (str): Avatar style
        voice_id (str): Voice ID
        mock (bool): Whether to mock the API call
    
    Returns:
        initial_response (requests.Response): Response from the API call

    Notes:
        Requires: os.environ["HEYGEN_API_TOKEN"] to be set
    """
    
    # Load API key from environment variables
    heygen_api_key = os.environ["HEYGEN_API_TOKEN"]

    # Define the API endpoints
    initial_url = 'https://api.heygen.com/v2/video/generate'

    # Define the headers
    headers = {
        'X-Api-Key': heygen_api_key,
        'Content-Type': 'application/json'
    }

    payload = {
        "video_inputs": [
            {
            "character": {
                "type": "avatar",
                "avatar_id": avatar_id,
                "avatar_style": avatar_style
            },
            "voice": {
                "type": "text",
                "input_text": input_text,
                "voice_id": voice_id
            }
            }
        ],
        "test": True,
        "caption": False,
        "dimension": {
            "width":  1024,
            "height": 768
        }
    }
    
    if mock:
        initial_response = MockResponse(200, 
                                json_data={'code': 100, 'data': {'video_id': 'b0d9ce8168c2472a9be715a9550dd182'}, 'message': 'Success'}, 
                                text_data="Example text response")
    else:
        initial_response = requests.post(initial_url, 
                                         data=json.dumps(payload), 
                                         headers=headers) # Trial API: Max 5 per day
    
    print(f"Initial response (in the function): {initial_response.json()}")

    return initial_response

def retrieve_video(video_id, path_to_downloaded_video=None):
    """
    Retrieve video using HeyGen API.

    Note: this hasn't been tested because they changed the API to v2, and currently seem to have conflicting versions.
    It's not worth it to solve given our daily limit of 5 calls.

    Args;
        video_id: ID of the video to retrieve
        path_to_downloaded_video (str): Path to save the downloaded video

    Returns:
        video_url (str): URL of the video

    Raises:
        Exception: If the video generation fails
    
    Notes:
        Requires os.environ["HEYGEN_API_TOKEN"] to be set
    """

    # Load API key from environment variables
    heygen_api_key = os.environ["HEYGEN_API_TOKEN"]

    # Define the API endpoints
    status_url = 'https://api.heygen.com/v1/video_status.get'
    
    # Define the headers
    headers = {
        'X-Api-Key': heygen_api_key,
        'Content-Type': 'application/json'
        
    }

    # Initiate the video retrieval
    print(f"Retrieving video. ID: {video_id}")

    # Check the status periodically
    while True:
        status_response = requests.get(f"{status_url}?video_id={video_id}", headers=headers)
        print(status_response.json())
        if status_response.status_code == 100:
            status = status_response.json()["data"]["status"]
            print(f"Current status: {status}")
            if status == "completed":
                video_url = status_response.json()["data"]['video_url']

                if path_to_downloaded_video: # Optionally download the video
                    video_response = requests.get(video_url)
                    if video_response.status_code == 200:
                        with open(path_to_downloaded_video, 'wb') as file:
                            file.write(video_response.content)
                        print(f"Video successfully downloaded as {path_to_downloaded_video}")
                
                return video_url

            elif status == "error":
                raise Exception("An error occurred during video generation.")
        else:
            raise Exception("Failed to check video status.")

        time.sleep(10)  # Wait for 5 seconds before checking again

def read_yaml_config(file_path):
    """
    Reads a YAML configuration file and returns the parsed content.

    Args:
        file_path (str): The path to the YAML configuration file.

    Returns:
        dict: The parsed YAML content.

    """
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
    return config
