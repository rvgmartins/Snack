import pandas as pd
from dotenv import load_dotenv, find_dotenv
from utils import create_video, retrieve_video

import os

if __name__=="__main__":
    # # Load env variables
    _ = load_dotenv(find_dotenv())

    # Define the script
    script_table_path = 'script_table.csv'
    with open(script_table_path, 'r', encoding='utf-8') as file:
        df_script = pd.read_csv(script_table_path)

    # Process into script - clean and concat
    audio_list = df_script.loc[:, 'Audio'].tolist()
    audio_list = [audio.split('"')[1] for audio in audio_list if '"' in audio]
    script = '\n\n'.join(audio_list)

    # Define inputs
    input_text = script
    avatar_id = "josh_lite_20230714"
    avatar_style = "normal"
    voice_id = "1bd001e7e50f421d891986aad5158bc8" 
    path_to_downloaded_video = 'downloaded_video.mp4'
    
    # Create video
    mock = True # Set to False to place the request. Note: with trial API, we have only 5 per day
    initial_response = create_video(input_text, avatar_id, avatar_style, voice_id, mock=mock)
    
    # retrieve video
    # Note: this is not tested, their API and docs are conflicting and we have only 5 tries per day.
    # It should work more or less, but will probably need some plumbing.
    video_id = initial_response.json()['data']['video_id']
    video_url = retrieve_video(initial_response, path_to_downloaded_video)    
    

    print("\n\nAll done in heygen.py")