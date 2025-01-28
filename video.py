import io
import subprocess
import requests
import pandas as pd
import os
from pathlib import Path
from random import randint
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from pydub import AudioSegment
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

_ = load_dotenv(find_dotenv(), override=True)
client = OpenAI()

PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
INPUT_VIDEO = "input_video.mp4"
OUTPUT_VIDEO = "output_video.mp4"


def get_audio_length(audio_bytes):
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
    length_ms = len(audio)
    length_seconds = length_ms / 1000  # Convert milliseconds to seconds
    return length_seconds


def get_videos(keywords, n_videos_retrieved=50, orientation="landscape"):
    headers = {"Authorization": PEXELS_API_KEY}
    url = "https://api.pexels.com/videos/search"
    params = {"query": keywords, "per_page": n_videos_retrieved, "orientation": orientation}

    response = requests.get(url, headers=headers, params=params)

    return response.json()


def download_video_from_pexels(api_key, video_id, save_path):
    try:
        url = f"https://api.pexels.com/videos/videos/{video_id}"
        headers = {"Authorization": api_key}
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            video_url = response.json()["video_files"][0]["link"]
            response = requests.get(video_url)
            with open(save_path, "wb") as f:
                f.write(response.content)
            print("Video downloaded successfully!")
        else:
            print("Failed to download video. Status code:", response.status_code)
    except Exception as e:
        print("An error occurred:", e)


def get_keywords_for_visual(input_text, model="gpt-3.5-turbo", temperature=0.0):
    prompt_template = ChatPromptTemplate(
        messages=[
            SystemMessage(
                content="""
                You're an expert in creating keywords to search for stock videos. Your task is to read script notes and return the three best keywords to find a stock video that follows what's asked in the script. You should return the keywords in a comma-separated list. Example:
                
                procrastination, anger, frustration
                """
            ),
            HumanMessage(content=f"The script is as follows:\n\n{input_text}"),
        ]
    )

    model = ChatOpenAI(
        model=model,
        temperature=temperature,
    )

    chain = prompt_template | model | StrOutputParser()

    response = chain.invoke({})

    return response


def clean_script_for_audio(input_text, model="gpt-3.5-turbo", temperature=0.0):
    # Concatenate all text in the input_text column
    full_text = " ".join(input_text.astype(str))

    # Replace all occurrences of 'Narrator:' with an empty string
    cleaned_text = full_text.replace("Narrator:", "")

    prompt_template = ChatPromptTemplate(
        messages=[
            SystemMessage(
                content="""
                You're an expert in cleaning script notes for video text-to-speech production. Your task is to modify a text so it can be used in the OpenAI text-to-speech API, removing all parts that are not to be narrated. Example:
                
                (Upbeat music playing)
                """
            ),
            HumanMessage(content=f"The script notes are as follows:\n\n{cleaned_text}"),
        ]
    )

    model = ChatOpenAI(
        model=model,
        temperature=temperature,
    )

    chain = prompt_template | model | StrOutputParser()

    response = chain.invoke({})

    return response


def process_audio(input_text, filename, model="tts-1", voice="alloy"):
    speech_file_path = Path(__file__).parent / filename
    response = client.audio.speech.create(
        model=model,
        voice=voice,
        input=input_text,
    )

    response.stream_to_file(speech_file_path)

    return response.content


def main(platform):
    # Process script
    df = pd.read_csv("script_table.csv")

    # Generate audio
    audio_text = df["Audio"]
    audio_text_clean = clean_script_for_audio(audio_text)
    audio = process_audio(audio_text_clean, "video.mp3")

    # Generate visual
    visual_text = df["Visual"]
    keywords = get_keywords_for_visual(visual_text)

    print("Keywords:", keywords)

    # Get videos
    if platform.lower() in ["tiktok", "instagram"]:
        orientation = "portrait"
    else:
        orientation = "landscape"
    videos = get_videos(keywords, orientation=orientation)

    # Get audio duration
    audio_length = get_audio_length(audio)

    print("Audio length:", audio_length)

    # Select video to use matching audio length
    selected_videos = []
    tolerance = 1
    while not selected_videos:
        selected_videos = [
            video
            for video in videos["videos"]
            if video["duration"] >= audio_length - tolerance
            and video["duration"] <= audio_length + tolerance
        ]

        for video in selected_videos:
            print("video[duration]:", video["duration"])

        print("audio_length - tolerance: ", audio_length - tolerance)
        print("audio_length + tolerance: ", audio_length + tolerance)

        tolerance += 1

    selected_video = selected_videos[randint(0, len(selected_videos) - 1)]
    selected_video_id = selected_video["id"]
    selected_video_duration = selected_video["duration"]

    print("Selected video duration:", selected_video_duration)

    # Download video from link
    download_video_from_pexels(
        api_key=PEXELS_API_KEY,
        video_id=selected_video_id,
        save_path=INPUT_VIDEO,
    )

    # Add voice over to video
    subprocess.run(
        [
            "ffmpeg",
            "-y",  # Overwrite existing output file
            "-i",
            INPUT_VIDEO,
            "-i",
            "video.mp3",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-strict",
            "experimental",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            OUTPUT_VIDEO,
        ]
    )
