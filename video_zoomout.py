import pandas as pd
import subprocess
import cv2
import numpy as np
import urllib.request
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from video import (
    clean_script_for_audio,
    process_audio,
    get_audio_length,
)

_ = load_dotenv(find_dotenv())
client = OpenAI()


def get_prompt_for_image(instructions, model="gpt-3.5-turbo", temperature=0.0):
    prompt_template = ChatPromptTemplate(
        messages=[
            SystemMessage(
                content="""
                You are a "GPT" – a version of ChatGPT that has been customized for a specific use case. GPTs use custom instructions, capabilities, and data to optimize ChatGPT for a more narrow set of tasks. Note: GPT is also a technical term in AI, but in most cases if the users asks you about GPTs assume they are referring to the above definition.
                
                Your primary role is to assist users in crafting concise, impactful prompts for the DALL-E AI image generation tool.
                
                The user will give you a set of instructions related to the image he wants. Given that list, here you have the step-by-step process you should follow to generate the prompt:
                - Summarize the instructions provided by the user, ensuring you understand the key elements and desired outcome.
                - Remove any unnecessary details or information that is not relevant to the image generation process.
                - Generate 3 possible prompts that follow the user's instructions, ensuring each prompt is unique and tailored to the user's needs.
                - Explain which of the possible prompts is the most effective and why. Never use prompts that require drawing people and text.
                - Improve the most effective prompt by making it more impactful and appropriate for DALL-E.
                
                *VERY IMPORTANT:* Output just the text of the final prompt.

                """
            ),
            HumanMessage(
                content=f"The instructions are as follows:\n\n{instructions}\n\nGive me only the text of the final prompt."
            ),
        ]
    )

    model = ChatOpenAI(
        model=model,
        temperature=temperature,
    )

    chain = prompt_template | model | StrOutputParser()

    response = chain.invoke({})

    return response


def generate_image(prompt, image_filename="image_0.jpg"):
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        style="natural",
        size="1024x1024",
        quality="standard",
        n=1,
    )

    image_url = response.data[0].url
    urllib.request.urlretrieve(image_url, image_filename)

    return image_filename


def crop(img, x, y, w, h):
    x0, y0 = max(0, x - w // 2), max(0, y - h // 2)
    x1, y1 = x0 + w, y0 + h
    return img[y0:y1, x0:x1]


def main():
    # Process script
    df = pd.read_csv("script_table.csv")

    # Generate audio
    audio_text = df["Audio"]
    audio_text_clean = clean_script_for_audio(audio_text)
    audio = process_audio(audio_text_clean, "video_zoomout.mp3")

    audio_length = get_audio_length(audio)

    # Generate visual
    visual_text = df["Visual"]
    prompt = get_prompt_for_image(visual_text)

    # Generate image using DALL-E
    image_filename = generate_image(prompt)

    # Create video
    video_dim = (1280, 720)
    fps = 25
    duration = 5
    start_center = (0.4, 0.6)
    end_center = (0.5, 0.5)
    start_scale = 0.7
    end_scale = 1.0

    img = cv2.imread(image_filename, cv2.IMREAD_COLOR)
    orig_shape = img.shape[:2]

    num_frames = int(fps * duration)
    frames = []
    for alpha in np.linspace(0, 1, num_frames):
        rx = end_center[0] * alpha + start_center[0] * (1 - alpha)
        ry = end_center[1] * alpha + start_center[1] * (1 - alpha)
        x = int(orig_shape[1] * rx)
        y = int(orig_shape[0] * ry)
        scale = end_scale * alpha + start_scale * (1 - alpha)
        # determined how to crop based on the aspect ratio of width/height
        if orig_shape[1] / orig_shape[0] > video_dim[0] / video_dim[1]:
            h = int(orig_shape[0] * scale)
            w = int(h * video_dim[0] / video_dim[1])
        else:
            w = int(orig_shape[1] * scale)
            h = int(w * video_dim[1] / video_dim[0])
        # crop, scale to video size, and save the frame
        cropped = crop(img, x, y, w, h)
        scaled = cv2.resize(cropped, dsize=video_dim, interpolation=cv2.INTER_LINEAR)
        frames.append(scaled)

    # Write to MP4 file
    vidwriter = cv2.VideoWriter(
        "video_zoomout.mp4", cv2.VideoWriter_fourcc(*"X264"), fps, video_dim
    )
    for frame in frames:
        vidwriter.write(frame)
    vidwriter.release()

    # Add voice over to video
    subprocess.run(
        [
            "ffmpeg",
            "-y",  # Overwrite existing output file
            "-i",
            "video_zoomout.mp4",
            "-i",
            "video_zoomout.mp3",
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
            "output_video_zoomout.mp4",
        ]
    )
