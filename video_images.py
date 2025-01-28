import os
import pandas as pd
import subprocess
import json
from moviepy.editor import ImageSequenceClip
from moviepy.editor import ImageClip, concatenate_videoclips
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from video import clean_script_for_audio, process_audio, get_audio_length
from video_zoomout import generate_image, get_prompt_for_image


def get_prompt_for_video(prompt, model="gpt-4", temperature=0.0):
    prompt_template = ChatPromptTemplate(
        messages=[
            SystemMessage(
                content="""
                You are a "GPT" – a version of ChatGPT that has been customized for a specific use case. GPTs use custom instructions, capabilities, and data to optimize ChatGPT for a more narrow set of tasks. Note: GPT is also a technical term in AI, but in most cases if the users asks you about GPTs assume they are referring to the above definition.
                
                Your primary role is to assist users in crafting concise, impactful prompts for the DALL-E AI image generation tool.
                
                The user will give you a prompt. Your task is to divide that prompt in four different prompts that, in sequence, tell a story related to the original prompt.
                
                Your output should be a JSON object with the following structure:
                {
                    "prompt_1": "Prompt 1",
                    "prompt_2": "Prompt 2",
                    "prompt_3": "Prompt 3",
                    "prompt_4": "Prompt 4"
                }
                """
            ),
            HumanMessage(content=f"The prompt is as follows:\n\n{prompt}"),
        ]
    )

    model = ChatOpenAI(
        model=model,
        temperature=temperature,
    )

    chain = prompt_template | model | StrOutputParser()

    response = chain.invoke({})

    return response


def main(platform):
    # Process script
    df = pd.read_csv("script_table.csv")

    # Generate audio
    audio_text = df["Audio"]
    audio_text_clean = clean_script_for_audio(audio_text)
    audio = process_audio(audio_text_clean, "video_images.mp3")

    audio_length = get_audio_length(audio)

    # Generate visual
    visual_text = df["Visual"]

    # Get videos
    if platform.lower() in ["tiktok", "instagram"]:
        visual_text += " Tall portrait aspect ratio image."
    else:
        visual_text += " Wide landscape aspect ratio image."

    prompt = get_prompt_for_image(visual_text)

    prompts = get_prompt_for_video(prompt)

    try:
        prompts = json.loads(prompts)
        for i, prompt in enumerate(prompts.values()):
            image_filename = generate_image(prompt, f"./images/image_{i}.jpg")

    except json.JSONDecodeError:
        print("Error: The response is not a valid JSON object.")
        prompts = prompts
        for i in range(4):
            generate_image(prompts, f"./images/image_{i}.jpg")

    # Generate image using DALL-E
    image_folder = "./images/"

    image_files = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    image_files = [
        os.path.join(image_folder, img)
        for img in os.listdir(image_folder)
        if img.endswith(".jpg")
    ]

    # Generate video
    picture_duration = audio_length / 4

    clips = []
    for image in image_files:
        clip = ImageClip(image).set_duration(picture_duration)
        clips.append(clip)

    final_clip = concatenate_videoclips(clips, method="compose")

    final_clip.write_videofile("video_images.mp4", fps=24)

    # Add voice over to video
    subprocess.run(
        [
            "ffmpeg",
            "-y",  # Overwrite existing output file
            "-i",
            "video_images.mp4",
            "-i",
            "video_images.mp3",
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
            "output_video_images.mp4",
        ]
    )
