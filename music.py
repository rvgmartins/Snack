import scipy
import time
from pydub import AudioSegment
from transformers import AutoProcessor, MusicgenForConditionalGeneration

FILENAME = "music.wav"


def generate_music(duration, filename=FILENAME):
    """
    Generates music using the musicgen-small model.

    This function uses the musicgen-small model from the transformers library to generate music based on given text inputs.
    It takes in a list of text inputs describing the desired music and generates audio values using the model.
    The generated audio values are then saved as a WAV file.

    Returns:
        None
    """

    # Load the musicgen-small processor and model
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

    # Prepare the inputs for the model
    inputs = processor(
        text=[
            "80s pop track with bassy drums and synth",
            "90s rock song with loud guitars and heavy drums",
        ],
        padding=True,
        return_tensors="pt",
    )

    # Generate audio values using the model
    audio_values = model.generate(
        **inputs, do_sample=True, guidance_scale=3, max_new_tokens=duration
    )
    sampling_rate = model.config.audio_encoder.sampling_rate

    # Save the generated audio as a WAV file
    scipy.io.wavfile.write(
        filename, rate=sampling_rate, data=audio_values[0, 0].numpy()
    )


from pydub import AudioSegment


def get_wav_duration(file_path):
    """
    Calculate the duration of a WAV audio file in seconds.

    Args:
        file_path (str): The path to the WAV audio file.

    Returns:
        float: The duration of the audio file in seconds.
    """
    audio = AudioSegment.from_wav(file_path)
    return len(audio) / 1000.0  # pydub returns length in milliseconds


# Call the function to generate music
start_time = time.time()
generate_music(500, FILENAME)
end_time = time.time()

execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

# Call the function to get the duration of the WAV file
wav_duration = get_wav_duration(FILENAME)
print(f"WAV file duration: {wav_duration} seconds")
