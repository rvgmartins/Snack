# Snack 2.0

Snack 2.0 is a Streamlit application that uses AI to generate personalized campaign recommendations based on user input and analysis of relevant data. 

## Installing

To install the dependencies, run `pip install -r requirements.txt`. We recommend using a virtual environment to manage dependencies.

## Running the code

To run the code locally, do `streamlit run st_poc.py`. This will launch a streamlit instance in localhost.

## Environment variables

For the code to run, one must set two environment variables via `.env`: OPENAI_API_KEY and PEXELS_API_KEY. OPENAI_API_KEY can be obtained via the [OpenAI platform](https://platform.openai.com/) and is used for the LLM interactions. PEXELS_API_KEY can be obtained via the [Pexels plaftorm](https://www.pexels.com/api/) and is used to retrieve images. 

Optionally one can set HEYGEN_API_TOKEN which is obtained via [Heygen](https://docs.heygen.com/) and is used to generate avatar videos. This is only required if one is using the `heygen.py` script separately (see the "Video generation" section).
 
To set the environment variables, first obtain the required keys, then copy the `.env_template` file, rename it as `.env` and fill in the required values.

## Video generation

The video generation is currently done separately and is not part of the main product. To run the video generation, follow use the `heygen.py` script. A HeyGen API Token is required (set via .env).

We're using the HeyGen API to create videos from the script.

For a list of voices:
https://docs.google.com/spreadsheets/d/1nSZCWmazqgr5CY-ShO-UV376xlmbcDttGl75JYjnp0U/edit#gid=0

To get an updated list of voices, run this snippet:

```
voices_url = 'https://api.heygen.com/v1/voice.list'

# Define the headers
headers = {
    'X-Api-Key': heygen_api_key,
    'Content-Type': 'application/json'
}

voices = requests.get(voices_url, headers=headers) 

import pandas as pd
df = pd.DataFrame(voices.json()["data"][])
df.to_csv("voices.csv")
```

For avatars, it's similar but a different endpoint.


## Known issues 
* [Issues related to `ffmpeg` and/or `ffprobe`](https://github.com/jiaaro/pydub/issues/404)
  * You might need to download `ffmpeg`. In MacOS, use homebrew.
* Don't forget to update `.env`


## Future improvements

This is a list of known issues/limitations that could be addressed:

* The identification of the topic is now more or less hardcoded - it must be generic. This requires integrating with Tubular dynamically instead of using a static csv.
* The analysis part runs every time
* The state management mixes "steps" with "actions" (like activate_create_scripts)
* Generate scripts independently, i.e. run a loop with X calls instead of asking GPT to create X scripts.
* Make the video (the images-based one) scene-by-scene, based on the script. This way we avoid the issue of matching lengths.
* Replace the static videos (heygen and sync-labs) with API calls or remove them altogether.
* Add a toggle for dummy data, so we save time and money when testing.

