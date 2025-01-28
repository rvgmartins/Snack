import os
import random
import pandas as pd
import streamlit as st
import time
import matplotlib.pyplot as plt
from openai import OpenAI
from wordcloud import WordCloud
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import YoutubeLoader
from utils import read_yaml_config
from video import main as video_main
from video_images import main as video_images_main

def reset_session():
    """
    Deletes all keys in the session state, effectively resetting the session.
    """
    for key in st.session_state.keys():
        del st.session_state[key]

def move_to_step_2():
    # update cost
    st.session_state.current_cost += st.session_state.cost_dictionary[st.session_state.current_campaign_step]

    # Update step
    st.session_state.current_campaign_step = "Step 2: Platform Selection"
    st.session_state.display_chatbot = False

    # at this point, we need to get the topic
    summary = generate_journey_summary(st.session_state.messages)
    topic = get_journey_topic(st.session_state.messages)
    
    # export as state variables
    st.session_state["summary"] = summary
    st.session_state["topic"] = topic.lower()

    


def move_to_step_3():
    # update cost
    st.session_state.current_cost += st.session_state.cost_dictionary[st.session_state.current_campaign_step]

    st.session_state.get_most_popular_activated = True
    st.session_state.current_campaign_step = "Step 3: Engagement Analysis"

    st.session_state.selected_platform = selected_platform

def move_to_step_4():
    # update cost
    st.session_state.current_cost += st.session_state.cost_dictionary[st.session_state.current_campaign_step]

    st.session_state.current_campaign_step = "Step 4: Script Generation"
    st.session_state["display_analysis_insights"] = True


def move_to_step_5():
    # update cost
    st.session_state.current_cost += st.session_state.cost_dictionary[st.session_state.current_campaign_step]

    st.session_state.current_campaign_step = "Step 5: Influencer Collaboration"
    st.session_state["display_generated_scripts"] = True
    st.session_state["create_scripts_activated"] = False

def move_to_step_6(df):
    # update cost
    st.session_state.current_cost += st.session_state.cost_dictionary[st.session_state.current_campaign_step]

    st.session_state.current_campaign_step = "Step 6: Content Production"
    st.session_state["df_influencers"] = df
    st.session_state["create_videos_activated"] = True

def move_to_step_7():
    # update cost
    st.session_state.current_cost += st.session_state.cost_dictionary[st.session_state.current_campaign_step]

    st.session_state.current_campaign_step = "Final step: Review and Launch"

    st.rerun()



def activate_analyze_videos():
    st.session_state.analyze_videos_activated = True
    st.session_state.create_scripts_activated = False


# def activate_get_most_popular():
#     NOTE: replaced with "move_to_step_3"
#     st.session_state.get_most_popular_activated = True


def activate_create_analysis_dataframe():
    st.session_state.create_analysis_dataframe_activated = True


def activate_create_scripts():
    st.session_state.create_scripts_activated = True

def enrich_dataframe(transcript, model="gpt-3.5-turbo", temperature=0.0):
    system_prompt = "You are a video transcript analyzer. Your job is to extract information about the characteristics of a video from its transcript."
    human_prompt = """I'll give you a transcript of a video. Based on that, give me a list of 5 keywords about the video. The output should be like in the following example:
    
    anger, fear, happiness, sadness, surprise
    
    Here is the transcript of the video:
    
    [START TRANSCRIPT]
    {transcript}
    [END TRANSCRIPT]
    
    Remember: just 5 keywords separated by commas.
    """.format(
        transcript=transcript
    )

    prompt_template = ChatPromptTemplate(
        messages=[
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]
    )

    model = ChatOpenAI(
        model=model,
        temperature=temperature,
    )

    chain = prompt_template | model | StrOutputParser()
    keywords = chain.invoke({})

    return keywords


def format_dataframe(dataframe):
    # Keep columns of interest
    dataframe = dataframe[
        [
            "Video_Title",
            "Platform",
            "Video_URL",
            "YouTube_Category",
            "Likes",
            "Comments",
            "Published_Date",
            "Creator",
            "Creator_FollowerCount_or_YouTube_Subscribers",
            "Duration (seconds)",
            "transcript",
            "view_count",
            "keywords",
        ]
    ]

    # Rename columns
    dataframe.columns = [
        "Title",
        "Platform",
        "URL",
        "Category",
        "Likes",
        "Comments",
        "Published Date",
        "Creator",
        "Subscribers",
        "Duration",
        "Transcript",
        "Views",
        "Keywords",
    ]

    # Reorder columns
    dataframe = dataframe[
        [
            "Title",
            "Platform",
            "Creator",
            "Subscribers",
            "Views",
            "Likes",
            "Comments",
            "Category",
            "Keywords",
            "Transcript",
            "URL",
            "Published Date",
            "Duration",
        ]
    ]

    return dataframe

@st.cache_data(show_spinner=False)
def generate_insights(transcripts, model="gpt-4-turbo", temperature=0.0):

    prompt_template = ChatPromptTemplate(
        messages=[
            SystemMessage(
                content=f"""You are a GPT. Your role is to analyze transcripts of popular videos and generate reports that highlight commonalities between them, providing insights to help users understand what makes these videos popular. Focus on identifying themes, language, structure, emotional tone, and any recurring patterns, such as specific topics or storytelling techniques. Your goal is to empower users to create new, popular videos based on these insights. You'll work with whatever transcripts are provided, drawing conclusions based solely on the content of those transcripts without seeking additional information. Provide a detailed report with examples from the transcripts, organized into clear sections for themes, language, structure, and emotional tone. This format is most beneficial when passing insights onto a script-writing GPT, as it offers structured, actionable insights that can be directly incorporated into new scripts. Adapt the tone of your analysis to match the tone of the transcripts provided. This helps in maintaining the essence and style of the original content, making your insights more tailored and relevant to the type of videos being analyzed.
                """
            ),
            HumanMessage(
                content=f"Compare the following transcripts:\n\n{transcripts}"
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

@st.cache_data(show_spinner=False)
def create_scripts(
    analysis_insights, n_scripts, platform, model="gpt-3.5-turbo", temperature=0.4
):
    prompt_template = ChatPromptTemplate(
        messages=[
            SystemMessage(
                content=f"You're a video script writer. Your task is to write scripts for a video based on other popular videos about the same topic. Each script you write must have a table format, with three columns: Scene, Visual, Audio. The Audio be no longer than 200 words and it has to have a narrator."
            ),
            HumanMessage(
                content=f"Generate {n_scripts} scripts for a {platform} video based on this report of what worked well in some popular videos about the topic:\n\n{analysis_insights}"
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


def extract_video_info(url, dataframe, index):
    loader = YoutubeLoader.from_youtube_url(
        url,
        add_video_info=True,
        language=["en"],
        translation="en",
    )
    video = loader.load()[0]
    transcript = video.page_content

    dataframe.loc[index, "transcript"] = transcript
    dataframe.loc[index, "thumbnail_url"] = video.metadata["thumbnail_url"]
    dataframe.loc[index, "title"] = video.metadata["title"]
    dataframe.loc[index, "view_count"] = video.metadata["view_count"]
    # dataframe.loc[index, "length"] = video.metadata["length"]
    # dataframe.loc[index, "publish_date"] = video.metadata["publish_date"]
    # dataframe.loc[index, "author"] = video.metadata["author"]

    return dataframe


def prepare_data_for_thumbnails(df):
    thumbnails = [thumbnail for thumbnail in df["thumbnail_url"].tolist()]
    titles = [title for title in df["title"].tolist()]
    view_counts = [view_count for view_count in df["view_count"].tolist()]

    captions = [
        f"{title} | Views: {int(view_count)}"
        for title, view_count in zip(titles, view_counts)
    ]

    return thumbnails, captions



def generate_wordcloud_figure(
    text, width=800, height=400, background_color="white", colormap="viridis"
):
    """
    Generates a word cloud and returns a matplotlib figure object.
    Parameters:
    - text (str): The text data from which to generate the word cloud.
    - width (int, optional): The width of the word cloud image. Default is 800.
    - height (int, optional): The height of the word cloud image. Default is 400.
    - background_color (str, optional): The background color for the word cloud image. Default is 'white'.
    Returns:
    - fig (matplotlib.figure.Figure): The figure object containing the word cloud.
    """
    # Create a word cloud object
    wordcloud = WordCloud(
        width=width, height=height, background_color=background_color, colormap=colormap
    ).generate(text)
    # Create a figure
    fig, ax = plt.subplots(figsize=(width / 100, height / 100))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    return fig


def transform_text_to_csv(text):
    # Split the text into lines
    lines = text.split("\n")

    # Initialize an empty list to store the data
    data = []

    # Iterate over each line
    for line in lines:
        # Split the line into columns
        columns = line.split("|")

        # Remove leading and trailing whitespaces from each column
        columns = [col.strip() for col in columns]

        # Append the columns to the data list
        data.append(columns)

    # Create a DataFrame from the data
    df = pd.DataFrame(data)

    # Set the column names
    df.columns = ["Scene", "Visual", "Audio"]

    # Save DataFrame as CSV
    df.to_csv("script_table.csv", index=False)


def convert_text_to_csv(response, model="gpt-3.5-turbo", temperature=0.0):
    prompt_template = ChatPromptTemplate(
        messages=[
            SystemMessage(
                content="""
                            You're a helpful assistant. Your task is to convert a text file with markdown-formatted text to a CSV file. The text file contains a table with three columns: Scene, Visual, and Audio. Each row in the table represents a scene in a video script. Important: To do this only for Script 1, give me only the csv data, no other comments, and use '|' as the delimiter character.
                            """
            ),
            HumanMessage(content=f"The text is as follows:\n\n{response}"),
        ]
    )

    model = ChatOpenAI(
        model=model,
        temperature=temperature,
    )

    chain = prompt_template | model | StrOutputParser()
    response = chain.invoke({})

    return response


def generate_journey_summary(messages_history, model="gpt-3.5-turbo", temperature=0.0):
    """
    Generates a summary of the users input, that will be used throughout the application.

    In the future, this should return a structured format (e.g. json) that we use elsewhere.
    """
    user_messages_history = "\n".join([m["content"] for m in messages_history if m["role"] == "user"])

    prompt_template = ChatPromptTemplate(
        messages=[
            SystemMessage(
                content="""
                You're a helpful assistant. Your task is to generate a summary of the user's input, which will be used throughout the application. The user's input includes details about a video he wants to make, so your summary should include the key points of the video, such as the topic, which platform it is meant for, and any other significant characteristics the user mentioned in his conversation. Your summary should be concise and informative, providing a clear overview of the video's content, purpose and characteristics.

                Don't include the word 'summary' in your response.
                            """
            ),
            HumanMessage(content=f"Here is the history of the user's input:\n\n{user_messages_history}"),
        ]
    )

    model = ChatOpenAI(
        model=model,
        temperature=temperature,
    )

    chain = prompt_template | model | StrOutputParser()
    response = chain.invoke({})

    return response



def get_journey_topic(messages_history, model="gpt-3.5-turbo", temperature=0.0):
    """
    Generates a summary of the users input, that will be used throughout the application.

    In the future, this should return a structured format (e.g. json) that we use elsewhere.
    """


    prompt_template = ChatPromptTemplate(
        messages=[
            SystemMessage(
                content="""
                You're a helpful assistant. Your task is to identify the topic of the conversation based on the conversation. The topic is either PROCRASTINATION, BEAUTY or MORTGAGE. 

                Return only one of these values as your answer. If you're not sure, return OTHER. REMEMBER: Return a SINGLE WORD.
                            """
            ),
            HumanMessage(content=f"Here is the history of the user's input:\n\n{messages_history}"),
        ]
    )

    model = ChatOpenAI(
        model=model,
        temperature=temperature,
    )

    chain = prompt_template | model | StrOutputParser()
    response = chain.invoke({})

    return response


_ = load_dotenv(find_dotenv())
client = OpenAI()

# Load configuration
config = read_yaml_config("config.yaml")

# Activate mock run
mock_run = True

# Initialize session state
if "get_most_popular_activated" not in st.session_state:
    st.session_state.display_chatbot = True

if "get_most_popular_activated" not in st.session_state:
    st.session_state.get_most_popular_activated = False

if "create_analysis_dataframe_activated" not in st.session_state:
    st.session_state.create_analysis_dataframe_activated = False

if "display_settings_activated" not in st.session_state:
    st.session_state.display_settings_activated = False

if "analyze_videos_activated" not in st.session_state:
    st.session_state.analyze_videos_activated = False

if "create_scripts_activated" not in st.session_state:
    st.session_state.create_scripts_activated = False

if "create_videos_activated" not in st.session_state:
    st.session_state.create_videos_activated = False

if "run" not in st.session_state:
    st.session_state.run = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "campaign_briefing" not in st.session_state:
    st.session_state.campaign_briefing = {
        "first_message": None,
        "campaign_name": None,
        "objective": None,
        "target_audience": None,
        "key_messages": None,
        "preferred_platforms": None,
        "budget_and_timeline": None
    }

if "current_campaign_step" not in st.session_state:
    st.session_state.current_campaign_step = "Step 1: Campaign Creation"

if "selected_platform" not in st.session_state:
    st.session_state.selected_platform = None

if "selected_platform" not in st.session_state:
    st.session_state.selected_platform = None

if "display_analysis_insights" not in st.session_state:
    st.session_state.display_analysis_insights = False

if "display_generated_scripts" not in st.session_state:
    st.session_state.display_generated_scripts = False

if "analysis_insights" not in st.session_state:
    st.session_state.analysis_insights = None

if "generated_scripts" not in st.session_state:
    st.session_state.generated_scripts = None

if "n_scripts" not in st.session_state:
    st.session_state.n_scripts = 0

if "df_influencers" not in st.session_state:
    st.session_state.df_influencers = None

if "cost_dictionary" not in st.session_state:
    st.session_state.cost_dictionary = {
        "Step 1: Campaign Creation": 50,
        "Step 2: Platform Selection": 0,
        "Step 3: Engagement Analysis": 20,
        "Step 4: Script Generation": 30,
        "Step 5: Influencer Collaboration": 100,
        "Step 6: Content Production": 50,
        "Final step: Review and Launch": 0,
    }

if "current_cost" not in st.session_state:
    st.session_state.current_cost = 0

# Initialize app
st.set_page_config(page_title="Snack 2.0")
st.title("Snack 2.0")
st.divider()

st.markdown(f"## 🚀 Create Campaign")
st.markdown("Welcome! To get started, tell the assistant about the campaign you'd like to create.")

# Side bar

with st.sidebar:
    st.markdown("#### Currently on:")
    st.markdown(f"## 📝 {st.session_state.current_campaign_step}")
    st.markdown("*Total campaign cost*: $" + str(st.session_state.current_cost))

    # After first step, show summary
    if st.session_state.current_campaign_step != "Step 1: Campaign Creation":
        st.divider()
        st.markdown("## Overview")
        st.markdown(st.session_state["summary"])

    if st.session_state.current_campaign_step == "Step 2: Platform Selection":
        # show briefing
        with st.expander("✅ Step 1: Campaign Creation"):
            for key, value in st.session_state.campaign_briefing.items():
                if key != "first_message":
                    st.markdown(f"**{key.replace('_', ' ').title()}**: {value}")
            
            # Cost for this step
            st.markdown(f"*Cost*: ${st.session_state.cost_dictionary['Step 1: Campaign Creation']}")

    if st.session_state.current_campaign_step == "Step 3: Engagement Analysis":
        
        # show briefing
        with st.expander("✅ Step 1: Campaign Creation"):
            for key, value in st.session_state.campaign_briefing.items():
                if key != "first_message":
                    st.markdown(f"**{key.replace('_', ' ').title()}**: {value}")
                
            # Cost for this step
            st.markdown(f"*Cost*: ${st.session_state.cost_dictionary['Step 1: Campaign Creation']}")

        # show platforms
        with st.expander("✅ Step 2: Platform Selection"):
            st.markdown(f"{st.session_state['selected_platform']}")

            # Cost for this step
            st.markdown(f"*Cost*: ${st.session_state.cost_dictionary['Step 2: Platform Selection']}")

    if st.session_state.current_campaign_step == "Step 4: Script Generation":
        
        # show briefing
        with st.expander("✅ Step 1: Campaign Creation"):
            for key, value in st.session_state.campaign_briefing.items():
                if key != "first_message":
                    st.markdown(f"**{key.replace('_', ' ').title()}**: {value}")

            # Cost for this step
            st.markdown(f"*Cost*: ${st.session_state.cost_dictionary['Step 1: Campaign Creation']}")

        # show platforms
        with st.expander("✅ Step 2: Platform Selection"):
            st.markdown(f"{st.session_state['selected_platform']}")

            # Cost for this step
            st.markdown(f"*Cost*: ${st.session_state.cost_dictionary['Step 2: Platform Selection']}")

        # analysis
        with st.expander("✅ Step 3: Engagement Analysis"):
            st.markdown(f"Engagement analysis completed successfully.")

            # Cost for this step
            st.markdown(f"*Cost*: ${st.session_state.cost_dictionary['Step 3: Engagement Analysis']}")

    if st.session_state.current_campaign_step == "Step 5: Influencer Collaboration":
        
        # show briefing
        with st.expander("✅ Step 1: Campaign Creation"):
            for key, value in st.session_state.campaign_briefing.items():
                if key != "first_message":
                    st.markdown(f"**{key.replace('_', ' ').title()}**: {value}")

            # Cost for this step
            st.markdown(f"*Cost*: ${st.session_state.cost_dictionary['Step 1: Campaign Creation']}")

        # show platforms
        with st.expander("✅ Step 2: Platform Selection"):
            st.markdown(f"{st.session_state['selected_platform']}")

            # Cost for this step
            st.markdown(f"*Cost*: ${st.session_state.cost_dictionary['Step 2: Platform Selection']}")

        # analysis
        with st.expander("✅ Step 3: Engagement Analysis"):
            st.markdown(f"Engagement analysis completed successfully.")

            # Cost for this step
            st.markdown(f"*Cost*: ${st.session_state.cost_dictionary['Step 3: Engagement Analysis']}")

        # script generation
        with st.expander("✅ Step 4: Script Generation"):
            st.markdown(f"{st.session_state.n_scripts} scripts generated successfully.")  

            # Cost for this step
            st.markdown(f"*Cost*: ${st.session_state.cost_dictionary['Step 4: Script Generation']}")

    if st.session_state.current_campaign_step == "Step 6: Content Production":
        
        # show briefing
        with st.expander("✅ Step 1: Campaign Creation"):
            for key, value in st.session_state.campaign_briefing.items():
                if key != "first_message":
                    st.markdown(f"**{key.replace('_', ' ').title()}**: {value}")

            # Cost for this step
            st.markdown(f"*Cost*: ${st.session_state.cost_dictionary['Step 1: Campaign Creation']}")

        # show platforms
        with st.expander("✅ Step 2: Platform Selection"):
            st.markdown(f"{st.session_state['selected_platform']}")

            # Cost for this step
            st.markdown(f"*Cost*: ${st.session_state.cost_dictionary['Step 2: Platform Selection']}")

        # analysis
        with st.expander("✅ Step 3: Engagement Analysis"):
            st.markdown(f"Engagement analysis completed successfully.")

            # Cost for this step
            st.markdown(f"*Cost*: ${st.session_state.cost_dictionary['Step 3: Engagement Analysis']}")

        # script generation
        with st.expander("✅ Step 4: Script Generation"):
            st.markdown(f"{st.session_state.n_scripts} scripts generated successfully.")

            # Cost for this step
            st.markdown(f"*Cost*: ${st.session_state.cost_dictionary['Step 4: Script Generation']}")
        
        # influencers
        with st.expander("✅ Step 5: Influencer Collaboration"):
            selected_influencers = st.session_state.df_influencers.loc[st.session_state.df_influencers["Selected"] == True, "Creator Name"].tolist()
            
            selected_influencers_bullet_list = "\n".join(f"* {item}" for item in selected_influencers)
            
            st.markdown(selected_influencers_bullet_list)

            # Cost for this step
            st.markdown(f"*Cost*: ${st.session_state.cost_dictionary['Step 5: Influencer Collaboration']}")

    if st.session_state.current_campaign_step == "Final step: Review and Launch":
        
        # show briefing
        with st.expander("✅ Step 1: Campaign Creation"):
            for key, value in st.session_state.campaign_briefing.items():
                if key != "first_message":
                    st.markdown(f"**{key.replace('_', ' ').title()}**: {value}")

            # Cost for this step
            st.markdown(f"*Cost*: ${st.session_state.cost_dictionary['Step 1: Campaign Creation']}")

        # show platforms
        with st.expander("✅ Step 2: Platform Selection"):
            st.markdown(f"{st.session_state['selected_platform']}")

            # Cost for this step
            st.markdown(f"*Cost*: ${st.session_state.cost_dictionary['Step 2: Platform Selection']}")

        # analysis
        with st.expander("✅ Step 3: Engagement Analysis"):
            st.markdown(f"Engagement analysis completed successfully.")

            # Cost for this step
            st.markdown(f"*Cost*: ${st.session_state.cost_dictionary['Step 3: Engagement Analysis']}")

        # script generation
        with st.expander("✅ Step 4: Script Generation"):
            st.markdown(f"{st.session_state.n_scripts} scripts generated successfully.")

            # Cost for this step
            st.markdown(f"*Cost*: ${st.session_state.cost_dictionary['Step 4: Script Generation']}")
        
        # influencers
        with st.expander("✅ Step 5: Influencer Collaboration"):
            selected_influencers = st.session_state.df_influencers.loc[st.session_state.df_influencers["Selected"] == True, "Creator Name"].tolist()
            
            selected_influencers_bullet_list = "\n".join(f"* {item}" for item in selected_influencers)
            
            st.markdown(selected_influencers_bullet_list)

            # Cost for this step
            st.markdown(f"*Cost*: ${st.session_state.cost_dictionary['Step 5: Influencer Collaboration']}")
        
        # videos
        with st.expander("✅ Step 6: Content Production"):
            st.markdown(f"Videos created successfully.")

            # Cost for this step
            st.markdown(f"*Cost*: ${st.session_state.cost_dictionary['Step 6: Content Production']}")

        if st.button("🚀 Launch Campaign", type="primary"):
            st.balloons()
        

###
# Here we add the chatbot
# it returns a JSON with the user input

if st.session_state.display_chatbot:
    chatbot_system_prompt = """You're a helpful video creation assistant. Your job is to help users create videos for their marketing campaign by providing guidance, suggestions, and feedback. You'll work with users to understand their video ideas, provide insights, and help them develop their concepts into engaging video content. Your goal is to support users in creating high-quality videos that resonate with their audience and achieve the goals of their campaign. Make sure to review the user's input carefully, ask clarifying questions, and provide constructive feedback to help them refine their ideas.

    Here's a checklist of things you need to make sure you ask the user:

    * Campaign Name
    * Objective
    * Target Audience
    * Key Messages
    * Preferred Platforms
    * Budget and Timeline
    
    Ask for each of these one at at time. Keep your interactions short. Inform the user when you think she has provided enough information.

    REMEMBER: Ask each question one at a time. Don't ask for all the information at once.
    """

    
    with st.container(height=300):

        if prompt := st.chat_input("Tell me about the campaign briefing you have in mind."):
            
            # save to messages
            st.session_state.messages.append({"role": "user", "content": prompt})

            # save to briefing env variable. This relies on the LLM asking the questions in the right order.
            # This is fragile and we should move either to a form or to a model that generates the briefing based on the conversation.
            for key in ["first_message", "campaign_name", "objective", "target_audience", "key_messages", "preferred_platforms", "budget_and_timeline"]:
                if st.session_state.campaign_briefing[key] is None:
                    st.session_state.campaign_briefing[key] = prompt
                    break


            with st.chat_message("assistant"):
                
                stream = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": chatbot_system_prompt}
                    ] + 
                    [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                    stream=True,
                )
                response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})

    st.button("➡️ Step 2: Platform Selection", on_click=move_to_step_2)

else:
    st.button("🆕 Create a new campaign", on_click=reset_session)


if st.session_state.current_campaign_step=="Step 2: Platform Selection":
    st.divider()
    st.write("## ⚙️ Platform Selection")

    with st.form(key="choose_platforms_form"):

        selected_platform = st.selectbox(
            label = "Choose the platforms you want to target",
            options=["YouTube", "TikTok", "Instagram", "Facebook"], 
            placeholder = "Choose the platforms you want to target",
            label_visibility="collapsed"
            )
        
        platforms_submit_button = st.form_submit_button(label="➡️ Step 3: Engagement Analysis", on_click=move_to_step_3)


# Get most popular videos
if st.session_state["get_most_popular_activated"]:
    
    # Get data on the most popular videos
    st.divider()
    st.markdown(f"## 🏆 Engagement Analysis")

    with st.spinner("Searching videos..."):
        if mock_run:
            time.sleep(random.uniform(3, 6))  # To make it look more real

            if st.session_state["topic"] == "mortgage":
                dataframe = pd.read_csv("video_info_mortgage.csv")
            elif st.session_state["topic"] == "procrastination":
                dataframe = pd.read_csv("video_info_procrastination.csv")
            elif st.session_state["topic"] == "beauty":
                dataframe = pd.read_csv("video_info_beauty.csv")
            else:
                # streamlit display warning
                st.warning("No videos found that match this topic. Please start again.")
                time.sleep(2)
                reset_session()
                st.stop()

        else:
            # Get data from tubular
            df_tubular = pd.read_csv(config["tubular_filename"])
            
            dataframe = df_tubular.copy()

            counter = 0
            for index, row in df_tubular.iterrows():
                if counter >= config["max_video"]:
                    break

                url = row["Video_URL"]

                try:
                    dataframe = extract_video_info(url, dataframe, index)
                    dataframe.to_csv(config["video_info_filename"])

                    counter += 1

                except Exception as e:
                    print(f"Error loading video: {e}")
                    continue

    # Display thumbnails
    st.markdown(f"#### Most popular videos:")
    df = dataframe[dataframe["thumbnail_url"].notna()]  # Remove rows without thumbnail

    thumbnails, captions = prepare_data_for_thumbnails(df)

    st.image(thumbnails, width=350, caption=captions)

    # Start next step
    activate_create_analysis_dataframe()

# Create analysis dataframe
if st.session_state["create_analysis_dataframe_activated"]:

    @st.cache_data(show_spinner=False)
    def create_analysis_dataframe():
        for index, row in df.iterrows():
            keywords = enrich_dataframe(row["transcript"])
            df.loc[index, "keywords"] = keywords

        st.markdown(f"#### Extracted data:")
        st.dataframe(format_dataframe(df))

    with st.spinner("Extracting data from videos..."):
        # Display dataframe
        create_analysis_dataframe()

        # Display wordcloud
        st.write("#### Wordcloud of all transcripts:")
        wordlcloud_text = " ".join(df["transcript"].str.lower().tolist())
        fig = generate_wordcloud_figure(wordlcloud_text)
        st.pyplot(fig)

# Run the analysis as part of step 3
if st.session_state["current_campaign_step"] == "Step 3: Engagement Analysis":
    # Raise error if video info is missing
    if not os.path.isfile(config["video_info_filename"]):
        st.error("No video information available. Please redo the process.")
        st.stop()

    # Write analysis insights
    
    with st.spinner("Performing analysis..."):
        transcripts = df["transcript"].tolist()
        transcripts = [
            transcript for transcript in transcripts if str(transcript) != "nan"
        ]

        analysis_insights = generate_insights(transcripts)

        st.session_state["analysis_insights"] = analysis_insights

        # Display insights
        st.divider()
        st.write("## 🤔 Analysis")
        st.write(st.session_state["analysis_insights"])

    st.button("➡️ Step 4: Script Generation", on_click=move_to_step_4)

if st.session_state["display_analysis_insights"]:
    st.divider()
    st.write("## 🤔 Analysis")
    st.write(st.session_state["analysis_insights"])

# End of step 3

# Step 4
if st.session_state["current_campaign_step"] == "Step 4: Script Generation":
    st.divider()
    st.write("## ⚙️ Settings for Script Generation")

    with st.form(key="settings_form"):
        
        n_scripts = st.selectbox(
            "Define the number of scripts", ("1", "2", "3", "4", "5")
        )
        st.session_state["n_scripts"] = int(n_scripts)
        
        submit_button = st.form_submit_button(label="🚀 Generate Scripts", on_click=activate_create_scripts)

if st.session_state["create_scripts_activated"]:
    # Write scripts
    n_scripts = st.session_state["n_scripts"]
    analysis_insights = st.session_state["analysis_insights"]

    st.divider()
    with st.spinner(f"Creating scripts..."):

        platform = st.session_state["selected_platform"]

        generated_scripts = create_scripts(analysis_insights, n_scripts, platform)
        
        st.session_state["generated_scripts"] = generated_scripts

    # Display scripts
    st.markdown(f"## ✍️ Suggested scripts")
    st.write(st.session_state["generated_scripts"])

    # Start next step
    st.button("➡️ Step 5: Influencer Collaboration", on_click=move_to_step_5)

if st.session_state["display_generated_scripts"]:
    st.markdown(f"## ✍️ Suggested scripts")
    st.write(st.session_state["generated_scripts"])


if st.session_state["current_campaign_step"] == "Step 5: Influencer Collaboration":

    raw_df = pd.read_csv("creators.csv")
    platform = st.session_state["selected_platform"]
    topic = st.session_state["topic"]

    # TODO: make it not be hardcoded
    map_topic_to_genre = {"beauty": "Beauty",
                        "mortgage": "Home & DIY",
                        "procrastination": "People & Blogs"}

    map_platform_to_follower_columns = {"YouTube": "YT_Current_All_Time_Subscribers",
                    "TikTok": "TikTok_Current_All_Time_Followers", 
                    "Instagram": "Instagram_Current_All_Time_Followers", 
                    "Facebook": "Facebook_Current_All_Time_Followers" 
                    }

    map_platform_to_growth_columns = {"YouTube": "YT_%_30_days_sub_growth",
                    "TikTok": "TikTok_%_30_days_Followers_Growth", 
                    "Instagram": "Instagram_%_30_days_Followers_Growth", 
                    "Facebook": "Facebook_%_30_days_Followers_Growth" 
                    }

    df = raw_df[["creator_name",
                "Content Genre", 
                map_platform_to_follower_columns[platform],
                map_platform_to_growth_columns[platform]]]    

    # Filter on topic
    df = df[df["Content Genre"] == map_topic_to_genre[topic]]

    # Sort by followers
    df = df.sort_values(by=map_platform_to_follower_columns[platform], ascending=False)

    # Format growth column as percentage
    df[map_platform_to_growth_columns[platform]] = df[map_platform_to_growth_columns[platform]].map("{:,.2%}".format)

    # Rename columns
    df = df.rename(columns={"creator_name": "Creator Name",
                            map_platform_to_follower_columns[platform]: "Followers",
                            map_platform_to_growth_columns[platform]: "Growth Last 30 days"})

    # Retrieve top 10
    df = df.head(10)

    # Add a column "Selected" , boolean, all false to begin with
    df["Selected"] = False


    # Display influencers
    st.markdown(f"## 🤝 Influencer Collaboration")
    st.markdown("These are the top influencers in this genre. Select the ones you'd like to collaborate with.")
    edited_df = st.data_editor(df)

    # Start next step
    st.button("➡️ Step 6: Content Production", on_click=move_to_step_6, kwargs={"df": edited_df})
    

if st.session_state["current_campaign_step"] == "Step 6: Content Production":

    generated_scripts = st.session_state["generated_scripts"]
    platform = st.session_state["selected_platform"]

    # Create videos
    if st.session_state["create_videos_activated"]:
        st.divider()
        with st.spinner(f"Creating videos..."):

            prompt_template = ChatPromptTemplate(
                messages=[
                    SystemMessage(
                        content="""
                        You're a helpful assistante. Your task is to convert a text file with markdown-formatted text to a CSV file. The text file contains a table with three columns: Scene, Visual, and Audio. Each row in the table represents a scene in a video script. Important: To do this only for Script 1, give me only the csv data, no other comments, and use '|' as the delimiter character.
                        """
                    ),
                    HumanMessage(content=f"The text is as follows:\n\n{generated_scripts}"),
                ]
            )

            model = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0,
            )

            chain = prompt_template | model | StrOutputParser()

            response = chain.invoke({})

            csv_text = convert_text_to_csv(response)
            transform_text_to_csv(csv_text)

            video_main(platform)  # TODO: improve how we use stuff from `video.py`
            # video_zoomout_main()  # TODO: improve how we use stuff from `video_zoomout.py`
            video_images_main(platform)  # TODO: improve how we use stuff from `video_zoomout.py`

        move_to_step_7()

if st.session_state["current_campaign_step"] == "Final step: Review and Launch":

    st.markdown(f"## ✍️ Suggested videos")

    col1, col2 = st.columns(2)

    col1.video("output_video.mp4")
    col1.caption("Option 1: find a publicly available video that matches the topic.")
    
    col2.video("output_video_images.mp4")
    col2.caption("Option 2: create a set of AI-generated images based on the topic and animate them into a video.")

    col3, col4 = st.columns(2)

    col3.video("output_video_heygen.mp4")
    col3.caption("Option 3: use the HeyGen API.")

    col4.video("output_video_lipsync.mp4")
    col4.caption("Option 4: use the SyncLabs API do perform lipsync on an existing video.")

    st.divider()

    st.markdown("Well done! We have concluded the campaign creation process. You can now review the campaign using the sidebar to the left. When you're ready, hit the Launch button to launch the campaign.")

