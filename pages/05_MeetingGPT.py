import subprocess
import re
import math
import glob
import os
import streamlit as st
from pydub import AudioSegment
from openai import OpenAI
from langchain.chat_models.openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import StrOutputParser

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.1,
)


openai = OpenAI()

has_transcript = os.path.exists("./.cache/files/meeting_transcript.txt")


@st.cache_data()
def extract_audio_from_video(video_path):
    if has_transcript:
        return
    audio_path = re.sub(r"\.[^.]+$", ".mp3", video_path)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        audio_path,
    ]
    subprocess.run(command)


@st.cache_data()
def cut_audio_in_chunks(audio_path, chunk_minutes, chunks_folder):
    if has_transcript:
        return
    track = AudioSegment.from_mp3(audio_path)
    chunk_size = chunk_minutes * 60 * 1000
    chunk_count = math.ceil(len(track) / chunk_size)
    for i in range(chunk_count):
        start_time = i * chunk_size
        end_time = (i + 1) * chunk_size

        chunk = track[start_time:end_time]

        chunk.export(f"{chunks_folder}/chunk_{i}.mp3", format="mp3")


@st.cache_data()
def transcribe_chunks(chunks_folder, destination_folder):
    if has_transcript:
        return
    files = glob.glob(f"{chunks_folder}/*.mp3")
    files.sort()
    for file in files:
        with open(file, "rb") as audio_file, open(
            destination_folder, "a"
        ) as text_file:
            transcription = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
            )
            text_file.write(transcription.text)


st.set_page_config(
    page_title="MeetingGPT",
    page_icon="💼",
)

st.markdown(
    """
# MeetingGPT

Welcome to MeetingGPT, upload a video and I will give you a transcript,
a summary and a chat bot to ask any questions about it.

Get started by uploading a video file in the sidebar.
"""
)

with st.sidebar:
    video = st.file_uploader(
        "Video",
        type=["mp4", "mov", "avi"],
    )

if video:
    with st.status("Loading video...") as status:
        video_content = video.read()
        video_path = f"./.cache/files/meeting_videos/{video.name}"
        audio_path = re.sub(r"\.[^.]+$", ".mp3", video_path)
        audio_chunks_path = "./.cache/files/meeting_audio_chunks"
        transcript_path = "./.cache/files/meeting_transcript.txt"
        with open(video_path, "wb") as f:
            f.write(video_content)
        status.update(label="Extracting audio...")
        extract_audio_from_video(video_path)
        status.update(label="Cutting audio segments...")
        cut_audio_in_chunks(audio_path, 10, audio_chunks_path)
        status.update(label="Transcribing audio...")
        transcribe_chunks(
            audio_chunks_path,
            transcript_path,
        )

    transcript_tab, summary_tab, chat_tab = st.tabs(
        [
            "Transcript",
            "Summary",
            "Q&A",
        ]
    )

    with transcript_tab:
        with open(transcript_path, "r") as file:
            st.write(file.read())

    with summary_tab:
        start = st.button("Generate Summary")

        if start:
            loader = TextLoader(transcript_path)
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=800,
                chunk_overlap=100,
            )
            docs = loader.load_and_split(text_splitter=splitter)

            first_summary_prompt = ChatPromptTemplate.from_template(
                """
                Write a concise summary of the following:
                "{text}"
                CONCISE SUMMARY:
            """
            )

            first_summary_chain = (
                first_summary_prompt | llm | StrOutputParser()
            )

            summary = first_summary_chain.invoke(
                {
                    "text": docs[0].page_content,
                }
            )

            refine_prompt = ChatPromptTemplate.from_template(
                """
                Your job is to produce a final summary.
                We have provided an existing summary up to a certain point:
                {existing_summary}
                We have the opportunity to refine the existing summary
                (only if needed) with some more context below.
                ------------
                {context}
                ------------
                Given the new context, refine the original summary.
                If the context isn't useful, RETURN the original summary.
                """
            )

            refine_chain = refine_prompt | llm | StrOutputParser()

            with st.status("Summarizing...") as status:
                for idx, doc in enumerate(docs[1:]):
                    status.update(
                        label=f"Processing document {idx + 1}/{len(docs) - 1}"
                    )
                    summary = refine_chain.invoke(
                        {
                            "existing_summary": summary,
                            "context": doc.page_content,
                        }
                    )
                    st.write(summary)
            st.write(summary)
