import subprocess
import re
import math
import glob
import os
import streamlit as st
from pydub import AudioSegment
from openai import OpenAI


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
    with st.status("Loading video..."):
        video_content = video.read()
        video_path = f"./.cache/files/meeting_videos/{video.name}"
        audio_path = re.sub(r"\.[^.]+$", ".mp3", video_path)
        with open(video_path, "wb") as f:
            f.write(video_content)
    with st.status("Extracting audio..."):
        extract_audio_from_video(video_path)
    with st.status("Cutting audio segments..."):
        cut_audio_in_chunks(
            audio_path, 10, "./.cache/files/meeting_audio_chunks"
        )
    with st.status("Transcribing audio"):
        transcribe_chunks(
            "./.cache/files/meeting_audio_chunks",
            "./.cache/files/meeting_transcript.txt",
        )
