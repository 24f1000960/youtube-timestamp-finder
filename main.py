import os
import re
import json
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AI_API_TOKEN = os.getenv("AI_API_TOKEN")
CHAT_URL = os.getenv("CHAT_URL")


class AskRequest(BaseModel):
    video_url: str
    topic: str


class AskResponse(BaseModel):
    timestamp: str
    video_url: str
    topic: str


def extract_video_id(url: str) -> str:
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    if not match:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    return match.group(1)


def seconds_to_hhmmss(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def fix_timestamp_format(timestamp: str) -> str:
    if not timestamp:
        return "00:00:00"
    parts = timestamp.strip().split(":")
    if len(parts) == 2:
        return "00:" + timestamp.strip()
    elif len(parts) == 3:
        return timestamp.strip()
    return "00:00:00"


def get_transcript(video_id: str) -> str:
    fetched = YouTubeTranscriptApi().fetch(video_id)
    transcript_list = fetched.to_raw_data()

    formatted = ""
    for item in transcript_list:
        time_str = seconds_to_hhmmss(item["start"])
        text = item["text"].replace("\n", " ")
        formatted += f"[{time_str}] {text}\n"
    return formatted


def ask_gemini(transcript: str, topic: str) -> str:
    user_prompt = (
        "You are a precise timestamp finder. You will be given a video transcript "
        "with timestamps in HH:MM:SS format.\n\n"
        f'Find the FIRST moment where the speaker discusses: "{topic}"\n\n'
        "Rules:\n"
        "1. Look for the exact phrase OR its meaning/context\n"
        "2. Return the timestamp of that EXACT moment\n"
        "3. Do NOT return chapter starts unless the topic starts there\n"
        "4. Timestamp MUST be in HH:MM:SS format\n"
        "5. Return ONLY raw JSON, no markdown, no explanation\n\n"
        f"Transcript:\n{transcript}\n\n"
        'Return ONLY this JSON: {"timestamp": "HH:MM:SS"}'
    )

    headers = {
        "Authorization": f"Bearer {AI_API_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "google/gemini-2.0-flash-001",
        "messages": [{"role": "user", "content": user_prompt}],
        "temperature": 0.0,
    }

    response = requests.post(CHAT_URL, headers=headers, json=payload, timeout=60)

    if response.status_code != 200:
        raise HTTPException(
            status_code=500,
            detail=f"AI API error: {response.text}"
        )

    raw = response.json()["choices"][0]["message"]["content"]

    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        raw = "\n".join(lines).strip()

    try:
        result = json.loads(raw)
        return result.get("timestamp", "00:00:00")
    except json.JSONDecodeError:
        match = re.search(r"\d{2}:\d{2}:\d{2}", raw)
        if match:
            return match.group(0)
        return "00:00:00"


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    video_id = extract_video_id(request.video_url)

    try:
        transcript = get_transcript(video_id)
    except TranscriptsDisabled:
        raise HTTPException(
            status_code=400,
            detail="Transcripts are disabled for this video."
        )
    except NoTranscriptFound:
        raise HTTPException(
            status_code=400,
            detail="No transcript found for this video."
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Transcript error: {str(e)}"
        )

    timestamp = ask_gemini(transcript, request.topic)
    timestamp = fix_timestamp_format(timestamp)

    return AskResponse(
        timestamp=timestamp,
        video_url=request.video_url,
        topic=request.topic,
    )