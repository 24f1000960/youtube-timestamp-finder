import os
import json
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    video_url: str
    topic: str


class AskResponse(BaseModel):
    timestamp: str
    video_url: str
    topic: str


def extract_video_id(url: str) -> str:
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"youtu\.be\/([0-9A-Za-z_-]{11})",
        r"embed\/([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract video ID from URL: {url}")


def get_transcript(video_id: str) -> list:
    # New API (v1.0+): instantiate the class, then call .fetch()
    ytt = YouTubeTranscriptApi()
    fetched = ytt.fetch(video_id)
    # fetched is a FetchedTranscript object — iterate its snippets
    return [{"text": s.text, "start": s.start} for s in fetched.snippets]


def seconds_to_hhmmss(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def find_timestamp_with_llm(transcript: list, topic: str) -> str:
    token = os.environ.get("AIPIPE_TOKEN")
    if not token:
        raise RuntimeError("AIPIPE_TOKEN environment variable is not set")

    client = OpenAI(
        api_key=token,
        base_url="https://aipipe.org/openrouter/v1"
    )

    # Build timestamped transcript text
    transcript_lines = []
    for entry in transcript:
        ts = seconds_to_hhmmss(entry["start"])
        transcript_lines.append(f"[{ts}] {entry['text']}")

    transcript_text = "\n".join(transcript_lines)

    # Trim if too long — keep first 12000 chars
    if len(transcript_text) > 12000:
        transcript_text = transcript_text[:12000] + "\n... (truncated)"

    prompt = f"""Below is a timestamped transcript from a YouTube video.

Find the FIRST moment where this topic or phrase is spoken or discussed:
"{topic}"

TRANSCRIPT:
{transcript_text}

Respond ONLY with a valid JSON object:
{{"timestamp": "HH:MM:SS"}}

Rules:
- Return the EXACT timestamp from the transcript where the topic FIRST appears
- Format MUST be HH:MM:SS (e.g. "00:05:47", "01:23:45")
- Return ONLY the JSON, no explanation"""

    response = client.chat.completions.create(
        model="google/gemini-2.5-pro-preview-03-25",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )

    raw = response.choices[0].message.content.strip()

    try:
        result = json.loads(raw)
        timestamp = result.get("timestamp", "00:00:00")
    except Exception:
        match = re.search(r"\d{1,2}:\d{2}:\d{2}", raw)
        timestamp = match.group(0) if match else "00:00:00"

    # Normalize to HH:MM:SS
    parts = timestamp.split(":")
    if len(parts) == 2:
        timestamp = f"00:{parts[0].zfill(2)}:{parts[1].zfill(2)}"
    elif len(parts) == 3:
        timestamp = f"{parts[0].zfill(2)}:{parts[1].zfill(2)}:{parts[2].zfill(2)}"
    else:
        timestamp = "00:00:00"

    return timestamp


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    try:
        video_id = extract_video_id(request.video_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid video URL: {str(e)}")

    try:
        transcript = get_transcript(video_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Transcript error: {str(e)}")

    try:
        timestamp = find_timestamp_with_llm(transcript, request.topic)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

    return AskResponse(
        timestamp=timestamp,
        video_url=request.video_url,
        topic=request.topic
    )


@app.get("/")
def root():
    return {"status": "ok", "message": "YouTube Topic Timestamp Finder is running"}