import os
import json
import re
import requests
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


def get_transcript_fallback(video_id: str) -> list:
    """
    Fallback: fetch transcript by parsing YouTube's player response directly.
    This uses different endpoints that may bypass IP blocks.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    }
    
    # Fetch the video page
    url = f"https://www.youtube.com/watch?v={video_id}"
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    
    # Extract ytInitialPlayerResponse from the page
    match = re.search(r'ytInitialPlayerResponse\s*=\s*({.+?});', response.text)
    if not match:
        raise ValueError("Could not find player response in page")
    
    player_response = json.loads(match.group(1))
    
    # Get captions track URL
    captions = player_response.get('captions', {})
    caption_tracks = captions.get('playerCaptionsTracklistRenderer', {}).get('captionTracks', [])
    
    if not caption_tracks:
        raise ValueError("No captions available for this video")
    
    # Prefer English, fall back to first available
    caption_url = None
    for track in caption_tracks:
        if 'en' in track.get('languageCode', ''):
            caption_url = track.get('baseUrl')
            break
    if not caption_url:
        caption_url = caption_tracks[0].get('baseUrl')
    
    if not caption_url:
        raise ValueError("Could not find caption URL")
    
    # Fetch the captions (returns XML)
    caption_response = requests.get(caption_url, headers=headers, timeout=10)
    caption_response.raise_for_status()
    
    # Parse XML captions
    transcript = []
    for match in re.finditer(r'<text start="([\d.]+)"[^>]*>([^<]+)</text>', caption_response.text):
        start = float(match.group(1))
        text = match.group(2)
        # Decode HTML entities
        text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        text = text.replace('&#39;', "'").replace('&quot;', '"')
        transcript.append({"text": text, "start": start})
    
    if not transcript:
        raise ValueError("Could not parse captions")
    
    return transcript


def get_transcript(video_id: str) -> list:
    """
    Try youtube-transcript-api first, fall back to direct HTTP parsing.
    """
    # Method 1: youtube-transcript-api (faster, cleaner)
    try:
        ytt = YouTubeTranscriptApi()
        fetched = ytt.fetch(video_id)
        return [{"text": s.text, "start": s.start} for s in fetched]
    except Exception as e:
        error_msg = str(e).lower()
        # If it's an IP block or bot detection, try fallback
        if "blocking" in error_msg or "bot" in error_msg or "ip" in error_msg:
            pass  # Continue to fallback
        else:
            raise  # Re-raise other errors
    
    # Method 2: Direct HTTP request fallback
    return get_transcript_fallback(video_id)


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