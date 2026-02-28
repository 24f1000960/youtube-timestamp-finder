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


def fetch_captions_from_tracks(caption_tracks: list, headers: dict) -> list:
    """Helper to fetch and parse captions from track list."""
    if not caption_tracks:
        raise ValueError("No captions available")
    
    # Prefer English, fall back to first available
    caption_url = None
    for track in caption_tracks:
        lang = track.get('languageCode', '')
        if lang.startswith('en'):
            caption_url = track.get('baseUrl')
            break
    if not caption_url:
        caption_url = caption_tracks[0].get('baseUrl')
    
    if not caption_url:
        raise ValueError("Could not find caption URL")
    
    # Add format parameter to get JSON
    if '?' in caption_url:
        caption_url += '&fmt=json3'
    else:
        caption_url += '?fmt=json3'
    
    caption_response = requests.get(caption_url, headers=headers, timeout=10)
    caption_response.raise_for_status()
    
    # Parse JSON format
    try:
        caption_data = caption_response.json()
        transcript = []
        for event in caption_data.get('events', []):
            if 'segs' in event:
                start = event.get('tStartMs', 0) / 1000.0
                text = ''.join(seg.get('utf8', '') for seg in event['segs'])
                if text.strip():
                    transcript.append({"text": text.strip(), "start": start})
        if transcript:
            return transcript
    except json.JSONDecodeError:
        pass
    
    # Fallback: parse XML format
    transcript = []
    for m in re.finditer(r'<text start="([\d.]+)"[^>]*>([^<]*)</text>', caption_response.text):
        start = float(m.group(1))
        text = m.group(2)
        text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        text = text.replace('&#39;', "'").replace('&quot;', '"').replace('&nbsp;', ' ')
        if text.strip():
            transcript.append({"text": text.strip(), "start": start})
    
    if not transcript:
        raise ValueError("Could not parse captions")
    
    return transcript


def get_transcript_innertube_android(video_id: str) -> list:
    """
    Use YouTube's Innertube API with Android client.
    Android client is often less restricted than web.
    """
    innertube_url = "https://www.youtube.com/youtubei/v1/player"
    
    payload = {
        "context": {
            "client": {
                "hl": "en",
                "gl": "US",
                "clientName": "ANDROID",
                "clientVersion": "19.09.37",
                "androidSdkVersion": 30,
                "userAgent": "com.google.android.youtube/19.09.37 (Linux; U; Android 11) gzip"
            }
        },
        "videoId": video_id,
        "params": "CgIQBg=="  # Request captions
    }
    
    headers = {
        'Content-Type': 'application/json',
        'User-Agent': 'com.google.android.youtube/19.09.37 (Linux; U; Android 11) gzip',
        'X-Youtube-Client-Name': '3',
        'X-Youtube-Client-Version': '19.09.37',
    }
    
    response = requests.post(innertube_url, json=payload, headers=headers, timeout=15)
    response.raise_for_status()
    
    player_response = response.json()
    
    playability = player_response.get('playabilityStatus', {})
    if playability.get('status') == 'ERROR':
        raise ValueError(f"Video unavailable: {playability.get('reason', 'Unknown')}")
    
    captions = player_response.get('captions', {})
    caption_tracks = captions.get('playerCaptionsTracklistRenderer', {}).get('captionTracks', [])
    
    return fetch_captions_from_tracks(caption_tracks, headers)


def get_transcript_innertube_ios(video_id: str) -> list:
    """
    Use YouTube's Innertube API with iOS client.
    """
    innertube_url = "https://www.youtube.com/youtubei/v1/player"
    
    payload = {
        "context": {
            "client": {
                "hl": "en",
                "gl": "US",
                "clientName": "IOS",
                "clientVersion": "19.09.3",
                "deviceModel": "iPhone14,3",
                "userAgent": "com.google.ios.youtube/19.09.3 (iPhone14,3; U; CPU iOS 15_6 like Mac OS X)"
            }
        },
        "videoId": video_id,
        "params": "CgIQBg=="
    }
    
    headers = {
        'Content-Type': 'application/json',
        'User-Agent': 'com.google.ios.youtube/19.09.3 (iPhone14,3; U; CPU iOS 15_6 like Mac OS X)',
        'X-Youtube-Client-Name': '5',
        'X-Youtube-Client-Version': '19.09.3',
    }
    
    response = requests.post(innertube_url, json=payload, headers=headers, timeout=15)
    response.raise_for_status()
    
    player_response = response.json()
    
    playability = player_response.get('playabilityStatus', {})
    if playability.get('status') == 'ERROR':
        raise ValueError(f"Video unavailable: {playability.get('reason', 'Unknown')}")
    
    captions = player_response.get('captions', {})
    caption_tracks = captions.get('playerCaptionsTracklistRenderer', {}).get('captionTracks', [])
    
    return fetch_captions_from_tracks(caption_tracks, headers)


def get_transcript_innertube_tv(video_id: str) -> list:
    """
    Use YouTube's Innertube API with TV embedded client.
    """
    innertube_url = "https://www.youtube.com/youtubei/v1/player"
    
    payload = {
        "context": {
            "client": {
                "hl": "en",
                "gl": "US",
                "clientName": "TVHTML5_SIMPLY_EMBEDDED_PLAYER",
                "clientVersion": "2.0"
            }
        },
        "videoId": video_id
    }
    
    headers = {
        'Content-Type': 'application/json',
        'User-Agent': 'Mozilla/5.0 (SMART-TV; Linux; Tizen 5.0) AppleWebKit/537.36',
    }
    
    response = requests.post(innertube_url, json=payload, headers=headers, timeout=15)
    response.raise_for_status()
    
    player_response = response.json()
    
    playability = player_response.get('playabilityStatus', {})
    if playability.get('status') == 'ERROR':
        raise ValueError(f"Video unavailable: {playability.get('reason', 'Unknown')}")
    
    captions = player_response.get('captions', {})
    caption_tracks = captions.get('playerCaptionsTracklistRenderer', {}).get('captionTracks', [])
    
    return fetch_captions_from_tracks(caption_tracks, headers)


def get_transcript_third_party(video_id: str) -> list:
    """
    Use third-party transcript API as final fallback.
    Tries multiple free services.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json',
    }
    
    # Try kome.ai (free, no auth required for basic usage)
    try:
        url = f"https://kome.ai/api/tools/youtube-transcript?video_id={video_id}"
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            data = response.json()
            transcript = []
            # Parse kome.ai response format
            if isinstance(data, list):
                for item in data:
                    if 'text' in item and 'start' in item:
                        transcript.append({"text": item['text'], "start": float(item['start'])})
            elif 'transcript' in data:
                for item in data['transcript']:
                    if 'text' in item:
                        transcript.append({"text": item['text'], "start": float(item.get('start', 0))})
            if transcript:
                return transcript
    except Exception:
        pass
    
    # Try youtubetranscript.com
    try:
        url = f"https://youtubetranscript.com/?server_vid2={video_id}"
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            # Parse XML response
            transcript = []
            for m in re.finditer(r'<text start="([\d.]+)"[^>]*>([^<]*)</text>', response.text):
                start = float(m.group(1))
                text = m.group(2)
                text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
                text = text.replace('&#39;', "'").replace('&quot;', '"')
                if text.strip():
                    transcript.append({"text": text.strip(), "start": start})
            if transcript:
                return transcript
    except Exception:
        pass
    
    raise ValueError("Third-party transcript APIs failed")


def get_transcript(video_id: str) -> list:
    """
    Try multiple methods to fetch transcript, with fallbacks.
    """
    errors = []
    
    # Method 1: youtube-transcript-api
    try:
        ytt = YouTubeTranscriptApi()
        fetched = ytt.fetch(video_id)
        return [{"text": s.text, "start": s.start} for s in fetched]
    except Exception as e:
        errors.append(f"yt-api: {str(e)[:50]}")
    
    # Method 2: Android Innertube (often less restricted)
    try:
        return get_transcript_innertube_android(video_id)
    except Exception as e:
        errors.append(f"android: {str(e)[:50]}")
    
    # Method 3: iOS Innertube
    try:
        return get_transcript_innertube_ios(video_id)
    except Exception as e:
        errors.append(f"ios: {str(e)[:50]}")
    
    # Method 4: TV Innertube
    try:
        return get_transcript_innertube_tv(video_id)
    except Exception as e:
        errors.append(f"tv: {str(e)[:50]}")
    
    # Method 5: Third-party APIs
    try:
        return get_transcript_third_party(video_id)
    except Exception as e:
        errors.append(f"3rd-party: {str(e)[:50]}")
    
    # All methods failed
    raise ValueError(f"All transcript methods failed: {'; '.join(errors)}")


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