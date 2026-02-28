import os
import time
import json
import tempfile
import subprocess
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a single Gemini client using the API key from .env
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


class AskRequest(BaseModel):
    video_url: str
    topic: str


class AskResponse(BaseModel):
    timestamp: str
    video_url: str
    topic: str


def download_audio(video_url: str, output_path: str) -> str:
    """Download audio-only from YouTube using yt-dlp."""
    command = [
        "yt-dlp",
        "--no-playlist",
        "-x",                        # Extract audio only
        "--audio-format", "mp3",     # Convert to mp3
        "--audio-quality", "0",      # Best quality
        "-o", output_path,
        video_url
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr}")
    return output_path


def upload_and_wait(file_path: str):
    """Upload file to Gemini Files API and wait until it's ACTIVE."""
    uploaded = client.files.upload(
        file=file_path,
        config=types.UploadFileConfig(mime_type="audio/mpeg")
    )

    # Poll until the file is ACTIVE (ready to use)
    max_wait = 120  # seconds
    waited = 0
    while uploaded.state.name != "ACTIVE":
        if waited > max_wait:
            raise RuntimeError("File upload timed out waiting for ACTIVE state")
        if uploaded.state.name == "FAILED":
            raise RuntimeError("File upload failed")
        time.sleep(5)
        waited += 5
        uploaded = client.files.get(name=uploaded.name)

    return uploaded


def find_timestamp_with_gemini(uploaded_file, topic: str) -> str:
    """Ask Gemini to find when the topic is spoken in the audio."""
    prompt = f"""Listen to this audio carefully. Find the FIRST moment where the topic or phrase "{topic}" is spoken or discussed.

Return ONLY a JSON object in this exact format with no other text:
{{"timestamp": "HH:MM:SS"}}

Rules:
- Use HH:MM:SS format ONLY (e.g. "00:05:47", "01:23:45")
- Always include hours (even if 0: use "00:")
- If the topic is not found, return your best guess for when related content starts
- Return ONLY the JSON, nothing else"""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[uploaded_file, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema={
                "type": "object",
                "properties": {
                    "timestamp": {
                        "type": "string",
                        "description": "Timestamp in HH:MM:SS format"
                    }
                },
                "required": ["timestamp"]
            }
        )
    )

    result = json.loads(response.text)
    timestamp = result.get("timestamp", "00:00:00")

    # Validate and fix format — must be HH:MM:SS
    parts = timestamp.split(":")
    if len(parts) == 2:
        # MM:SS -> add 00: prefix
        timestamp = f"00:{parts[0].zfill(2)}:{parts[1].zfill(2)}"
    elif len(parts) == 3:
        timestamp = f"{parts[0].zfill(2)}:{parts[1].zfill(2)}:{parts[2].zfill(2)}"
    else:
        timestamp = "00:00:00"

    return timestamp


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    audio_path = None
    uploaded_file = None

    try:
        # Step 1: Create a temp file path for the audio
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            audio_path = tmp.name

        # Step 2: Download audio from YouTube
        try:
            download_audio(request.video_url, audio_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Audio download error: {str(e)}")

        # Check the file actually exists and has content
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            mp3_path = audio_path.replace(".mp3", "") + ".mp3"
            if os.path.exists(mp3_path) and os.path.getsize(mp3_path) > 0:
                audio_path = mp3_path
            else:
                raise HTTPException(status_code=400, detail="Audio download produced empty file")

        # Step 3: Upload to Gemini Files API
        try:
            uploaded_file = upload_and_wait(audio_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Gemini upload error: {str(e)}")

        # Step 4: Ask Gemini to find the timestamp
        try:
            timestamp = find_timestamp_with_gemini(uploaded_file, request.topic)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Gemini processing error: {str(e)}")

        return AskResponse(
            timestamp=timestamp,
            video_url=request.video_url,
            topic=request.topic
        )

    finally:
        # Step 5: Cleanup — delete temp audio file
        if audio_path and os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except Exception:
                pass
        # Delete from Gemini Files too (saves quota)
        if uploaded_file:
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception:
                pass


@app.get("/")
def root():
    return {"status": "ok", "message": "YouTube Topic Timestamp Finder is running"}