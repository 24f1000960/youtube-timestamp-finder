import os
import time
import json
import tempfile
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import yt_dlp
from google import genai
from google.genai import types
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# Load .env file for local dev; on Render, env vars come from the dashboard
load_dotenv()

# ---------- App setup ----------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set!")

client = genai.Client(api_key=GEMINI_API_KEY)

# ---------- Request / Response models ----------
class AskRequest(BaseModel):
    video_url: str
    topic: str

class AskResponse(BaseModel):
    timestamp: str
    video_url: str
    topic: str

# ---------- Helper: Download audio only ----------
def download_audio(video_url: str, output_path: str) -> str:
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "128",
        }],
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    mp3_path = output_path + ".mp3"
    if not os.path.exists(mp3_path):
        raise FileNotFoundError(f"Audio file not found after download: {mp3_path}")
    return mp3_path

# ---------- Helper: Upload to Gemini and wait until ACTIVE ----------
def upload_and_wait(audio_path: str):
    print(f"Uploading {audio_path} to Gemini Files API...")
    uploaded_file = client.files.upload(
        file=audio_path,
        config=types.UploadFileConfig(mime_type="audio/mpeg")
    )

    max_wait = 120
    waited = 0
    while uploaded_file.state.name != "ACTIVE":
        if waited > max_wait:
            raise TimeoutError("Gemini file upload took too long to become ACTIVE.")
        print(f"  File state: {uploaded_file.state.name} — waiting...")
        time.sleep(5)
        waited += 5
        uploaded_file = client.files.get(name=uploaded_file.name)

    print("  File is ACTIVE!")
    return uploaded_file

# ---------- Helper: Normalize timestamp to HH:MM:SS ----------
def normalize_timestamp(ts: str) -> str:
    ts = ts.strip()
    parts = ts.split(":")
    try:
        if len(parts) == 1:
            total_sec = int(parts[0])
            h = total_sec // 3600
            m = (total_sec % 3600) // 60
            s = total_sec % 60
        elif len(parts) == 2:
            m, s = int(parts[0]), int(parts[1])
            h = 0
        else:
            h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
        return f"{h:02d}:{m:02d}:{s:02d}"
    except Exception:
        return "00:00:00"

# ---------- Main endpoint ----------
@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    audio_path = None
    gemini_file = None

    try:
        # Step 1: Download audio to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix="") as tmp:
            tmp_base = tmp.name

        print(f"Downloading audio for: {request.video_url}")
        audio_path = download_audio(request.video_url, tmp_base)
        print(f"Audio saved to: {audio_path}")

        # Step 2: Upload to Gemini Files API
        gemini_file = upload_and_wait(audio_path)

        # Step 3: Ask Gemini to find the timestamp
        prompt = f"""
You are analyzing an audio file. Find the FIRST moment when this topic is spoken or discussed.

Topic to find: "{request.topic}"

Return ONLY the timestamp in HH:MM:SS format (e.g. "00:05:47") of when this topic FIRST appears.
- Always use 3 parts separated by colons: HH:MM:SS
- 5 minutes 47 seconds = "00:05:47"
- 1 hour 34 minutes 9 seconds = "01:34:09"
"""

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                types.Part.from_uri(file_uri=gemini_file.uri, mime_type="audio/mpeg"),
                prompt
            ],
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
        timestamp = normalize_timestamp(result.get("timestamp", "00:00:00"))

        return AskResponse(
            timestamp=timestamp,
            video_url=request.video_url,
            topic=request.topic
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Step 4: Cleanup temp files
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
            print(f"Cleaned up local file: {audio_path}")
        if gemini_file:
            try:
                client.files.delete(name=gemini_file.name)
                print(f"Deleted Gemini file: {gemini_file.name}")
            except Exception:
                pass

# ---------- Health check ----------
@app.get("/")
def root():
    return {"status": "running", "endpoint": "POST /ask"}