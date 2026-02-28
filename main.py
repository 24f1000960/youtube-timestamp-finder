import os
import json
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
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


class AskRequest(BaseModel):
    video_url: str
    topic: str


class AskResponse(BaseModel):
    timestamp: str
    video_url: str
    topic: str


def find_timestamp(video_url: str, topic: str) -> str:
    token = os.environ.get("AIPIPE_TOKEN")
    if not token:
        raise RuntimeError("AIPIPE_TOKEN environment variable is not set")

    client = OpenAI(
        api_key=token,
        base_url="https://aipipe.org/openrouter/v1"
    )

    prompt = f"""You are analyzing a YouTube video at this URL: {video_url}

Find the FIRST moment in the video where the topic or phrase "{topic}" is spoken or discussed.

Respond ONLY with a valid JSON object like this:
{{"timestamp": "00:05:47"}}

Rules:
- Format MUST be HH:MM:SS (always include hours, e.g. "00:05:47")
- Return ONLY the JSON object, no explanation, no markdown"""

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
        timestamp = find_timestamp(request.video_url, request.topic)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    return AskResponse(
        timestamp=timestamp,
        video_url=request.video_url,
        topic=request.topic
    )


@app.get("/")
def root():
    return {"status": "ok", "message": "YouTube Topic Timestamp Finder is running"}