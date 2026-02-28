import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
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

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


class AskRequest(BaseModel):
    video_url: str
    topic: str


class AskResponse(BaseModel):
    timestamp: str
    video_url: str
    topic: str


def find_timestamp_with_gemini(video_url: str, topic: str) -> str:
    """Pass the YouTube URL directly to Gemini and ask it to find the timestamp."""
    prompt = f"""Watch this YouTube video and find the FIRST moment where the topic or phrase "{topic}" is spoken or discussed.

Return ONLY a JSON object in this exact format:
{{"timestamp": "HH:MM:SS"}}

Rules:
- HH:MM:SS format ONLY (e.g. "00:05:47", "01:23:45")
- Always include hours (even if zero: "00:")
- Return ONLY the JSON, nothing else"""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=types.Content(
            parts=[
                types.Part(
                    file_data=types.FileData(
                        mime_type="video/*",
                        file_uri=video_url
                    )
                ),
                types.Part(text=prompt)
            ]
        ),
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
        timestamp = find_timestamp_with_gemini(request.video_url, request.topic)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini error: {str(e)}")

    return AskResponse(
        timestamp=timestamp,
        video_url=request.video_url,
        topic=request.topic
    )


@app.get("/")
def root():
    return {"status": "ok", "message": "YouTube Topic Timestamp Finder is running"}