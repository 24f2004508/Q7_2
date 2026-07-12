import json
import os

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

GROQ_API_KEY = os.environ["GROQ_API_KEY"]

SYSTEM_PROMPT = """
You solve arithmetic word problems.

Return ONLY valid JSON.

Rules:
- Output exactly two keys:
  - reasoning (string)
  - answer (integer)
- reasoning must be at least 80 characters.
- answer must be an integer.
- Ignore irrelevant numbers.
- No markdown.
- No extra keys.
"""

class SolverRequest(BaseModel):
    problem_id: str
    problem: str


@app.post("/solve")
def solve(req: SolverRequest):

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": req.problem,
            },
        ],
        "temperature": 0,
        "response_format": {
            "type": "json_object"
        },
    }

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
    )

    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=response.text,
        )

    result = response.json()

    try:
        output = json.loads(
            result["choices"][0]["message"]["content"]
        )
    except Exception:
        raise HTTPException(
            status_code=500,
            detail=result,
        )

    # Validate output
    if set(output.keys()) != {"reasoning", "answer"}:
        raise HTTPException(
            status_code=500,
            detail="Model returned incorrect keys.",
        )

    if not isinstance(output["answer"], int):
        raise HTTPException(
            status_code=500,
            detail="Answer is not an integer.",
        )

    if len(output["reasoning"]) < 80:
        raise HTTPException(
            status_code=500,
            detail="Reasoning too short.",
        )

    return output