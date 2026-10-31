"""Optional local multimodal figure analysis through Ollama's vision API."""
import base64
import json
import mimetypes
import os
import urllib.request
from pathlib import Path
from typing import Sequence


def analyze_image_with_ollama(image_path: str, prompt: str, model: str) -> str:
    return analyze_images_with_ollama([image_path], prompt, model)


def analyze_images_with_ollama(image_paths: Sequence[str], prompt: str, model: str) -> str:
    images = [
        base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
        for image_path in image_paths
    ]
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": images}],
        "stream": False,
    }).encode("utf-8")
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.loads(response.read().decode("utf-8"))
    return result["message"]["content"].strip()


def analyze_images_with_openai(image_paths: Sequence[str], prompt: str,
                               model: str) -> str:
    """Send local images and a prompt to the OpenAI Chat Completions API."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY in the environment to use OpenAI vision")

    content = [{"type": "text", "text": prompt}]
    for image_path in image_paths:
        path = Path(image_path)
        media_type = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded_image = base64.b64encode(path.read_bytes()).decode("ascii")
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{media_type};base64,{encoded_image}",
                "detail": "high",
            },
        })

    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 512,
        "temperature": 0,
    }).encode("utf-8")
    request = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.loads(response.read().decode("utf-8"))
    message = result["choices"][0]["message"]["content"]
    if isinstance(message, list):
        message = " ".join(
            part.get("text", "") for part in message
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(message).strip()