"""Optional local multimodal figure analysis through Ollama's vision API."""
import base64
import json
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