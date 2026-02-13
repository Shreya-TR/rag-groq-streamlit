import base64
import os

from config import VISION_MODEL

try:
    from groq import Groq
except ImportError:
    Groq = None


def image_to_caption(path: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or Groq is None:
        return ""

    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    client = Groq(api_key=api_key)
    prompt = (
        "Describe this image for retrieval in a RAG system. "
        "Extract key entities, numbers, chart trends, labels, and relationships."
    )
    resp = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }
        ],
        temperature=0.1,
    )
    return (resp.choices[0].message.content or "").strip()
