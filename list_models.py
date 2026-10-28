"""List the Gemini models your API key can use for text generation."""
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client()

print("Models that support text generation:\n")
for m in client.models.list():
    actions = m.supported_actions or []
    if "generateContent" in actions:
        print(f"- {m.name.removeprefix('models/')}  |  {m.display_name}")
