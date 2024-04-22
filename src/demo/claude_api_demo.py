import json
import os

import anthropic

with open("../../config.json") as f:
    config = json.load(f)
    api_key = config["ANTHROPIC_API_KEY"]

os.environ["ANTHROPIC_API_KEY"] = api_key

client = anthropic.Anthropic()

message = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1000,
    temperature=0.0,
    system="",
    messages=[{"role": "user", "content": "現在の日本の首相は誰ですか？"}],
)

print(message.content[0].text)
