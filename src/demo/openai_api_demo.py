import json

import yaml
from openai import OpenAI

with open("../../config.json") as f:
    config = json.load(f)
    api_key = config["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

with open("prompt.yaml", "r") as file:
    prompt = yaml.safe_load(file)

keywords = ["安倍首相", "富士山", "馬"]
user_prompt = prompt["user"].format(keywords="\n".join(f"- {keyword.strip()}" for keyword in keywords))
completion = client.chat.completions.create(
    model="gpt-3.5-turbo", temperature=0.5, messages=[{"role": "system", "content": prompt["system"]}, {"role": "user", "content": user_prompt}]
)

print(completion.choices[0].message.content)
