import json
import os

import clip
import torch
import yaml
from diffusers import StableDiffusionPipeline
from openai import OpenAI
from PIL import Image


class Pipeline:
    def __init__(self):
        self.setup_device()
        self.sentence_generator = SentenceGenerator()
        self.stable_diffusion = StableDiffusion(self.device)
        self.clip = CLIP(self.device)

    def setup_device(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        self.device = device

    def __call__(self, keywords, save_dir="../../public/img"):
        sentence = self.sentence_generator(keywords)
        img_name = sentence.replace(" ", "_") + ".png"
        save_path = os.path.join(save_dir, img_name)
        save_path = self.stable_diffusion(sentence, save_path)
        result = self.clip(save_path, keywords)
        return result


class SentenceGenerator:
    def __init__(self):
        with open("../../config.json") as f:
            config = json.load(f)
            api_key = config["OPENAI_API_KEY"]
        self.client = OpenAI(api_key=api_key)

        with open("../../public/prompts/en/create_noun_phrase.yaml") as f:
            self.prompt = yaml.safe_load(f)

    def __call__(self, keywords):
        print("Creating sentence with keywords:", keywords)
        user_prompt = self.prompt["user"].format(keywords="\n".join(f"- {keyword.strip()}" for keyword in keywords))
        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0.5,
            messages=[{"role": "system", "content": self.prompt["system"]}, {"role": "user", "content": user_prompt}],
        )
        sentence = completion.choices[0].message.content.replace("。", "").replace(".", "")
        print("Generated sentence:", sentence)
        return sentence


class StableDiffusion:
    def __init__(self, device, model="CompVis/stable-diffusion-v1-4"):
        self.pipe = StableDiffusionPipeline.from_pretrained(model, use_auth_token=True)
        self.pipe = self.pipe.to(device)
        self.pipe.enable_attention_slicing()

    def __call__(self, prompt, save_path):
        print("Creating image with prompt:", prompt)
        _ = self.pipe(prompt, num_inference_steps=1)
        image = self.pipe(prompt).images[0]
        if save_path:
            image.save(save_path)
            print("Image saved at:", save_path)
        return save_path


class CLIP:
    def __init__(self, device, model="ViT-B/32"):
        model, preprocess = clip.load(model, device=device)
        self.device = device
        self.model = model
        self.preprocess = preprocess

    def __call__(self, img_path, text):
        print("Analyzing image:", img_path)
        print("With text prompts:", text)
        image = self.preprocess(Image.open(img_path)).unsqueeze(0).to(self.device)
        text = clip.tokenize(text).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)
            logits_per_image, logits_per_text = self.model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        print("Label probs:", probs)
        return probs


if __name__ == "__main__":
    pipe = Pipeline()
    keywords = ["rabbit", "moon", "football"]
    result = pipe(keywords)
    print(result)
