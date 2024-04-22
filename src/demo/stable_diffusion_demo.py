# make sure you're logged in with `huggingface-cli login`
import torch
from diffusers import StableDiffusionPipeline
from torch import autocast

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=True)

prompt = "a photo of an astronaut riding a horse on mars"
pipe = pipe.to("mps")

pipe.enable_attention_slicing()

prompt = "a dog"
n_prompt = "bad fingers"

_ = pipe(prompt, negative_prompt=n_prompt, num_inference_steps=1)

image = pipe(prompt, negative_prompt=n_prompt).images[0]

image.save("../../public/img/astronaut_rides_horse.png")
