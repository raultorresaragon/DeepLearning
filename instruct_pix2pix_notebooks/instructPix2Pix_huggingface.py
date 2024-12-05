import PIL
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers.utils import load_image

pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained("your_cool_model", torch_dtype=torch.float16).to("cuda")
generator = torch.Generator("cuda").manual_seed(0)

image = load_image("https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/test_pix2pix_4.png")
prompt = "add some ducks to the lake"
num_inference_steps = 20
image_guidance_scale = 1.5
guidance_scale = 10

edited_image = pipeline(
   prompt,
   image=image,
   num_inference_steps=num_inference_steps,
   image_guidance_scale=image_guidance_scale,
   guidance_scale=guidance_scale,
   generator=generator,
).images[0]
edited_image.save("edited_image.png")