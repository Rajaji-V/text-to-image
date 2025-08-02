from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
prompt = input()

image = pipe(prompt).images[0]
image.show()
image.save("generated_image.png")
print("Image saved as generated_image.png")

