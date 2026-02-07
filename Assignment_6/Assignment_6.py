import torch
from diffusers import DiffusionPipeline

model_id = "QuantFunc/Nunchaku-Qwen-Image-2512"

pipe = DiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

pipe.enable_attention_slicing()

prompt = "A futuristic AI robot coding in VS Code, cinematic lighting"

image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]

image.save("output.png")
print("Image saved as output.png")
