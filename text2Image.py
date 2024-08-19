import torch
from diffusers import FluxPipeline

# Load the model and specify GPU 0
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda:0")

# Optionally enable CPU offloading to save VRAM if needed
pipe.enable_model_cpu_offload()

prompt = "A cat holding a sign that says hello world"

# Ensure the random generator is also set to CPU, as it may not support bfloat16 on GPU
generator = torch.Generator(device="cpu").manual_seed(0)

# Generate the image
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    output_type="pil",
    num_inference_steps=50,
    max_sequence_length=512,
    generator=generator
).images[0]

# Save the image
image.save("flux-dev.png")
