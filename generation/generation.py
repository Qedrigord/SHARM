import os
import json
import torch
import requests
from PIL import Image
from diffusers import AutoPipelineForImage2Image

# Configuration
access_token = "YOUR_ACCESS_TOKEN"  # Replace with actual token 
procedure = "image_guided"          # Options: image_guided, description_guided, or multimodally_guided
input_path = "input.json"           # Replace with your input json
output_folder = "output"            # Replace with the name of your desired output folder
os.makedirs(output_folder, exist_ok=True)

# Load data from JSON file
with open(input_path, "r") as f:
    data = json.load(f)

# Initialize model
pipeline = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)
pipeline = pipeline.to("cuda")

# Process each entry
for i, entry in enumerate(data):
    image_url = entry["image_url"]
    
    if procedure == "image_guided":
        prompt = entry["title"]
        weight = 0.1
    elif procedure == "description_guided":
        prompt = entry["description"]
        weight = 1
    elif procedure == "multimodally_guided":
        prompt = entry["description"]
        weight = 0.5

    try:
        raw_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

        # Generate image
        result_image = pipeline(prompt=prompt, image=raw_image, strength=weight).images[0]

        # Save result
        output_path = os.path.join(output_folder, f"output_{i}.png")
        result_image.save(output_path)
        torch.cuda.empty_cache()
        print(f"Saved: {output_path}")

    except Exception as e:
        print(f"Error processing image {i}: {e}")
