import json
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load BLIP 
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

# File paths
input_file = "input.json"       # Replace with your input json
output_file = "output.json"     # Replace with your desired output filename

# Load input data
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Generate descriptions
for item in data:
    try:
        image = Image.open(requests.get(item["image_url"], stream=True).raw).convert("RGB")

        inputs = processor(image, return_tensors="pt").to("cuda")
        output = model.generate(**inputs, min_length=50)
        description = processor.decode(output[0], skip_special_tokens=True)

        item["description"] = description
    except Exception as e:
        item["description"] = f"Error: {str(e)}"

# Save output data
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Finished generating descriptions for: {input_file}")

