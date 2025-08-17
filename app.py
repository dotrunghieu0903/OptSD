from cleanfid import fid
from huggingface_hub import notebook_login, login
import ImageReward as RM
import json
import lpips
import matplotlib.pyplot as plt
import numpy as np
from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.utils import get_precision
import os
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
import time
import torch
import torchvision.transforms as transforms
from tqdm import tqdm


# Replace 'YOUR_TOKEN' with your actual Hugging Face token
login(token="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")

coco_dir = "coco"
annotations_dir = os.path.join(coco_dir, "annotations")
val2017_dir = os.path.join(coco_dir, "val2017")

# Path to the captions annotation file
captions_file = os.path.join(annotations_dir, 'captions_val2017.json')

# Load the captions data
with open(captions_file, 'r') as f:
    captions_data = json.load(f)

# Xây dựng dictionary ánh xạ image_id đến kích thước
image_id_to_dimensions = {img['id']: (img['width'], img['height'], img['file_name'])
                          for img in captions_data['images']}

print(f"Đọc {len(captions_data['annotations'])} captions từ file chú thích COCO...")
processed_image_ids = set() # Để đảm bảo mỗi ảnh gốc chỉ được xử lý một lần cho mục đích prompt chính

image_filename_to_caption = {}
image_dimensions = {} # Lưu trữ {filename: (width, height)}
# Create a dictionary to store captions by image ID
for annotation in captions_data['annotations']:
    image_id = annotation['image_id']
    caption = annotation['caption']

    if image_id in image_id_to_dimensions and image_id not in processed_image_ids:
      width, height, original_filename = image_id_to_dimensions[image_id]
      image_filename_to_caption[original_filename] = caption
      image_dimensions[original_filename] = (width, height)
      processed_image_ids.add(image_id)

print(f"Extracted captions for {len(processed_image_ids)} images.")
print(f"Created mapping for {len(image_filename_to_caption)} images.")
print(f"Extracted dimensions for {len(image_dimensions)} images from annotations.")

# Create a directory to save generated images
generated_image_dir = "/stable_diffusion_outputs"
# os.makedirs(generated_image_dir, exist_ok=True)
print(f"Created directory for generated images: {generated_image_dir}")

resized_generated_image_dir = "/resized_stable_diffusion_outputs"
# os.makedirs(resized_generated_image_dir, exist_ok=True)
print(f"Created directory for resized generated images: {resized_generated_image_dir}")

import time
import threading
from pynvml import *

from diffusers import DiffusionPipeline, FluxPipeline
import torch

from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.utils import get_precision

# A list to store VRAM usage samples
vram_samples = []
stop_monitoring = threading.Event()

def monitor_vram(device_index=0):
    """
    Monitors VRAM usage of a specified GPU at a regular interval.
    """
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(device_index)

    while not stop_monitoring.is_set():
        try:
            info = nvmlDeviceGetMemoryInfo(handle)
            # Convert bytes to GB
            used_vram_gb = info.used / 1024**3
            vram_samples.append(used_vram_gb)
            time.sleep(0.1) # Sample every 100 milliseconds
        except NVMLError as error:
            print(f"Error during VRAM monitoring: {error}")
            break
    nvmlShutdown()

def generate_image_and_monitor(pipeline, prompt, output_path, filename):
    """
    Generates an image using the provided pipeline while monitoring VRAM.
    Returns generation time and metadata.
    """
    # Reset VRAM samples and stop event for this run
    global vram_samples
    vram_samples = []
    stop_monitoring.clear()

    # Start the monitoring thread
    monitor_thread = threading.Thread(target=monitor_vram)
    monitor_thread.start()

    # Run the image generation function
    result = None
    generation_time = -1 # Initialize with a value indicating failure
    metadata = {}

    try:
        start_time = time.time()
        # Assuming the pipeline expects prompt and returns an image object (e.g., PIL Image)
        generated_image = pipeline(prompt).images[0] # Assuming the pipeline returns a list of images
        end_time = time.time()
        generation_time = end_time - start_time

        # Save the generated image
        generated_image.save(output_path)

        # Create metadata
        metadata = {
            "generated_image_path": output_path,
            "original_flickr_filename": filename, # Store original filename for linking
            "caption_used": prompt,
            "generation_time": generation_time
        }

        print(f"Image generation completed in {generation_time:.2f} seconds.")

    except Exception as e:
        print(f"An error occurred during image generation: {e}")
        generation_time = -1 # Indicate failure
    finally:
        # Stop the monitoring thread
        stop_monitoring.set()
        monitor_thread.join()

    # Analyze the collected VRAM data (optional, can be done later with collected metadata)
    if vram_samples:
        avg_vram = sum(vram_samples) / len(vram_samples)
        peak_vram = max(vram_samples)

        print(f"\n--- VRAM Usage Statistics ---")
        print(f"Average VRAM used: {avg_vram:.2f} GB")
        print(f"Peak VRAM used: {peak_vram:.2f} GB")
        print(f"Number of samples: {len(vram_samples)}")
        metadata["average_vram_gb"] = avg_vram
        metadata["peak_vram_gb"] = peak_vram

    else:
        print("No VRAM data was collected.")
        metadata["average_vram_gb"] = None
        metadata["peak_vram_gb"] = None

    return generation_time, metadata # Return both generation time and metadata



try:
# Assuming 'pipeline' and 'image_filename_to_caption' are already defined from previous steps
  precision = get_precision()  # auto-detect your precision is 'int4' or 'fp4' based on your GPU
# "mit-han-lab/svdq-int4-flux.1-dev"
  transformer = NunchakuFluxTransformer2dModel.from_pretrained(
    "mit-han-lab/svdq-int4-flux.1-dev")
# transformer = NunchakuFluxTransformer2dModel.from_pretrained(
#     f"mit-han-lab/svdq-{precision}-flux.1-schnell"
# )
  pipeline = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=torch.bfloat16).to("cuda")

  # pipeline = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev",torch_dtype=torch.bfloat16).to("cuda")
  pipeline.enable_model_cpu_offload()
except Exception as e:
  print(f"Error loading pipeline: {e}")
  pipeline = None # Set pipeline to None if loading fails

# Limit the number of images to generate for demonstration purposes
num_images_to_generate = 1 # You can increase this for more comprehensive evaluation

# Keep track of generation time
generation_times = {}
generation_metadata = [] # Danh sách để lưu trữ thông tin metadata cho mỗi ảnh

print(f"Start to generate {min(num_images_to_generate, len(image_filename_to_caption))} images...")

# Create a reverse mapping from filename to image_id
image_filename_to_id = {v[2]: k for k, v in image_id_to_dimensions.items()}

for i, (filename, prompt) in enumerate(tqdm(list(image_filename_to_caption.items())[:num_images_to_generate], desc="Generating images")):
    output_path = os.path.join(generated_image_dir, filename)
    # Skip if the image already exists
    if os.path.exists(output_path):
        print(f"Skipping generation for {filename} (already exists)")
        continue
    try:
        # Generate image and monitor VRAM
        # The generate_image_and_monitor function already saves the image and appends metadata
        generation_time, metadata = generate_image_and_monitor(pipeline, prompt, output_path, filename)
        # Add the generation time to the dictionary
        generation_times[filename] = generation_time
        # The metadata is already appended inside generate_image_and_monitor
        # No need to append metadata again here.
        print(f"Generated and saved {filename} (Prompt {i+1}/{num_images_to_generate}: {prompt[:50]}...)")
    except Exception as e:
        print(f"Error when generating image for prompt '{prompt[:50]}...': {e}")
        generation_times[filename] = -1 # Indicate an error
print("Image generation complete.")