from diffusers import FluxPipeline
from huggingface_hub import login
from metrics import calculate_clip_score, calculate_fid, calculate_lpips, calculate_psnr_resized, compute_image_reward
from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.utils import get_precision
import os
from quantization.preprocessing import preprocessing_coco
from resizing_image import resize_images
from shared.resources_monitor import generate_image_and_monitor, write_generation_metadata_to_file
import torch
from tqdm import tqdm
from PIL import Image
import json

# Replace 'YOUR_TOKEN' with your actual Hugging Face token
login(token="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")

coco_dir = "coco"
annotations_dir = os.path.join(coco_dir, "annotations")
val2017_dir = os.path.join(coco_dir, "val2017")

image_filename_to_caption, image_dimensions, image_id_to_dimensions = preprocessing_coco(annotations_dir)

# Create a directory to save generated images
generated_image_dir = "quantization/sd_outputs"
os.makedirs(generated_image_dir, exist_ok=True)
print(f"Created directory for generated images: {generated_image_dir}")

resized_generated_image_dir = "quantization/resized_sd_outputs"
os.makedirs(resized_generated_image_dir, exist_ok=True)
print(f"Created directory for resized generated images: {resized_generated_image_dir}")

# Limit the number of images to generate for demonstration purposes
num_images_to_generate = 500

try:
# Assuming 'pipeline' and 'image_filename_to_caption' are already defined from previous steps
# auto-detect your precision is 'int4' or 'fp4' based on your GPU
  precision = get_precision()
# "mit-han-lab/svdq-int4-flux.1-dev"
  transformer = NunchakuFluxTransformer2dModel.from_pretrained(f"mit-han-lab/svdq-{precision}-flux.1-dev", offload=True)
# transformer = NunchakuFluxTransformer2dModel.from_pretrained(
#     f"mit-han-lab/svdq-{precision}-flux.1-schnell"
# )
  pipeline = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=torch.bfloat16).to("cuda")

  # pipeline = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev",torch_dtype=torch.bfloat16).to("cuda")
  pipeline.enable_model_cpu_offload()
except Exception as e:
  print(f"Error when loading pipeline: {e}")
  pipeline = None # Set pipeline to None if loading fails

# Keep track of generation time
generation_times = {}
generation_metadata = []

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
        # Generate image and monitor VRAM. The generate_image_and_monitor function already saves the image and appends metadata
        generation_time, metadata = generate_image_and_monitor(pipeline, prompt, output_path, filename)
        # Add the generation time to the dictionary
        generation_times[filename] = generation_time
        print(f"Generated and saved {filename} (Prompt {i+1}/{num_images_to_generate}: {prompt[:50]}...)")
        generation_metadata.append(metadata)
    except Exception as e:
        print(f"Error when generating image for prompt '{prompt[:50]}...': {e}")
        generation_times[filename] = -1 # Indicate an error
print("Image generation complete.")

# Lưu metadata vào file JSON
GENERATION_METADATA_FILE = './quantization_metadata.json'
write_generation_metadata_to_file(GENERATION_METADATA_FILE)

# Optional: Print average generation time
successful_generations = [t for t in generation_times.values() if t > 0]
if successful_generations:
    average_time = sum(successful_generations) / len(successful_generations)
    print(f"Average generation time per image (for successful generations): {average_time:.2f} seconds")

resize_images(generated_image_dir, resized_generated_image_dir, image_dimensions)
print("Image resizing complete.")

# calculate_fid(generated_image_dir, resized_generated_image_dir, val2017_dir)
# print("FID calculation complete.")

# compute_image_reward(generated_image_dir, image_filename_to_caption)
# print("Image Reward calculation complete.")

# # Get the list of filenames for which resized images were generated successfully
# resized_generated_files = [f for f in os.listdir(resized_generated_image_dir) if os.path.isfile(os.path.join(resized_generated_image_dir, f))]

# # Calculate and print the average PSNR for resized images
# calculate_psnr_resized(val2017_dir, resized_generated_image_dir, resized_generated_files)
# print("PSNR calculation for resized images complete.")

# generated_files = [f for f in os.listdir(resized_generated_image_dir) if os.path.isfile(os.path.join(resized_generated_image_dir, f))]
# calculate_lpips(val2017_dir, resized_generated_image_dir, generated_files)
# print("LPIPS calculation complete.")

# calculate_clip_score(resized_generated_image_dir, image_filename_to_caption)
# print("CLIP score calculation complete.")

# 5. Calculate Image Quality Metrics (if not skipped)
metrics_results = {}
output_dir = "quantization/quantization_outputs"
os.makedirs(output_dir, exist_ok=True)

if False:  # Replace with args.skip_metrics if using argparse
    print("\n=== Skipping Image Quality Metrics (--skip_metrics flag set) ===")
else:
    print("\n=== Calculating Image Quality Metrics ===")

if not False:  # Replace with args.skip_metrics if using argparse
    # Create directory for resized images (needed for FID and PSNR)
    resized_output_dir = os.path.join(output_dir, "resized")
    os.makedirs(resized_output_dir, exist_ok=True)
    
    # Resize images for metrics calculation
    try:
        print("\n--- Resizing Images for Metrics Calculation ---")
        resize_images(output_dir, resized_output_dir, image_dimensions)
    except Exception as e:
        print(f"Error resizing images: {e}")
    
    # Calculate FID score
    try:
        print("\n--- Calculating FID Score ---")
        coco_val_dir = os.path.join(coco_dir, "val2017")
        fid_score = calculate_fid(output_dir, resized_output_dir, coco_val_dir)
        metrics_results["fid_score"] = fid_score
    except Exception as e:
        print(f"Error calculating FID: {e}")
    
    # Calculate CLIP Score
    try:
        print("\n--- Calculating CLIP Score ---")
        clip_score = calculate_clip_score(output_dir, image_filename_to_caption)
        metrics_results["clip_score"] = clip_score
    except Exception as e:
        print(f"Error calculating CLIP Score: {e}")
    
    # Calculate ImageReward
    try:
        print("\n--- Calculating ImageReward ---")
        image_reward = compute_image_reward(output_dir, image_filename_to_caption)
        metrics_results["image_reward"] = image_reward
    except Exception as e:
        print(f"Error calculating ImageReward: {e}")
    
    # Calculate LPIPS - we need original images to compare with generated images
    try:
        print("\n--- Calculating LPIPS ---")
        # For COCO dataset, we need to select a subset of filenames for LPIPS calculation
        # We should compare resized images to ensure dimensions match
        
        # Create a directory for resized original images
        resized_original_dir = os.path.join(output_dir, "resized_original")
        os.makedirs(resized_original_dir, exist_ok=True)
        
        # Get list of generated filenames that have been resized
        generated_filenames = [f for f in os.listdir(resized_output_dir) if os.path.isfile(os.path.join(resized_output_dir, f)) 
                            and f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        
        # Use the metrics_subset parameter to limit the number of images for metrics
        subset_size = min(5, len(generated_filenames))# TODO args.metrics_subset
        selected_filenames = generated_filenames[:subset_size]
        
        # Manually resize original COCO images to match generated images for LPIPS calculation
        original_dir = os.path.join(coco_dir, "val2017")
        for filename in tqdm(selected_filenames, desc="Resizing original images for LPIPS"):
            original_path = os.path.join(original_dir, filename)
            resized_original_path = os.path.join(resized_original_dir, filename)
            
            if os.path.exists(original_path):
                try:
                    # Get the size of the resized generated image for consistency
                    generated_img_path = os.path.join(resized_output_dir, filename)
                    generated_img = Image.open(generated_img_path)
                    target_size = generated_img.size
                    
                    # Resize the original image to match the generated image
                    original_img = Image.open(original_path).convert("RGB")
                    resized_original_img = original_img.resize(target_size, Image.LANCZOS)
                    resized_original_img.save(resized_original_path)
                except Exception as e:
                    print(f"Error resizing original image {filename}: {e}")
        
        # Now calculate LPIPS using the resized original and generated images
        lpips_score = calculate_lpips(resized_original_dir, resized_output_dir, selected_filenames)
        metrics_results["lpips"] = lpips_score
    except Exception as e:
        print(f"Error calculating LPIPS: {e}")
    
    # Calculate PSNR - we need original images to compare with generated images
    try:
        print("\n--- Calculating PSNR ---")
        # For COCO dataset, we use resized generated images
        original_dir = os.path.join(coco_dir, "val2017")
        generated_filenames = [f for f in os.listdir(resized_output_dir) if os.path.isfile(os.path.join(resized_output_dir, f))
                            and f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        
        # Use the metrics_subset parameter to limit the number of images for metrics
        subset_size = min(5, len(generated_filenames))# TODO: args.metrics_subset
        psnr_score = calculate_psnr_resized(original_dir, resized_output_dir, generated_filenames[:subset_size])
        metrics_results["psnr"] = psnr_score
    except Exception as e:
        print(f"Error calculating PSNR: {e}")

# Save metadata to JSON file
metadata_file = os.path.join(output_dir, "quantization_metadata.json")
try:
    # Add metrics to metadata
    combined_metadata = {
        "model_info": {
            "precision": precision
        },
        "generation_metadata": generation_metadata,
        #"metrics": metrics_results
    }
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(combined_metadata, f, ensure_ascii=False, indent=4)
    print(f"Saved metadata to: {metadata_file}")
except Exception as e:
    print(f"Error when saving metadata JSON: {e}")

# Calculate and print statistics
successful_generations = [t for t in generation_times.values() if t > 0]
if successful_generations:
    avg_time = sum(successful_generations) / len(successful_generations)
    print(f"\n=== Generation Statistics ===")
    print(f"Successfully generated {len(successful_generations)}/{len(image_filename_to_caption)} images")
    print(f"Average generation time: {avg_time:.2f} seconds per image")
    print(f"Total generation time: {sum(successful_generations):.2f} seconds")
    
    # Calculate average VRAM usage across all images
    all_vram_data = [meta.get("average_vram_gb") for meta in generation_metadata if meta.get("average_vram_gb") is not None]
    if all_vram_data:
        overall_avg_vram = sum(all_vram_data) / len(all_vram_data)
        print(f"Average VRAM usage across all {len(all_vram_data)} images: {overall_avg_vram:.2f} GB")

# Save summary report
with open(os.path.join(output_dir, "quant_pruning_summary.txt"), "w", encoding="utf-8") as f:
    f.write("=== Combined Quantization and Pruning Summary ===\n\n")
    f.write(f"Processed {len(image_filename_to_caption)} COCO captions\n")
    f.write(f"Inference steps: {args.steps}\n")
    f.write(f"Quantization: {precision}\n")
    if successful_generations:
        f.write(f"Successfully generated {len(successful_generations)}/{len(image_filename_to_caption)} images\n")
        f.write(f"Average generation time: {avg_time:.2f} seconds per image\n")
        f.write(f"Total generation time: {sum(successful_generations):.2f} seconds\n")
        # Add VRAM usage to summary
        all_vram_data = [meta.get("average_vram_gb") for meta in generation_metadata if meta.get("average_vram_gb") is not None]
        if all_vram_data:
            overall_avg_vram = sum(all_vram_data) / len(all_vram_data)
            f.write(f"Average VRAM usage across all {len(all_vram_data)} images: {overall_avg_vram:.2f} GB\n")
    
    # Add metrics results to summary
    if metrics_results:
        f.write("\n=== Image Quality Metrics ===\n")
        for metric_name, metric_value in metrics_results.items():
            if metric_value is not None:
                f.write(f"{metric_name.upper()}: {metric_value:.4f}\n")

print(f"Completed. Results saved to {output_dir}")