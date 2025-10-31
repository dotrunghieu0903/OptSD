import os
from PIL import Image

def process_flickr8k(flickr_images_dir, flickr_captions_path, limit=None):
    """
    Process Flickr8k dataset and return captions dictionary.
    
    Args:
        flickr_images_dir: Directory containing Flickr8k images
        flickr_captions_path: Path to the captions.txt file
        limit: Maximum number of captions to load (default: None - load all)
    
    Returns:
        dict: Dictionary mapping image filenames to their captions.
        dict: Dictionary mapping image filenames to their dimensions.
    """
    # Dictionary to store image dimensions
    image_dimensions = {}

    # Load captions from captions.txt
    # The format is likely "image_filename.jpg, caption text"
    captions = {}
    processed_count = 0
    if os.path.exists(flickr_captions_path):
        with open(flickr_captions_path, 'r') as f:
            # Skip header if it exists
            lines = f.readlines()
            if lines and ',' in lines[0]: # Simple check for header
                lines = lines[1:]
            
            # Sort lines to ensure deterministic order when taking first N images
            lines = sorted(lines)
            
            for line in lines:
                line = line.strip()
                if line and ',' in line:
                    parts = line.split(',', 1)
                    if len(parts) == 2:
                        filename = parts[0].strip()
                        caption = parts[1].strip()
                        if filename not in captions: # Store only the first caption for each image
                            captions[filename] = caption
                            processed_count += 1
                            
                            # Try to get image dimensions if the image file exists
                            img_path = os.path.join(flickr_images_dir, filename)
                            if os.path.exists(img_path):
                                try:
                                    with Image.open(img_path) as img:
                                        image_dimensions[filename] = img.size
                                except Exception as e:
                                    print(f"Error reading image dimensions for {filename}: {e}")
                            
                            # Stop when we reach the limit (if specified)
                            if limit is not None and processed_count >= limit:
                                break
                    else:
                        print(f"Skipping malformed line in captions.txt: {line}")
        
        if limit is not None:
            print(f"Loaded {len(captions)} captions from {flickr_captions_path} (limit: {limit})")
        else:
            print(f"Loaded {len(captions)} captions from {flickr_captions_path}")
    else:
        print(f"Error: Captions file not found at {flickr_captions_path}")
        captions = {} # Ensure captions dictionary is empty if file not found
    
    return captions, image_dimensions