import os
import json

def preprocessing_coco(annotations_dir):
    # Path to the captions annotation file
    captions_file = os.path.join(annotations_dir, 'captions_val2017.json')

    # Load the captions data
    with open(captions_file, 'r') as f:
        captions_data = json.load(f)

    # Build dictionary mapping image_id to size
    image_id_to_dimensions = {img['id']: (img['width'], img['height'], img['file_name'])
                            for img in captions_data['images']}

    print(f"Read {len(captions_data['annotations'])} captions from COCO annotation file...")
    # To ensure each original image is processed only once for the main prompt purpose
    processed_image_ids = set()

    image_filename_to_caption = {}
    # Store {filename: (width, height)}
    image_dimensions = {}
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
    return image_filename_to_caption, image_dimensions, image_id_to_dimensions