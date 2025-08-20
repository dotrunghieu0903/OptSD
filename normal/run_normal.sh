#!/bin/bash

# Run normal (non-optimized) image generation process
echo "Running normal image generation (without optimization)"

# Make script executable
chmod +x normal_coco.py

# Run directly from the normal directory
python normal_coco.py --num_images 100 --steps 25 --guidance_scale 7.5

# Or you can run through the app with optimization disabled
# python app.py --num_images 100 --steps 25 --guidance_scale 7.5
