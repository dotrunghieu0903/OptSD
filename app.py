import json
import argparse
import os
import sys
from argparse import Namespace

# Define a main function that loads configuration from config.json
def load_config():
    config_path = 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def main():
    # Parse command-line arguments for optional overrides
    parser = argparse.ArgumentParser(description="Apply optimization techniques to diffusion models")
    parser.add_argument("--num_images", type=int, help="Number of images to process")
    parser.add_argument("--inference_steps", type=int, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, help="Guidance scale for image generation")
    parser.add_argument("--skip_metrics", action="store_true", help="Skip calculation of image quality metrics")
    parser.add_argument("--metrics_subset", type=int, help="Number of images to use for metrics calculation")
    parser.add_argument("--precision", type=str, help="Precision to use for quantization (e.g., 'int4', 'int8')")
    parser.add_argument("--model_name", type=str, default=None, help="Name of the model to use from config.json")
    parser.add_argument("--dataset_name", type=str, default=None, help="Name of the dataset to use from config.json")
    args = parser.parse_args()

    # Step 0: Load settings from config.json
    config = load_config()
    apply_pruning = config["optimization"]["apply_pruning"]
    apply_quantization = config["optimization"]["apply_quantization"]
    apply_kvcache = config["optimization"]["apply_kvcache"]
    is_optimization_enabled = config["optimization"]["is_optimization_enabled"]

    print(f"Configuration loaded: pruning={apply_pruning}, quantization={apply_quantization}, kvcache={apply_kvcache}, optimization_enabled={is_optimization_enabled}")

    # Create a dictionary of arguments to pass to the optimization modules
    opt_args = {}
    if "model_params" in config:
        opt_args = config["model_params"]
    
    # Override with command line arguments if provided
    if args.num_images is not None:
        opt_args["num_images"] = args.num_images
    if args.inference_steps is not None:
        opt_args["inference_steps"] = args.inference_steps
    if args.guidance_scale is not None:
        opt_args["guidance_scale"] = args.guidance_scale
    if args.skip_metrics:
        opt_args["skip_metrics"] = True
    if args.metrics_subset is not None:
        opt_args["metrics_subset"] = args.metrics_subset
    if args.precision is not None:
        opt_args["precision"] = args.precision
    # if args.model_name is not None:
    #     opt_args["model_name"] = args.model_name

    # Step 1: Load dataset for example MS COCO 2017 or Flickr30k
    if args.dataset_name is not None:
        datasets = config["datasets"]
        dataset = next((d for d in datasets if d["name"] == args.dataset_name), None)
        if dataset is not None:
            print(f"Loading dataset: {dataset['name']}")
            opt_args["dataset_name"] = dataset["name"]
            opt_args["caption_path"] = dataset["caption_path"]
            opt_args["images_path"] = dataset["images_path"]

    module_args = Namespace(**opt_args)
    if is_optimization_enabled:
        print("Optimization model is enabled.")
        
        # Apply both quantization and pruning if specified
        if apply_pruning and apply_quantization:
            print("Applying both pruning and quantization...")
            try:
                # Add the project root to path to ensure modules can be imported
                sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
                
                # Import the combined pruning and quantization module
                from pruning.quant_pruning_coco import main as quant_pruning_main
                quant_pruning_main(module_args)
            except Exception as e:
                print(f"Error when doing quant_pruning module: {e}")

        # Apply pruning if specified
        if apply_pruning:
            print("Applying pruning...")
            try:
                # Add the project root to path to ensure modules can be imported
                sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
                
                # Import the pruning module
                from pruning.pruning_coco import main as pruning_main
                pruning_main(module_args)
            except Exception as e:
                print(f"Error when doing pruning module: {e}")

        # Apply quantization if specified
        if apply_quantization:
            print("Applying quantization...")
            try:
                # Add the project root to path to ensure modules can be imported
                sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
                
                # Import the quantization module
                from quantization.quant_coco import main as quantization_main
                quantization_main(module_args)
            except Exception as e:
                print(f"Error when doing quantization module: {e}")
        
        if apply_kvcache:
            print("Applying kv cache...")
            try:
                sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
            except Exception as e:
                print(f"Error when doing kv cache module: {e}")
    else:
        print("Optimization model is not enabled.")
        # Proceed with normal operations model without optimization
        try:
            # Add the project root to path to ensure modules can be imported
            sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
            
            # Import the normal image generation module
            from normal.normal_coco import main as normal_main
            normal_main(module_args)
        except Exception as e:
            print(f"Error when doing normal module: {e}")

    # # Step 2: Preprocess the dataset
    # print("Preprocessing dataset...")
    # # Call the preprocessing function here

    # # Step 3: Generate images
    # print("Generating images...")
    # # Call the image generation function here

    # # Step 4: Evaluate generated images
    # print("Evaluating generated images...")
    # # Call the evaluation function here with defined metrics

if __name__ == "__main__":
    main()
