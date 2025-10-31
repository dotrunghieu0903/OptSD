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

def import_module_main(module_path, main_function="main"):
    """
    Dynamically import a module and return its main function
    """
    try:
        module = __import__(module_path, fromlist=[main_function])
        return getattr(module, main_function)
    except ImportError as e:
        print(f"Error importing module {module_path}: {e}")
        return None
    except AttributeError as e:
        print(f"Error getting function {main_function} from module {module_path}: {e}")
        return None

def  execute_optimization_module(module_name, config, args, modules_config):
    """
    Execute a specific optimization module based on configuration
    """
    try:
        module_config = modules_config.get(module_name, {})
        module_path = module_config.get("path")
        
        if not module_path:
            print(f"No path configured for module {module_name}")
            return False
            
        main_function = import_module_main(module_path, "main")
        if main_function:
            print(f"Executing {module_name} optimization...")
            main_function(args)
            return True
        else:
            print(f"Could not import main function from {module_path}")
            return False
    except Exception as e:
        print(f"Error when executing {module_name} module: {e}")
        return False

def main():
    # Parse command-line arguments for optional overrides
    parser = argparse.ArgumentParser(description="Apply optimization techniques to diffusion models")
    parser.add_argument("--num_images", type=int, help="Number of images to process")
    parser.add_argument("--inference_steps", type=int, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, help="Guidance scale for image generation")
    parser.add_argument("--skip_metrics", action="store_true", help="Skip calculation of image quality metrics")
    parser.add_argument("--metrics_subset", type=int, help="Number of images to use for metrics calculation")
    parser.add_argument("--precision", type=str, choices=["float16", "bfloat16", "int8", "int4"], help="Precision to use for quantization (e.g., 'int4', 'int8')")
    parser.add_argument("--model_name", type=str, default=None, help="Name of the model to use from config.json")
    parser.add_argument("--dataset_name", type=str, default=None, help="Name of the dataset to use from config.json")
    parser.add_argument("--use_flash_attention", action="store_true", help="Use Flash Attention optimization")
    parser.add_argument("--use_kv_cache", action="store_true", help="Use KV Cache optimization")
    parser.add_argument("--enhanced", action="store_true", help="Use enhanced Flash Attention pipeline with robust error handling")
    
    args = parser.parse_args()

    # Step 0: Load settings from config.json
    config = load_config()
    apply_pruning = config["optimization"]["apply_pruning"]
    apply_quantization = config["optimization"]["apply_quantization"]
    apply_kvcache = config["optimization"]["apply_kvcache"]
    apply_flash_attention = config["optimization"].get("apply_flash_attention", False)
    is_optimization_enabled = config["optimization"]["is_optimization_enabled"]
    use_enhanced_pipeline = config["optimization"].get("use_enhanced_pipeline", False)
    
    # Get module configurations
    modules_config = config.get("modules", {})

    print(f"Configuration loaded: pruning={apply_pruning}, quantization={apply_quantization}, kvcache={apply_kvcache}, flash_attention={apply_flash_attention}, enhanced_pipeline={use_enhanced_pipeline}, optimization_enabled={is_optimization_enabled}")

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
    
    # Handle optimization technique flags
    if args.use_flash_attention:
        apply_flash_attention = True
        opt_args["use_flash_attention"] = True
    if args.use_kv_cache:
        apply_kvcache = True
        opt_args["use_kv_cache"] = True
    if args.enhanced:
        use_enhanced_pipeline = True
        opt_args["use_enhanced_pipeline"] = True
    
    if args.model_name is not None:
        opt_args["model_name"] = args.model_name

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
        
        # Add the project root to path to ensure modules can be imported
        sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
        
        # Define a function to apply pruning
        def apply_pruning_optimization(args):
            print("Applying pruning...")
            try:
                pruning_config = modules_config.get("pruning", {})
                pruning_path = pruning_config.get("path", "pruning.pruned")
                pruning_main = import_module_main(pruning_path, "main")
                if pruning_main:
                    pruning_main(args)
                    return True
                else:
                    print(f"Could not import main function from {pruning_path}")
                    return False
            except Exception as e:
                print(f"Error when applying pruning module: {e}")
                return False
        
        # Define a function to apply quantization
        def apply_quantization_optimization(args):
            print("Applying quantization...")
            try:
                success = execute_optimization_module("quantization", config, args, modules_config)
                return success
            except Exception as e:
                print(f"Error when executing quantization module: {e}")
                return False
        
        # Apply optimizations based on configuration
        optimizations_applied = 0
        
        # Apply pruning if specified
        if apply_pruning:
            if optimizations_applied > 0:
                print(f"Step {optimizations_applied + 1}: Applying pruning...")
            success = apply_pruning_optimization(module_args)
            if success:
                optimizations_applied += 1
        
        # Apply quantization if specified
        if apply_quantization:
            if optimizations_applied > 0:
                print(f"Step {optimizations_applied + 1}: Applying quantization...")
            success = apply_quantization_optimization(module_args)
            if success:
                optimizations_applied += 1
        
        # Apply KV cache if specified
        if apply_kvcache:
            print("Applying KV cache...")
            execute_optimization_module("kvcache", config, module_args, modules_config)
        
        # Apply Flash Attention if specified
        if apply_flash_attention:
            print("Applying Flash Attention...")
            try:
                flash_attention_config = modules_config.get("flash_attention", {})
                
                # Check if we should use the enhanced pipeline
                if use_enhanced_pipeline:
                    print("Using enhanced Flash Attention pipeline with robust error handling...")
                    flash_attention_path = flash_attention_config.get("enhanced_path", "flash_attn.flash_attn_pipeline")
                else:
                    flash_attention_path = flash_attention_config.get("path", "flash_attn.flash_attn_app")
                
                flash_attention_main = import_module_main(flash_attention_path, "main")
                if flash_attention_main:
                    flash_attention_main(module_args)
                else:
                    print(f"Could not import main function from {flash_attention_path}")
            except Exception as e:
                print(f"Error when applying Flash Attention: {e}")
                
        # Apply combined optimizations if multiple are specified
        if (apply_flash_attention and apply_kvcache) or (apply_flash_attention and apply_quantization):
            print("Applying combined optimizations...")
            try:
                combination_config = modules_config.get("combination", {})
                combination_path = combination_config.get("path", "combination.combined_optimization")
                combined_optimization_main = import_module_main(combination_path, "main")
                if combined_optimization_main:
                    combined_optimization_main(module_args)
                else:
                    print(f"Could not import main function from {combination_path}")
            except Exception as e:
                print(f"Error when applying combined optimizations: {e}")
    else:
        print("Optimization model is not enabled.")
        # Proceed with normal operations model without optimization
        try:
            # Add the project root to path to ensure modules can be imported
            sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
            
            normal_config = modules_config.get("normal", {})
            normal_path = normal_config.get("path", "normal.normal_coco")
            normal_main = import_module_main(normal_path, "main")
            if normal_main:
                normal_main(module_args)
            else:
                print(f"Could not import main function from {normal_path}")
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
