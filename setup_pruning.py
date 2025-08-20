"""
Installation script to set up pruning requirements
"""

import sys
import subprocess
import os

def install_requirements():
    """Install required packages for pruning."""
    print("Installing required packages for pruning...")
    
    # List of required packages
    required_packages = [
        "torch>=2.0.0",
        "matplotlib",
        "tqdm",
        "pillow",
        "numpy"
    ]
    
    # Install using pip
    for package in required_packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    # Check if required torch utils are available
    try:
        import torch.nn.utils.prune as prune
        print("PyTorch pruning modules successfully imported!")
    except ImportError:
        print("Warning: torch.nn.utils.prune not available. Make sure you have PyTorch >=1.4.0")

if __name__ == "__main__":
    install_requirements()
    print("\nSetup complete!")
    print("\nTo use pruning, run one of the following:")
    print("  python app-pruned.py")
    print("  python benchmark_pruning.py")
    print("\nSee pruning_guide.md for more information on pruning techniques and best practices.")
