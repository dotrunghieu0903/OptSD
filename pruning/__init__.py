# This file makes the pruning directory a Python package
# This allows for relative imports within the package

from .pruned import get_model_size, get_sparsity, apply_magnitude_pruning
# Export commonly used functions for easier imports
