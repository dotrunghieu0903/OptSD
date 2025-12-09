# OptSD Reproduction Guide

This guide provides complete reproduction materials for the OptSD (Optimized Stable Diffusion) project, including configurations, seeds, prompts, and measurement scripts for independent verification.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Environment Setup](#environment-setup)
3. [Project Structure](#project-structure)
4. [Configuration Files](#configuration-files)
5. [Data Preparation](#data-preparation)
6. [Experimental Design](#experimental-design)
7. [Reproduction Steps](#reproduction-steps)
8. [Performance Metrics](#performance-metrics)
9. [Results Verification](#results-verification)
10. [Troubleshooting](#troubleshooting)

## Project Overview

OptSD implements several optimization techniques for Stable Diffusion models:

- **Baseline**: Standard Stable Diffusion v1.5
- **Quantization**: INT8 and INT4 precision reduction
- **Flash Attention**: Memory-efficient attention mechanism
- **KV Cache**: Key-Value cache optimization
- **Combined Optimizations**: Multiple techniques applied together

## Environment Setup

### Prerequisites
- NVIDIA GPU with CUDA support (minimum 8GB VRAM)
- Conda or Miniconda installed
- Python 3.9 or higher
- At least 32GB system RAM

### Installation Steps

1. **Create and activate conda environment:**
   ```bash
   conda create -n optsd python=3.9
   conda activate optsd
   ```

2. **Install PyTorch with CUDA support:**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Install core dependencies:**
   ```bash
   pip install diffusers transformers accelerate
   pip install opencv-python pillow numpy
   pip install lpips pytorch-fid clip-score
   pip install GPUtil psutil
   ```

## Project Structure

```
OptSD/
├── README.md                   # Project overview
├── REPRODUCTION_GUIDE.md       # This file
├── config.json                 # Main configuration
├── app.py                      # Main application entry point
├── requirements.txt            # Python dependencies
├── experiments/
│   ├── configs/               # Experiment configurations
│   │   ├── baseline.json
│   │   ├── quantization.json
│   │   ├── flash_attention.json
│   │   └── combined.json
│   ├── prompts/               # Standardized prompts
│   │   └── coco_prompts.json
│   ├── scripts/               # Reproduction scripts
│   │   ├── run_experiments.py
│   │   ├── measure_performance.py
│   │   └── verify_results.py
│   └── results/               # Output directory
├── dataset/                    # Dataset processing
│   ├── coco.py
│   └── generate_prompts.py
├── quantization/               # Quantization modules
├── flash_attn/                # Flash attention modules
├── kvcache/                   # KV cache modules
├── combination/               # Combined optimization modules
└── normal/                    # Baseline modules
```

## Configuration Files

### Main Configuration (config.json)
The main configuration file defines:
- Model parameters (inference steps, guidance scale, image dimensions)
- Dataset configuration (COCO val2017 subset)
- Optimization flags for each technique
- Module paths and dependencies
- Output directories and metrics calculation

### Experiment Configurations

**Baseline Configuration:**
- No optimizations applied
- Standard float16 precision
- Expected baseline performance metrics

**Quantization Experiments:**
- INT8 quantization configuration
- INT4 quantization configuration
- Precision settings and expected performance improvements

**Flash Attention Experiments:**
- Basic Flash Attention implementation
- Enhanced pipeline with error handling
- Memory optimization settings

**Combined Optimization:**
- Multiple techniques applied simultaneously
- Interaction effects and performance expectations

## Data Preparation

### COCO Dataset Setup

1. **Download COCO 2017 validation set:**
   - Images: ~1GB (5,000 images)
   - Annotations: ~241MB (captions and metadata)

2. **Extract and organize data:**
   ```
   data/
   ├── annotations/
   │   └── captions_val2017.json
   └── val2017/
       ├── 000000000139.jpg
       ├── 000000000285.jpg
       └── ... (5,000 images)
   ```

3. **Generate standardized prompts:**
   - Fixed seed (42) for reproducibility
   - 1,000 selected prompts with enhanced descriptions
   - Deterministic ordering and selection criteria

### Prompt Enhancement Strategy
- Remove trailing punctuation
- Add quality-enhancing prefixes ("professional photograph", "high detail")
- Include technical suffixes ("sharp focus", "high resolution")
- Filter out poor quality captions (too short, generic phrases)

## Experimental Design

### Reproducibility Measures
- **Fixed Seeds**: All random number generators seeded consistently
  - Global seed: 42
  - PyTorch seed: 12345
  - NumPy seed: 67890
  - Random seed: 54321

- **Deterministic Behavior**: 
  - CUDA deterministic mode enabled
  - Consistent batch ordering
  - Fixed prompt selection and ordering

### Experimental Parameters
- **Number of Images**: 1,000 per experiment
- **Inference Steps**: 50 (standard quality)
- **Guidance Scale**: 7.5 (balanced creativity/adherence)
- **Image Resolution**: 512x512 pixels
- **Batch Size**: 4 (memory optimization)

### Baseline Expectations
- **Inference Time**: ~2.5 seconds per image
- **Memory Usage**: ~6.2 GB GPU memory
- **Image Quality**: FID ≤ 16.0, CLIP Score ≥ 0.30

### Optimization Targets
- **Quantization (INT8)**: 30-40% speed improvement, 35% memory reduction
- **Quantization (INT4)**: 60-80% speed improvement, 55% memory reduction
- **Flash Attention**: 15-25% speed improvement, 25% memory reduction
- **Combined**: Multiplicative benefits with minimal quality loss

## Reproduction Steps

### Step 1: Environment Preparation
1. Follow environment setup instructions
2. Verify GPU availability and CUDA installation
3. Test basic PyTorch GPU functionality

### Step 2: Data Download and Processing
1. Download COCO 2017 validation dataset
2. Extract to appropriate directories
3. Generate standardized prompt set with fixed seeds
4. Verify data integrity and completeness

### Step 3: Baseline Experiment
1. Configure baseline settings (no optimizations)
2. Run baseline experiment with full metrics collection
3. Verify baseline performance against expected values
4. Save baseline results as reference

### Step 4: Optimization Experiments
1. Run each optimization technique separately:
   - INT8 quantization
   - INT4 quantization
   - Flash Attention (basic)
   - Flash Attention (enhanced)
   - KV Cache optimization

2. Run combined optimization experiments:
   - Flash Attention + KV Cache
   - Quantization + Flash Attention
   - Triple combination (if applicable)

### Step 5: Performance Measurement
For each experiment, measure:
- **Timing**: Average inference time per image
- **Memory**: Peak GPU and CPU memory usage
- **Quality**: FID, LPIPS, CLIP Score, Inception Score
- **Model Size**: Optimized model storage requirements

### Step 6: Results Verification
1. Compare results against expected performance ranges
2. Verify reproducibility across multiple runs
3. Generate performance comparison tables
4. Create visualization of optimization trade-offs

## Performance Metrics

### Image Quality Metrics
- **FID (Fréchet Inception Distance)**: Lower is better, measures distribution similarity
- **LPIPS (Learned Perceptual Image Patch Similarity)**: Lower is better, perceptual similarity
- **CLIP Score**: Higher is better, text-image alignment
- **Inception Score**: Higher is better, image quality and diversity

### Performance Metrics
- **Inference Time**: Average seconds per image generation
- **Memory Usage**: Peak GPU memory consumption
- **Throughput**: Images generated per second
- **Model Size**: Storage requirements for optimized models

### Efficiency Metrics
- **Speedup**: Baseline time / Optimized time
- **Memory Reduction**: (Baseline memory - Optimized memory) / Baseline memory
- **Quality Retention**: Optimized quality / Baseline quality

## Results Verification

### Verification Criteria
- **Performance Tolerance**: ±10% variation acceptable
- **Quality Tolerance**: ±5% variation for FID, CLIP Score
- **Reproducibility**: Results should be consistent across runs
- **Statistical Significance**: Multiple seed validation

### Expected Results Table

| Technique | Speedup | Memory Reduction | FID Score | CLIP Score |
|-----------|---------|------------------|-----------|------------|
| Baseline | 1.00x | 0% | 15.8 | 0.320 |
| INT8 Quant | 1.39x | 34% | 16.2 | 0.315 |
| INT4 Quant | 1.79x | 55% | 17.1 | 0.295 |
| Flash Attn | 1.19x | 23% | 15.9 | 0.318 |
| Combined | 1.65x | 42% | 16.5 | 0.308 |

### Validation Process
1. **Automated Verification**: Scripts compare results against expected ranges
2. **Statistical Testing**: Multiple runs with different seeds
3. **Quality Assessment**: Visual inspection of generated images
4. **Performance Profiling**: Detailed timing and memory analysis

## Troubleshooting

### Common Issues

**CUDA Out of Memory:**
- Reduce batch size to 1 or 2
- Enable gradient checkpointing
- Use mixed precision training

**Slow Performance:**
- Verify GPU utilization
- Check for CPU bottlenecks
- Ensure proper CUDA installation

**Quality Degradation:**
- Verify quantization settings
- Check for numerical instabilities
- Validate model weights after optimization

**Reproducibility Issues:**
- Confirm all seeds are set correctly
- Check for non-deterministic operations
- Verify data loading order

### Performance Debugging
1. **Profile Memory Usage**: Use nvidia-smi and torch profiler
2. **Timing Analysis**: Measure each pipeline stage separately
3. **Quality Metrics**: Compare individual image generations
4. **Model Validation**: Verify optimization didn't corrupt weights

### Getting Help
- Check project issues on GitHub repository
- Verify environment matches requirements exactly
- Compare your results with provided baseline measurements
- Document any deviations from expected performance

## Citation and Acknowledgments

If you use this reproduction package, please cite:

```bibtex
@misc{optsd_reproduction_2025,
  title={OptSD: The Efficiency Diffusion Models and Optimization Techniques: A Comprehensive Review},
  author={[Trung-Hieu Do and Vinh-Tiep Nguyen]},
  year={2025},
  howpublished={\url{https://github.com/dotrunghieu0903/OptSD}}
}
```

## License and Usage

This reproduction package is provided under [License Type]. Please ensure compliance with:
- Stable Diffusion model license terms
- COCO dataset usage requirements  
- Individual optimization technique licenses

---

**Last Updated**: December 10, 2024
**Version**: 1.0
**Compatibility**: Tested on CUDA 11.8, PyTorch 2.0+
