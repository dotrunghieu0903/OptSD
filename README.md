# Diffusion-Model-Optimization
<p align='center'>
    </br>
    <img src='SDOpt.png' width='300'>
</p>

## Module 1. Quantization Model: Unet & Transformer
### 1.1. Quantization Aware Training
### 1.2. Post Training Quantization
### 1.3. Quantization granularity
### 1.4. Quantization range
### 1.5. Asymmetric and symmetric quantization

## Module 2. Key-Value Cache (Decode & Greedy) - Auto regressive+Bidirectional Attention
## Module 3. Distillation Model: Distill GPT, Tiny LLaMa
### 3.1 Num-Inference-steps
### 3.2 Num-Image-per-prompt
## Module 4. Flash Attention
## Module 5. PEFT(Parameter-Efficient-Fine-Tuning)/LoRA
## Module 6. Early Exit
## Module 7. Model Pruning
### 7.1 Magnitude Pruning
### 7.2 Structured Pruning
### 7.3 Layerwise Pruning
### 7.4 Attention Head Pruning


# Flash Attention Enhanced Diffusion Model - Metrics Integration Guide

This guide explains how to use the enhanced `example.py` script with integrated metrics calculation and VRAM monitoring, similar to the features found in `combined_optimization.py`.

## 🚀 New Features Added

### 1. Image Quality Metrics
- **FID Score**: Fréchet Inception Distance for comparing generated vs real images
- **CLIP Score**: Measures text-image alignment using CLIP embeddings
- **Image Reward**: Measures image quality using ImageReward model
- **LPIPS**: Learned Perceptual Image Patch Similarity for perceptual distance
- **PSNR**: Peak Signal-to-Noise Ratio for pixel-level comparison

### 2. VRAM Monitoring
- Real-time GPU memory usage tracking during generation
- Memory usage statistics (average, peak, samples)
- Optional VRAM monitoring with `--monitor-vram` flag

### 3. Enhanced Output Structure
- Organized output directories with timestamps
- Comprehensive metadata files (JSON format)
- Generation summary reports
- Metrics results saved separately

## 📋 Command Line Arguments

### New Arguments Added:

```bash
# Metrics Control
--skip-metrics              # Skip calculation of image quality metrics
--metrics-subset N          # Number of images to use for metrics (default: 500)
--monitor-vram             # Enable VRAM monitoring during generation

# Output Control
--output-dir DIR           # Base output directory (default: flash_attention_outputs)
--prompt "text"           # Single prompt for generation when dataset is 'none'

# Dataset Options
--dataset {flickr8k,coco,none}  # Dataset choice (default: none)
```

## 🛠️ Usage Examples

### 1. Generate from Flickr8k Dataset with Full Metrics
```bash
python flash_attention/example.py \
    --dataset flickr8k \
    --num-images 50 \
    --flash-attn \
    --kv-cache \
    --monitor-vram \
    --precision int4 \
    --steps 30 \
    --guidance 7.5 \
    --height 512 \
    --width 512
```

### 2. Generate Single Image with VRAM Monitoring
```bash
python flash_attention/example.py \
    --dataset none \
    --prompt "a beautiful landscape with mountains and a lake, high detail, professional lighting" \
    --flash-attn \
    --kv-cache \
    --monitor-vram \
    --height 1024 \
    --width 1024
```

### 3. Skip Metrics for Faster Generation
```bash
python flash_attention/example.py \
    --dataset flickr8k \
    --num-images 100 \
    --skip-metrics \
    --flash-attn \
    --precision int8
```

### 4. Calculate Metrics on Subset Only
```bash
python flash_attention/example.py \
    --dataset flickr8k \
    --num-images 1000 \
    --metrics-subset 100 \
    --flash-attn \
    --kv-cache
```

## 📁 Output Structure

When you run the script, it creates an organized output directory:

```
{output-dir}/
└── {dataset}_{optimizations}_{timestamp}/
    ├── generated_image1.png
    ├── generated_image2.png
    ├── ...
    ├── resized/                    # Resized images for FID calculation
    │   ├── generated_image1.png
    │   └── ...
    ├── generation_metadata.json   # Detailed metadata for each image
    ├── metrics_results.json      # All calculated metrics
    └── generation_summary.txt    # Human-readable summary
```

### Example Output Directory Name:
`flickr8k_flash_kvcache_quant_int4_20240915_143022/`

## 📊 Metrics Output

### metrics_results.json Example:
```json
{
  "clip_score": 24.5832,
  "image_reward": 0.7234,
  "fid_score": 12.3456,
  "lpips_score": 0.2345,
  "psnr_score": 28.9876
}
```

### generation_metadata.json Example:
```json
[
  {
    "filename": "image1.jpg",
    "output_path": "/path/to/generated_image1.png",
    "caption": "a dog sitting in the grass",
    "generation_time": 3.45,
    "optimization_settings": {
      "flash_attention": true,
      "kv_cache": true,
      "precision": "int4",
      "pruning": 0.0
    },
    "generation_settings": {
      "steps": 30,
      "guidance_scale": 7.5,
      "height": 512,
      "width": 512
    },
    "average_vram_gb": 8.23,
    "peak_vram_gb": 9.87
  }
]
```

## 🔧 How the Metrics Work

### 1. **CLIP Score**
- Measures semantic alignment between text prompts and generated images
- Higher scores indicate better text-image matching
- Range: Typically 0-100, higher is better

### 2. **FID Score**
- Compares generated images to real images using Inception features
- Lower scores indicate better quality (closer to real images)
- Requires original images for comparison

### 3. **Image Reward**
- Uses a trained reward model to assess image quality
- Higher scores indicate better perceived quality
- Range: Typically -2 to +2, higher is better

### 4. **LPIPS**
- Measures perceptual similarity between generated and original images
- Lower scores indicate better similarity
- Range: 0-1, lower is better

### 5. **PSNR**
- Pixel-level comparison metric
- Higher scores indicate better pixel accuracy
- Range: Typically 20-40 dB, higher is better

## 🚨 Memory Management

The enhanced script includes several memory optimizations:

1. **Automatic Memory Cleanup**: Regular `torch.cuda.empty_cache()` calls
2. **Batch Processing**: Processes images one by one to avoid memory buildup
3. **Error Recovery**: Continues generation even if individual images fail
4. **VRAM Monitoring**: Tracks memory usage to identify bottlenecks

## ⚠️ Important Notes

1. **Dataset Requirements**: 
   - For Flickr8k: Ensure `flickr8k/Images/` and `flickr8k/captions.txt` exist
   - For COCO: Ensure proper COCO dataset structure

2. **Metrics Dependencies**:
   - ImageReward: `pip install image-reward`
   - LPIPS: `pip install lpips`
   - CLIP: `pip install transformers`
   - FID: `pip install clean-fid`

3. **VRAM Monitoring**:
   - Requires `pynvml`: `pip install pynvml`
   - Or `nvidia-ml-py3`: `pip install nvidia-ml-py3`

4. **Performance Impact**:
   - Metrics calculation adds processing time
   - Use `--skip-metrics` for faster generation
   - Use `--metrics-subset` to limit metric calculation

## 🔍 Troubleshooting

### Common Issues:

1. **Out of Memory**: Use smaller batch sizes, reduce image resolution, or use quantization
2. **Missing Dependencies**: Install required packages for metrics
3. **Dataset Not Found**: Check dataset paths and structure
4. **VRAM Monitoring Fails**: Install GPU monitoring libraries

### Performance Tips:

1. Use `--metrics-subset` to limit expensive metric calculations
2. Enable `--flash-attn` and `--kv-cache` for faster generation
3. Use quantization (`--precision int4`) for memory efficiency
4. Skip metrics with `--skip-metrics` during development

This enhanced script now provides the same comprehensive metrics and monitoring capabilities as `combined_optimization.py` while maintaining the Flash Attention optimization focus.