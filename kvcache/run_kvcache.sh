#!/bin/bash

# Lỗi "Intermediate timesteps for SCM is not supported when num_inference_steps != 2"
# xảy ra vì mô hình SCM yêu cầu đúng 2 bước suy luận.

# ==============================================
# Các lệnh chạy cho mô hình Stable Consistency (SCM)
# ==============================================

# Mô hình SCM bắt buộc phải sử dụng đúng 2 bước suy luận
echo "Running SANA with SCM model (2 inference steps)..."
conda activate optsd && python kvcache/sana_kvcache_without_unet.py --prompt "A girl going into a wooden building ." --guidance_scale 3.5 --steps 2 --benchmark --num_runs 1

# Sử dụng cách tiếp cận gốc với UNet
echo "Running SANA with original UNet approach (2 inference steps)..."
conda activate optsd && python kvcache/sana_kvcache.py --prompt "A girl going into a wooden building ." --guidance_scale 3.5 --steps 2 --benchmark --num_runs 1

# ==============================================
# Các lệnh chạy cho mô hình SANA thông thường (không phải SCM)
# ==============================================

# Nếu bạn không sử dụng SCM mà là SANA thông thường, hãy bỏ comment các dòng sau:
# echo "Running with standard SANA model (30 inference steps)..."
# conda activate optsd && python kvcache/sana_kvcache_without_unet.py --prompt "A girl going into a wooden building ." --guidance_scale 3.5 --steps 30 --benchmark --num_runs 1

# =========================================================
# Script chạy SANA model với KV caching không sử dụng UNet
# =========================================================

# Đây là script tối ưu cho việc chạy SANA với transformer thay vì UNet

# Lưu ý quan trọng: 
# - Mô hình SCM yêu cầu đúng 2 bước suy luận (steps=2)
# - Mô hình SANA thông thường có thể sử dụng nhiều bước hơn (steps=20-50)

# 1. Chạy với SCM (Stable Consistency Models) - buộc phải dùng 2 bước suy luận
# echo "Running with SCM(Stable Consistency Models) model (2 inference steps)..."
# conda activate optsd && python kvcache/sana_kvcache_without_unet.py \
#   --prompt "A girl going into a wooden building ." \
#   --guidance_scale 3.5 \
#   --steps 2 \
#   --benchmark \
#   --num_runs 1

# 2. Chạy với mô hình SANA thông thường (30 bước suy luận)
# echo "Running with standard SANA model (30 inference steps)..."
# conda activate optsd && python kvcache/sana_kvcache_without_unet.py \
#   --prompt "A girl going into a wooden building ." \
#   --guidance_scale 3.5 \
#   --steps 30 \
#   --benchmark \
#   --num_runs 1

# 3. Tùy chọn chạy với debug mode để có thêm thông tin
# echo "Running in debug mode..."
# conda activate optsd && python kvcache/sana_kvcache_without_unet.py \
#   --prompt "A girl going into a wooden building ." \
#   --guidance_scale 3.5 \
#   --steps 2 \
#   --debug \
#   --benchmark \
#   --num_runs 1
rm -rf kvcache/outputs/*

conda activate optsd
python kvcache/sana_kvcache.py --use_coco --num_images 1000 --steps 50 --guidance_scale 3.5 --metrics_subset 5 --monitor_vram

python kvcache/sana_kvcache.py --use_flickr8k --num_images 1000 --steps 50 --guidance_scale 3.5 --metrics_subset 5 --monitor_vram