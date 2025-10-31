# Run app.py
python app.py

python app.py --dataset_name "flickr8k" --num_images 5 --inference_steps 50 --guidance_scale 3.5 --metrics_subset 5
python app.py --dataset_name "coco" --num_images 5 --inference_steps 50 --guidance_scale 3.5 --metrics_subset 5

# Run quantization + pruning module
python pruning/quant_pruning.py --pruning_amount 0.15 --num_images 100 --steps 25 --guidance_scale 3.5 --metrics_subset 100 --use_coco

# Run kv cache
python kvcache/sana_kvcache.py --use_coco --num_images 1000 --steps 50 --guidance_scale 3.5 --metrics_subset 5 --monitor_vram
python kvcache/sana_kvcache.py --use_flickr8k --num_images 1000 --steps 50 --guidance_scale 3.5 --metrics_subset 5 --monitor_vram

# Make script executable
chmod +x normal_coco.py

# Run directly from the normal directory
python normal_coco.py --num_images 100 --steps 25 --guidance_scale 7.5

# Run flash attention app with different settings from inside instead of app.py
python flash_attn/flash_attn_app.py --monitor-vram --metrics-subset 1000 --num-images 1000 --steps 25 --dataset coco --flash-attn --kv-cache --pruning 0.3 --precision int4
python flash_attn/flash_attn_app.py --monitor-vram --metrics-subset 1000 --num-images 1000 --steps 25 --dataset flickr8k --flash-attn --kv-cache --pruning 0.3 --precision int4

# Run pruning analysis script with different methods
python pruning/generate_synthetic_graph.py
python pruning/generate_synthetic_graph.py --method magnitude --num_images 50 --dataset coco