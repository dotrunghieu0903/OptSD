conda create -n optsd python=3.12 -y

conda activate optsd

conda env create -f environment.yml
conda env update -f environment.yml

# Download and extract COCO dataset manually
wget http://images.cocodataset.org/zips/val2017.zip -O coco/val2017.zip

# Download COCO 2017 Annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O coco/annotations_trainval2017.zip

unzip coco/val2017.zip -d coco
unzip coco/annotations_trainval2017.zip -d coco

# Download and extract flickr30k dataset manually
wget "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part00"
wget "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part01"
wget "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part02"
cat flickr30k_part00 flickr30k_part01 flickr30k_part02 > flickr30k.zip
rm flickr30k_part00 flickr30k_part01 flickr30k_part02
unzip -q flickr30k.zip -d ./flickr30k
rm flickr30k.zip


# Run quantization + pruning module
./pruning/run_quant_pruning.sh

# Run pruning evaluation
./pruning/run_pruning.sh

# Run quantization
./quantization/run_quant.sh
./quantization/run_quant_flickr8k.sh

python combined_optimization.py --model_path "black-forest-labs/FLUX.1-dev" --precision int4 --pruning_amount 0.3 --use_kv_cache --num_images 1 --steps 25
python combined_optimization.py --model_path "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers" --pruning_amount 0.3 --use_kv_cache --num_images 10 --steps 30

python combination/combined_optimization.py --model_path "black-forest-labs/FLUX.1-dev" --precision int4 --pruning_amount 0.3 --use_kv_cache --num_images 10 --steps 50 --monitor_vram
python combination/combined_optimization.py --model_path "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers" --precision int4 --pruning_amount 0.3 --use_kv_cache --num_images 1000 --steps 50 --monitor_vram


python combination/combined_optimization.py --model_path "black-forest-labs/FLUX.1-dev" --use_kv_cache --precision int4  --num_images 10 --steps 50 --monitor_vram
python combination/combined_optimization.py --model_path "black-forest-labs/FLUX.1-dev" --use_kv_cache --num_images 10 --steps 50 --monitor_vram
python combination/combined_optimization.py --model_path "black-forest-labs/FLUX.1-dev" --use_kv_cache --pruning_amount 0.0 --num_images 100 --steps 50 --monitor_vram --use_flickr8k
python combination/combined_optimization.py --model_path "black-forest-labs/FLUX.1-dev" --use_kv_cache --pruning_amount 0.3 --num_images 200 --steps 50 --monitor_vram --use_flickr8k

python combination/combined_optimization.py --model_path "black-forest-labs/FLUX.1-schnell" --use_kv_cache --precision int4  --num_images 10 --steps 50 --monitor_vram
python combination/combined_optimization.py --model_path "black-forest-labs/FLUX.1-schnell" --use_kv_cache --num_images 10 --steps 50 --monitor_vram
python combination/combined_optimization.py --model_path "black-forest-labs/FLUX.1-schnell" --use_kv_cache --pruning_amount 0.0 --num_images 100 --steps 50 --monitor_vram --use_flickr8k
python combination/combined_optimization.py --model_path "black-forest-labs/FLUX.1-schnell" --use_kv_cache --pruning_amount 0.3 --num_images 500 --steps 50 --monitor_vram --use_flickr8k

python combination/combined_optimization.py --model_path "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers" --use_kv_cache --num_images 10 --steps 50 --monitor_vram --use_flickr8k
python combination/combined_optimization.py --model_path "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers" --pruning_amount 0.3 --use_kv_cache --num_images 10 --steps 50 --monitor_vram --use_flickr8k

python kvcache/sam_kvcache.py --coco_dir /coco --num_images 5 --benchmark --num_runs 5 --metrics_subset 20

# FlashAttention
python flash_attention/flash_attn_app.py --dataset flickr8k --num-images 500 --flash-attn --kv-cache --monitor-vram --output-dir flash_attention --precision int4 --pruning 0.3 --steps 30

python flash_attention/flash_attn_app.py --dataset coco --num-images 500 --flash-attn --kv-cache --monitor-vram --output-dir flash_attention --precision int4 --pruning 0.3 --steps 30
python flash_attention/flash_attn_app.py --dataset coco --num-images 1 --flash-attn --monitor-vram --output-dir flash_attention --precision int4 --pruning 0.3 --steps 30
python flash_attention/flash_attn_app.py --dataset coco --num-images 10 --metrics-subset 10 --flash-attn --monitor-vram --output-dir flash_attention --precision int4 --pruning 0.3 --steps 30