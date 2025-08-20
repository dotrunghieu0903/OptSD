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

# Run quantization + pruning module
./pruning/run_quant_pruning.sh

# Run pruning evaluation
./pruning/run_pruning.sh

# Run quantization
./quantization/run_quantization.sh