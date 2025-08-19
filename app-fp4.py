import torch
from diffusers import FluxPipeline
from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.utils import get_precision

# Kiểm tra GPU và sử dụng fp4 cho Blackwell
precision = get_precision()
print(f"Đã phát hiện precision phù hợp: {precision}")

# Đảm bảo rằng đang sử dụng fp4 cho Blackwell
if precision != "fp4":
    print("Cảnh báo: GPU của bạn không phải Blackwell hoặc không hỗ trợ fp4 quantization")
    print("Đang tiếp tục với precision được phát hiện...")

# Tải mô hình với precision phù hợp
transformer = NunchakuFluxTransformer2dModel.from_pretrained(
    f"nunchaku-tech/nunchaku-flux.1-dev/svdq-{precision}_r32-flux.1-dev.safetensors"
)

# Sử dụng trong pipeline
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=torch.bfloat16
).to("cuda")

# Chạy pipeline
prompt = "A futuristic city with tall skyscrapers"
image = pipeline(prompt, num_inference_steps=50, guidance_scale=3.5).images[0]
image.save(f"blackwell-fp4-{precision}.png")
print(f"Đã tạo ảnh với mô hình {precision} trên GPU")

# check compatibility
from nunchaku.utils import check_hardware_compatibility

# Giả sử bạn có quantization_config
quantization_config = {"weight": {"dtype": "fp4_e2m1_all"}}

try:
    # Kiểm tra tương thích
    check_hardware_compatibility(quantization_config)
    print("GPU tương thích với cấu hình quantization")
except ValueError as e:
    print(f"Lỗi: {e}")
    # Xử lý khi không tương thích