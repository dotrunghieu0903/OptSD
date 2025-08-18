import os
from tqdm import tqdm
from PIL import Image

def resize_images(generated_image_dir, resized_generated_image_dir, image_dimensions):
    # Sử dụng tqdm để có thanh tiến trình cho việc resize image
    for filename in tqdm(os.listdir(generated_image_dir), desc="Resizing images"):
        generated_path = os.path.join(generated_image_dir, filename)
        resized_path = os.path.join(resized_generated_image_dir, filename)

        # Đảm bảo đó là file và không phải thư mục
        if os.path.isfile(generated_path):
            # Lấy kích thước gốc từ dictionary
            if filename in image_dimensions:
                original_width, original_height = image_dimensions[filename]

                try:
                    # Mở ảnh đã sinh ra
                    img = Image.open(generated_path)

                    # Resize ảnh
                    # Sử dụng Image.LANCZOS cho chất lượng cao hơn khi resize
                    resized_img = img.resize((original_width, original_height), Image.LANCZOS)

                    # Lưu ảnh đã resize vào thư mục mới
                    resized_img.save(resized_path)
                    print(f"Resized and saved {filename} to {resized_generated_image_dir}")

                except Exception as e:
                    print(f"Error when resizz or saved {filename}: {e}")
            else:
                print(f"No found size for {filename} in dictionary image_dimensions")