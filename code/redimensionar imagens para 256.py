# Notebook: redimensionar_imagens_para_256x256.ipynb

import os
from PIL import Image

# Caminho da pasta raiz
base_path = r"C:\Mestrado\Materias\pesquisa\tomates\tomatotest\processed_data_256_augumentation"

# Dimensões desejadas
TARGET_SIZE = (256, 256)

# Tipos de imagem válidos
VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

# Contador para estatísticas
total_images = 0
resized_images = 0

# Percorre recursivamente os diretórios
for root, _, files in os.walk(base_path):
    for filename in files:
        ext = os.path.splitext(filename)[1].lower()
        if ext in VALID_EXTENSIONS:
            image_path = os.path.join(root, filename)
            try:
                with Image.open(image_path) as img:
                    total_images += 1
                    if img.size != TARGET_SIZE:
                        img = img.resize(TARGET_SIZE, Image.BILINEAR)
                        img.save(image_path)
                        resized_images += 1
            except Exception as e:
                print(f"Erro ao processar {image_path}: {e}")

print(f"Total de imagens encontradas: {total_images}")
print(f"Imagens redimensionadas: {resized_images}")
