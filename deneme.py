import os
import shutil
import json

# Klasör yolları
source_folder = 'images'
destination_folder = 'train'

# JSON dosyası yolu
json_file_path = 'image_id/train_id.json'  # JSON dosyanızın tam yolu

# JSON dosyasını oku
with open(json_file_path, 'r') as file:
    files_to_move = json.load(file)

# Dosyaları taşı
for file_name in files_to_move.keys():
    source_file = os.path.join(source_folder, file_name)
    destination_file = os.path.join(destination_folder, file_name)
    
    # Dosyanın var olup olmadığını kontrol et
    if os.path.exists(source_file):
        shutil.move(source_file, destination_file)
        print(f'Taşındı: {file_name}')
    else:
        print(f'Bulunamadı: {file_name}')
