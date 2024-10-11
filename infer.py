import json
import cv2
import os
import numpy as np
import torch
from models.common import DetectMultiBackend  # Kendi modelinizi buna göre güncelleyin


test_images_path = 'datathon/test/test_images'  # Bu yolu güncelleyin

# Görüntü adı-ID eşleştirmesini yükle
image_file_name_to_image_id = json.load(open('image_file_name_to_image_id.json'))

# Modeli yükleme (örnek olarak bir önceden eğitilmiş model kullanılıyor)
model = DetectMultiBackend('yolov5s.pt')  # Kendi modelinizin yolunu güncelleyin
model.eval()  # Modeli değerlendirme moduna al

def pre_process(image):
    # Görüntüyü yeniden boyutlandırma
    input_size = (640, 640)  # Modelin beklediği giriş boyutu
    image_resized = cv2.resize(image, input_size)

    # Normalizasyon
    image_normalized = image_resized / 255.0

    # Batching
    image_batched = np.expand_dims(image_normalized, axis=0)

    return image_batched

def model(image):
    # Tensor'a dönüştürme
    image_tensor = torch.from_numpy(image).float()

    # Modeli çalıştırma
    with torch.no_grad():
        predictions = model(image_tensor)[0]

    bboxes = predictions[..., :4]
    scores = predictions[..., 4]
    labels = predictions[..., 5]

    return bboxes, labels, scores

results = []
for img_name in os.listdir(test_images_path):
    image = cv2.imread(os.path.join(test_images_path, img_name))

    # Ön işleme
    data = pre_process(image)
    bboxes, labels, scores = model(data)

    img_id = image_file_name_to_image_id.get(img_name, None)
    if img_id is not None:
        for bbox, label, score in zip(bboxes, labels, scores):
            bbox[2], bbox[3] = bbox[2] - bbox[0], bbox[3] - bbox[1]  # xyxy'den xywh'ye dönüştür
            res = {
                'image_id': img_id,
                'category_id': int(label) + 1,
                'bbox': list(bbox.astype('float64')),
                'score': float("{:.8f}".format(score.item()))
            }
            results.append(res)

# Sonuçları JSON dosyasına yaz
with open('your_name.json', 'w') as f:
    json.dump(results, f)

print("Sonuçlar başarıyla kaydedildi: your_name.json")
