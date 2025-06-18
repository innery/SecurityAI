import os
import numpy as np
from keras_facenet import FaceNet
from PIL import Image

embedder = FaceNet()

FACES_DIR = "faces"
EMBEDDINGS_DIR = "embeddings"

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

for filename in os.listdir(FACES_DIR):
    if filename.endswith((".jpg", ".png")):
        name = os.path.splitext(filename)[0]
        img_path = os.path.join(FACES_DIR, filename)
        img = Image.open(img_path).resize((160, 160))
        embedding = embedder.embeddings([np.array(img)])[0]
        np.save(os.path.join(EMBEDDINGS_DIR, f"{name}.npy"), embedding)
        print(f"[✓] {name} için embedding oluşturuldu.")