import cv2
import numpy as np
import os
from keras_facenet import FaceNet
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

embedder = FaceNet()

FACES_DIR = "faces"
EMBEDDINGS_DIR = "embeddings"

os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# Kayıtlı embedding'leri yükle
def load_embeddings():
    embeddings = {}
    for file in os.listdir(EMBEDDINGS_DIR):
        if file.endswith(".npy"):
            name = os.path.splitext(file)[0]
            vector = np.load(os.path.join(EMBEDDINGS_DIR, file))
            embeddings[name] = vector
    return embeddings

def extract_face(frame, box):
    x, y, w, h = box
    face = frame[y:y+h, x:x+w]
    face = cv2.resize(face, (160, 160))
    return face

def recognize(face_img, known_embeddings):
    embedding = embedder.embeddings([face_img])[0]
    for name, known_emb in known_embeddings.items():
        sim = cosine_similarity([embedding], [known_emb])[0][0]
        if sim > 0.6:
            return name
    return None

known_embeddings = load_embeddings()

cap = cv2.VideoCapture(1)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("🟢 Kamera başladı. 'e' tuşu ile yeni kişi ekle, 'q' ile çık.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = extract_face(frame, (x, y, w, h))
        name = recognize(face_img, known_embeddings)

        if name is None:
            name = "Bilinmiyor"

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("SecurityAI", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("e"):
        for (x, y, w, h) in faces:
            face_img = extract_face(frame, (x, y, w, h))
            name = input("📝 İsim gir: ")
            cv2.imwrite(os.path.join(FACES_DIR, f"{name}.jpg"), face_img)
            embedding = embedder.embeddings([face_img])[0]
            np.save(os.path.join(EMBEDDINGS_DIR, f"{name}.npy"), embedding)
            known_embeddings[name] = embedding
            print(f"[+] {name} sisteme eklendi.")
            break

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()