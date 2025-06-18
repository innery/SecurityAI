import cv2
import numpy as np
import os
import json
import random
from keras_facenet import FaceNet
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Yüz tanıma modeli
embedder = FaceNet()

# Duygu modeli
emotion_model = load_model("/Users/aylac/SecurityAI/model/model_2.h5")

# Yol tanımları
FACES_DIR = "faces"
EMBEDDINGS_DIR = "embeddings"
SCORES_PATH = "/Users/aylac/SecurityAI/data/score/scores.json"

os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# Duygu ağırlıkları
emotion_weights = {
    "angry": 90,
    "disgust": 70,
    "fear": 80,
    "happy": 10,
    "sad": 60,
    "surprise": 20,
    "neutral": 30
}

# Sınıf etiketleri (model eğitiminle eşleşmeli!)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Embedding'leri yükle
def load_embeddings():
    embeddings = {}
    for file in os.listdir(EMBEDDINGS_DIR):
        if file.endswith(".npy"):
            name = os.path.splitext(file)[0]
            vector = np.load(os.path.join(EMBEDDINGS_DIR, file))
            embeddings[name] = vector
    return embeddings

# Skor dosyasını yükle
def load_scores():
    if os.path.exists(SCORES_PATH):
        with open(SCORES_PATH, 'r') as f:
            return json.load(f)
    return {}

# Skoru kaydet
def save_scores(scores):
    with open(SCORES_PATH, 'w') as f:
        json.dump(scores, f)

# Yüz kes
def extract_face(frame, box):
    x, y, w, h = box
    face = frame[y:y+h, x:x+w]
    return cv2.resize(face, (160, 160))

# Duygu tahmini
def predict_emotion(face):
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face_resized = cv2.resize(face_gray, (48, 48))
    face_array = img_to_array(face_resized) / 255.0
    face_array = np.expand_dims(face_array, axis=0)
    face_array = np.expand_dims(face_array, axis=-1)
    preds = emotion_model.predict(face_array)[0]
    label = emotion_labels[np.argmax(preds)]
    return label

# Yüz tanıma
def recognize(face_img, known_embeddings):
    embedding = embedder.embeddings([face_img])[0]
    for name, known_emb in known_embeddings.items():
        sim = cosine_similarity([embedding], [known_emb])[0][0]
        if sim > 0.6:
            return name
    return None

# Başlat
known_embeddings = load_embeddings()
person_scores = load_scores()

cap = cv2.VideoCapture(1)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("🎥 Başlatıldı. 'e' = kişi ekle, 'q' = çık.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = extract_face(frame, (x, y, w, h))
        name = recognize(face_img, known_embeddings)
        emotion = predict_emotion(face_img)

        if name is None:
            name = "Bilinmiyor"
            risk_score = 0
        else:
            # Skoru al
            if name not in person_scores:
                person_scores[name] = random.randint(0, 10)
                save_scores(person_scores)
            score = person_scores[name]
            weight = emotion_weights.get(emotion, 50)
            risk_score = score * weight

        label = f"{name} ({emotion})"

        if risk_score > 500:
            label += "TEHLIKE!"

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255) if risk_score > 500 else (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if risk_score > 500 else (0, 255, 0), 2)

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
            if name not in person_scores:
                person_scores[name] = random.randint(0, 10)
                save_scores(person_scores)
            print(f"[+] {name} eklendi. Skor: {person_scores[name]}")
            break

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()