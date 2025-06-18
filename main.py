import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import os
import json
import random
import numpy as np
from keras.models import load_model
from keras_facenet import FaceNet
from keras.preprocessing.image import img_to_array
from sklearn.metrics.pairwise import cosine_similarity

# ---- AYARLAR ----
MODEL_PATHS = {
    "Model 1": "/Users/aylac/SecurityAI/model/model_1.h5",
    "Model 2": "/Users/aylac/SecurityAI/model/model_2.h5"
}
EMBEDDINGS_DIR = "embeddings"
FACES_DIR = "faces"
SCORES_PATH = "/Users/aylac/SecurityAI/data/score/scores.json"
CAMERA_INDEX = 1  

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
emotion_weights = {
    "angry": 85,
    "disgust": 55,
    "fear": 75,
    "happy": 15,
    "sad": 40,
    "surprise": 25,
    "neutral": 30
}

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(FACES_DIR, exist_ok=True)

# ---- FONKSİYONLAR ----
def load_embeddings():
    embeddings = {}
    for file in os.listdir(EMBEDDINGS_DIR):
        if file.endswith(".npy"):
            name = os.path.splitext(file)[0]
            embeddings[name] = np.load(os.path.join(EMBEDDINGS_DIR, file))
    return embeddings

def load_scores():
    if os.path.exists(SCORES_PATH):
        try:
            with open(SCORES_PATH, 'r') as f:
                data = f.read().strip()
                if data:
                    return json.loads(data)
        except:
            pass
    return {}

def save_scores(scores):
    with open(SCORES_PATH, 'w') as f:
        json.dump(scores, f)

def predict_emotion(face, model):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    array = img_to_array(resized) / 255.0
    array = np.expand_dims(array, axis=0)
    array = np.expand_dims(array, axis=-1)
    preds = model.predict(array)[0]
    return emotion_labels[np.argmax(preds)]

def extract_face(frame, box):
    x, y, w, h = box
    face = frame[y:y+h, x:x+w]
    return cv2.resize(face, (160, 160))

# ---- GUI SINIFI ----
class SecurityAIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SecurityAI - Model Seçimi")
        self.root.geometry("1600x800")

        frame = ttk.Frame(root)
        frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        self.selected_model_name = tk.StringVar(value=list(MODEL_PATHS.keys())[0])

        style = ttk.Style()
        style.configure("Custom.TRadiobutton", font=("Arial", 16))

        ttk.Label(frame, text="Lütfen bir model seçin:", font=("Arial", 20)).pack(pady=30)
        for name in MODEL_PATHS:
            ttk.Radiobutton(frame, text=name, variable=self.selected_model_name, value=name, style="Custom.TRadiobutton").pack(pady=10)

        ttk.Button(frame, text="Başlat", command=self.start_app).pack(pady=30)

    def start_app(self):
        selected_model_path = MODEL_PATHS[self.selected_model_name.get()]
        self.root.destroy()
        MainInterface(selected_model_path)

# ---- ANA UYGULAMA SINIFI ----
class MainInterface:
    def __init__(self, model_path):
        self.root = tk.Tk()
        self.root.title("SecurityAI - Canlı Takip")
        self.root.geometry("1600x800")

        self.embedder = FaceNet()
        self.emotion_model = load_model(model_path)
        self.embeddings = load_embeddings()
        self.scores = load_scores()

        # Sol panel
        left_panel = ttk.Frame(self.root)
        left_panel.grid(row=0, column=0, sticky="n", padx=20, pady=20)

        self.info_label = ttk.Label(left_panel, text="Kişi: ?\nDuygu: ?\nSkor: ?\nRisk: ?", font=("Arial", 14))
        self.info_label.pack(pady=10)

        self.warning_label = ttk.Label(left_panel, text="", foreground="red", font=("Arial", 16, "bold"))
        self.warning_label.pack(pady=10)

        self.name_entry = ttk.Entry(left_panel, font=("Arial", 12))
        self.name_entry.pack(pady=10)

        self.add_button = ttk.Button(left_panel, text="Yeni Kişi Ekle", command=self.add_person)
        self.add_button.pack(pady=10)

        # Kamera
        self.video_frame = tk.Label(self.root)
        self.video_frame.grid(row=0, column=1, padx=10, pady=10)

        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        self.update_video()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.mainloop()

    def update_video(self):
        ret, frame = self.cap.read()
        frame = cv2.flip(frame, 1)  # Ayna efekti: ön kamera gibi gösterim

        name = "Bilinmiyor"
        emotion = "?"
        score = 0
        risk = 0
        warning = ""

        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face = extract_face(frame, (x, y, w, h))
                name = self.recognize(face)
                emotion = predict_emotion(face, self.emotion_model)

                if name != "Bilinmiyor":
                    if name not in self.scores:
                        self.scores[name] = random.randint(0, 10)
                        save_scores(self.scores)
                    score = self.scores[name]
                    risk = score * emotion_weights.get(emotion, 50)
                    if risk > 500:
                        warning = "⚠️ TEHLİKE!"

                label = f"{name} ({emotion})"
                color = (0, 0, 255) if risk > 500 else (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                break

            self.info_label.config(text=f"Kişi: {name}\nDuygu: {emotion}\nSkor: {score}\nRisk: {risk}")
            self.warning_label.config(text=warning)

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_frame.imgtk = imgtk
            self.video_frame.configure(image=imgtk)

        self.root.after(20, self.update_video)

    def recognize(self, face_img):
        embedding = self.embedder.embeddings([face_img])[0]
        for name, known_emb in self.embeddings.items():
            sim = cosine_similarity([embedding], [known_emb])[0][0]
            if sim > 0.6:
                return name
        return "Bilinmiyor"

    def add_person(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror("Hata", "İsim boş olamaz!")
            return

        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Hata", "Kamera görüntüsü alınamadı.")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            messagebox.showinfo("Bilgi", "Yüz algılanamadı.")
            return

        (x, y, w, h) = faces[0]
        face_img = extract_face(frame, (x, y, w, h))
        cv2.imwrite(os.path.join(FACES_DIR, f"{name}.jpg"), face_img)
        embedding = self.embedder.embeddings([face_img])[0]
        np.save(os.path.join(EMBEDDINGS_DIR, f"{name}.npy"), embedding)
        self.embeddings[name] = embedding
        if name not in self.scores:
            self.scores[name] = random.randint(0, 10)
            save_scores(self.scores)
        messagebox.showinfo("Başarılı", f"{name} başarıyla eklendi.")
        self.name_entry.delete(0, tk.END)

    def close(self):
        self.cap.release()
        self.root.destroy()

# ---- BAŞLAT ----
if __name__ == "__main__":
    root = tk.Tk()
    app = SecurityAIApp(root)
    root.mainloop()