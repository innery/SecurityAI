import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Veri yolu (senin verdiğin dizine göre)
DATASET_DIR = "/Users/aylac/SecurityAI/data/dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")

# Görsel boyutu ve batch size
IMG_SIZE = (48, 48)
BATCH_SIZE = 64

# Veri artırma
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Veri yükleme
train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=BATCH_SIZE
)

test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=BATCH_SIZE
)

# Yeni ve güçlü CNN modeli
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(48, 48, 1)),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.4),
    Dense(7, activation='softmax')  # 7 duygu sınıfı
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Eğitim kontrolü
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint("better_model.h5", monitor='val_accuracy', save_best_only=True)

# Eğitim başlasın
history = model.fit(
    train_data,
    epochs=55,
    validation_data=test_data,
    callbacks=[early_stop, checkpoint]
)

# Eğitim sonrası final modeli kaydet
model.save("model_2.h5")
print("✅ Yeni model başarıyla eğitildi ve kaydedildi.")

import matplotlib.pyplot as plt

def egitim_grafigi(history, title="Model Eğitim Grafiği"):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    # Doğruluk Grafiği
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-o', label='Eğitim Doğruluğu')
    plt.plot(epochs, val_acc, 'r-o', label='Doğrulama Doğruluğu')
    plt.title(f'{title} - Doğruluk')
    plt.xlabel('Epoch')
    plt.ylabel('Doğruluk')
    plt.legend()

    # Kayıp Grafiği
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-o', label='Eğitim Kaybı')
    plt.plot(epochs, val_loss, 'r-o', label='Doğrulama Kaybı')
    plt.title(f'{title} - Kayıp')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Grafiği çiz
egitim_grafigi(history, "Model 2 - CNN + BatchNormalization")