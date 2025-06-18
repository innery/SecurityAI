from keras.models import load_model

model_1 = load_model("/Users/aylac/SecurityAI/model/model_1.h5")
model_2 = load_model("/Users/aylac/SecurityAI/model/model_2.h5")

def model_ozeti_yazdir(model, baslik):
    print(f"\n=== {baslik} ===")
    model.summary()

model_ozeti_yazdir(model_1, "Model 1 - BatchNormsuz CNN")
model_ozeti_yazdir(model_2, "Model 2 - BatchNormlu CNN")

import matplotlib.pyplot as plt

def egitim_grafigi(history, title):
    plt.figure(figsize=(12, 4))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

    import matplotlib.pyplot as plt
from matplotlib.table import Table

def model_ozeti_gorsel(model, baslik="Model Özeti"):
    # Modelden tablo verilerini al
    layer_names = []
    output_shapes = []
    param_counts = []

    for layer in model.layers:
        layer_names.append(layer.name)
        try:
            output_shapes.append(str(layer.output_shape))
        except:
            output_shapes.append("N/A")
        param_counts.append(layer.count_params())

    # Görsel oluştur
    fig, ax = plt.subplots(figsize=(10, len(layer_names) * 0.5 + 2))
    ax.set_axis_off()
    ax.set_title(baslik, fontweight="bold", fontsize=14)

    # Tablo oluştur
    tab = Table(ax, bbox=[0, 0, 1, 1])

    # Başlıklar
    columns = ["Layer Name", "Output Shape", "Param #"]
    cell_text = [columns] + list(zip(layer_names, output_shapes, param_counts))

    n_rows = len(cell_text)
    n_cols = len(columns)

    width = 1.0 / n_cols
    height = 1.0 / n_rows

    for i, row in enumerate(cell_text):
        for j, cell in enumerate(row):
            tab.add_cell(i, j, width, height, text=str(cell), loc='center',
                         facecolor='#f2f2f2' if i == 0 else 'white')

    for j in range(n_cols):
        tab.add_cell(n_rows, j, width, height / 2, text='', loc='center', facecolor='white')

    ax.add_table(tab)
    plt.show()

model_ozeti_gorsel(model_1, "Model 1 - BatchNormsuz CNN")
model_ozeti_gorsel(model_2, "Model 2 - BatchNormlu CNN")