import kagglehub
abdelghaniaaba_wildfire_prediction_dataset_path = kagglehub.dataset_download('abdelghaniaaba/wildfire-prediction-dataset')

print('Data source import complete.')
import numpy as np 
import pandas as pd 
import os
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random

def load_limited_images_flatten(dir_path, percent=0.1):
    x = []
    y = []
    for direct in os.listdir(dir_path):
        print(f"Loading dataset {dir_path} {direct}")
        class_dir = os.path.join(dir_path, direct)
        all_files = os.listdir(class_dir)
        sample_size = max(1, int(len(all_files) * percent))
        sampled_files = random.sample(all_files, sample_size)
        for filename in sampled_files:
            img_path = os.path.join(class_dir, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (32,32))
            img = img / 255.0
            x.append(img.flatten())
            y.append(direct)
    return x, y

# === Carregando imagens de treino (apenas 10%) ===
dir = '/kaggle/input/wildfire-prediction-dataset/train'
x, y = load_limited_images_flatten(dir, percent=0.1)

# === Carregando imagens de validação (apenas 10%) ===
dir_val = '/kaggle/input/wildfire-prediction-dataset/valid'
x_val, y_val = load_limited_images_flatten(dir_val, percent=0.1)

# === Pré-processamento dos rótulos ===
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_val_encoded = le.transform(y_val)

# === Treinamento do modelo Random Forest ===
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(x, y_encoded)

# === Avaliação ===
y_pred_val = clf.predict(x_val)

print("Accuracy:", accuracy_score(y_val_encoded, y_pred_val))
print("Classification Report:\n", classification_report(y_val_encoded, y_pred_val, target_names=le.classes_))
print("Confusion Matrix:\n", confusion_matrix(y_val_encoded, y_pred_val))

# === Teste com imagem única ===
test_img_path = "/kaggle/input/wildfire-prediction-dataset/test/wildfire/-59.03238,51.85132.jpg"
test_image = cv2.imread(test_img_path)
test_image = cv2.resize(test_image, (32,32))
test_image = test_image / 255.0
test_flatten = test_image.flatten().reshape(1, -1)

prediction = clf.predict(test_flatten)
predicted_class = le.inverse_transform(prediction)[0]
print("Classe prevista:", predicted_class)

# === Visualizar imagem testada ===
imgplot = plt.imshow(cv2.cvtColor(cv2.imread(test_img_path), cv2.COLOR_BGR2RGB))
plt.title(f"Classe prevista: {predicted_class}")
plt.axis('off')
plt.show()

# === Análise de erros: mostrar imagens mal classificadas ===
misclassified_idx = [i for i, (true, pred) in enumerate(zip(y_val_encoded, y_pred_val)) if true != pred]
print(f"Total de imagens mal classificadas: {len(misclassified_idx)}")

for idx in misclassified_idx[:5]:  # Mostra até 5 exemplos
    img_flat = x_val[idx]
    img = img_flat.reshape(32, 32, 3)
    true_label = le.inverse_transform([y_val_encoded[idx]])[0]
    pred_label = le.inverse_transform([y_pred_val[idx]])[0]
    plt.imshow(img)
    plt.title(f"Verdadeiro: {true_label} | Previsto: {pred_label}")
    plt.axis('off')
    plt.show()
