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

# === Carregando imagens de treino ===
dir = '/kaggle/input/wildfire-prediction-dataset/train'
x = []
y = []

for direct in os.listdir(dir):
    print("Loading dataset training {}".format(direct))
    for filename in os.listdir(os.path.join(dir,direct)):
        img_path = os.path.join(dir,direct,filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (32,32))
        img = img / 255.0
        x.append(img.flatten())  # <-- transforma a imagem 32x32x3 em um vetor 1D
        y.append(direct)

# === Carregando imagens de validação ===
dir_val = '/kaggle/input/wildfire-prediction-dataset/valid'
x_val = []
y_val = []

for direct in os.listdir(dir_val):
    print("Loading dataset validation {}".format(direct))
    for filename in os.listdir(os.path.join(dir_val,direct)):
        img_path = os.path.join(dir_val,direct,filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (32,32))
        img = img / 255.0
        x_val.append(img.flatten())
        y_val.append(direct)

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
