# =========================================
# Projeto: Detecção de Queimadas com Aprendizado de Máquina
# Algoritmo: Rede Neural Convolucional (CNN)
# =========================================

import kagglehub
wildfire_prediction_dataset_path = kagglehub.dataset_download('abdelghaniaaba/wildfire-prediction-dataset')

print('Data source import complete.')
import numpy as np
import pandas as pd
import os
import cv2
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve, f1_score, accuracy_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, BatchNormalization

# Função para carregar imagens, rótulos e caminhos, normalizando e redimensionando
def load_limited_images(dir_path, percent=0.1, return_paths=False):
    x = []
    y = []
    paths = []
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
            x.append(img)
            y.append(direct)
            if return_paths:
                paths.append(img_path)
    if return_paths:
        return x, y, paths
    return x, y

# === Carregamento dos dados ===
# Utiliza 10% das imagens de cada classe para reduzir o dataset
train_dir = '/kaggle/input/wildfire-prediction-dataset/train'
val_dir = '/kaggle/input/wildfire-prediction-dataset/valid'
test_dir = '/kaggle/input/wildfire-prediction-dataset/test'

x_train, y_train, train_paths = load_limited_images(train_dir, percent=0.1, return_paths=True)
x_val, y_val, val_paths = load_limited_images(val_dir, percent=0.1, return_paths=True)
x_test, y_test, test_paths = load_limited_images(test_dir, percent=0.1, return_paths=True)

X_train = np.array(x_train)
Y_train = np.array(y_train)
paths_train = np.array(train_paths)
X_val = np.array(x_val)
Y_val = np.array(y_val)
paths_val = np.array(val_paths)
X_test = np.array(x_test)
Y_test = np.array(y_test)
paths_test = np.array(test_paths)

# === Codificação dos rótulos ===
le = LabelEncoder()
Y_train_enc = le.fit_transform(Y_train)
Y_val_enc = le.transform(Y_val)
Y_test_enc = le.transform(Y_test)
Y_train_cat = to_categorical(Y_train_enc)
Y_val_cat = to_categorical(Y_val_enc)
Y_test_cat = to_categorical(Y_test_enc)

# === Definição e treinamento do modelo CNN ===
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Treinamento com validação
history = model.fit(X_train, Y_train_cat, validation_data=(X_val, Y_val_cat), batch_size=32, epochs=10)

# === Gráficos de desempenho ===
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Acurácia do Modelo')
plt.ylabel('Acurácia')
plt.xlabel('Época')
plt.legend(['Treino', 'Validação'], loc='lower right')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss do Modelo')
plt.ylabel('Loss')
plt.xlabel('Época')
plt.legend(['Treino', 'Validação'], loc='lower right')
plt.show()

# === Avaliação quantitativa no conjunto de teste ===
Y_pred_test = model.predict(X_test)
Y_pred_test_classes = np.argmax(Y_pred_test, axis=1)
print("\n=== Avaliação no Conjunto de Teste ===")
print("Classification Report:\n", classification_report(Y_test_enc, Y_pred_test_classes, target_names=le.classes_))
fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_predictions(Y_test_enc, Y_pred_test_classes, display_labels=le.classes_, cmap=plt.cm.Blues, ax=ax)
plt.title('Matriz de Confusão - Teste')
plt.show()
if Y_pred_test.shape[1] == 2:
    auc_test = roc_auc_score(Y_test_enc, Y_pred_test[:,1])
    print("AUC (Teste):", auc_test)
    fpr, tpr, _ = roc_curve(Y_test_enc, Y_pred_test[:,1])
    plt.plot(fpr, tpr, label=f'Teste (AUC = {auc_test:.2f})')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curva ROC - Teste')
    plt.legend()
    plt.show()

# === Avaliação quantitativa no conjunto de treinamento ===
Y_pred_train = model.predict(X_train)
Y_pred_train_classes = np.argmax(Y_pred_train, axis=1)
print("\n=== Avaliação no Conjunto de Treinamento ===")
print("Classification Report:\n", classification_report(Y_train_enc, Y_pred_train_classes, target_names=le.classes_))
fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_predictions(Y_train_enc, Y_pred_train_classes, display_labels=le.classes_, cmap=plt.cm.Blues, ax=ax)
plt.title('Matriz de Confusão - Treinamento')
plt.show()
if Y_pred_train.shape[1] == 2:
    auc_train = roc_auc_score(Y_train_enc, Y_pred_train[:,1])
    print("AUC (Treinamento):", auc_train)
    fpr, tpr, _ = roc_curve(Y_train_enc, Y_pred_train[:,1])
    plt.plot(fpr, tpr, label=f'Treinamento (AUC = {auc_train:.2f})')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curva ROC - Treinamento')
    plt.legend()
    plt.show()

# === Bloco de comparação dos resultados ===
from sklearn.metrics import f1_score
import numpy as np

# Calcula métricas
f1_train = f1_score(Y_train_enc, Y_pred_train_classes)
# Corrigido: calcular predição para validação
Y_pred_val = model.predict(X_val)
y_val_pred_classes = np.argmax(Y_pred_val, axis=1)
f1_val = f1_score(Y_val_enc, y_val_pred_classes)
f1_test = f1_score(Y_test_enc, Y_pred_test_classes)
acc_train = history.history['accuracy'][-1]
acc_val = history.history['val_accuracy'][-1]
acc_test = accuracy_score(Y_test_enc, Y_pred_test_classes)
auc_train_val = auc_train if Y_pred_train.shape[1]==2 else np.nan
auc_val = roc_auc_score(Y_val_enc, Y_pred_val[:,1]) if Y_pred_val.shape[1]==2 else np.nan
auc_test_val = auc_test if Y_pred_test.shape[1]==2 else np.nan

labels = ['Treinamento', 'Validação', 'Teste']
accs = [acc_train, acc_val, acc_test]
f1s = [f1_train, f1_val, f1_test]
aucs = [auc_train_val, auc_val, auc_test_val]

x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(8,5))
rects1 = ax.bar(x - width, accs, width, label='Acurácia')
rects2 = ax.bar(x, f1s, width, label='F1-score')
rects3 = ax.bar(x + width, aucs, width, label='AUC')

ax.set_ylabel('Pontuação')
ax.set_title('Comparação das principais métricas por conjunto')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

for rect in rects1 + rects2 + rects3:
    height = rect.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

plt.ylim(0, 1.05)
plt.show()

# === Exemplos visuais de classificação ===
print("\nExemplos de classificação no conjunto de teste:")
num_examples = 5
for i in range(num_examples):
    idx = np.random.randint(0, len(X_test))
    img = cv2.imread(paths_test[idx])
    true_label = le.inverse_transform([Y_test_enc[idx]])[0]
    pred_label = le.inverse_transform([Y_pred_test_classes[idx]])[0]
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Real: {true_label} | Previsto: {pred_label} | {'ACERTOU' if true_label == pred_label else 'ERROU'}")
    plt.axis('off')
    plt.show()