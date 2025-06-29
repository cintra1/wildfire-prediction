# =========================================
# Projeto: Detecção de Queimadas com Aprendizado de Máquina
# Algoritmo: Random Forest com Extração de Atributos (Histograma de Cores)
# =========================================

import kagglehub
wildfire_prediction_dataset_path = kagglehub.dataset_download('abdelghaniaaba/wildfire-prediction-dataset')

print('Data source import complete.')
import numpy as np 
import pandas as pd 
import os
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random

# Função para extrair histograma de cores de uma imagem, retorna um vetor 1D concatenando histogramas dos 3 canais RGB
def extract_color_histogram(img, bins=(8, 8, 8)):
    hist = cv2.calcHist([img], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# Função para carregar imagens e extrair atributos (histograma de cores)
def load_images_with_features(dir_path, percent=0.1, return_paths=False):
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
            features = extract_color_histogram(img)
            x.append(features)
            y.append(direct)
            if return_paths:
                paths.append(img_path)
    if return_paths:
        return np.array(x), np.array(y), np.array(paths)
    return np.array(x), np.array(y)

# === Carregando imagens de treino, validação e teste (apenas 10%) ===
dir_train = '/kaggle/input/wildfire-prediction-dataset/train'
dir_val = '/kaggle/input/wildfire-prediction-dataset/valid'
dir_test = '/kaggle/input/wildfire-prediction-dataset/test'

x_train, y_train, train_paths = load_images_with_features(dir_train, percent=0.1, return_paths=True)
x_val, y_val, val_paths = load_images_with_features(dir_val, percent=0.1, return_paths=True)
x_test, y_test, test_paths = load_images_with_features(dir_test, percent=0.1, return_paths=True)

X_train = np.array(x_train)
y_train = np.array(y_train)
paths_train = np.array(train_paths)
X_val = np.array(x_val)
y_val = np.array(y_val)
paths_val = np.array(val_paths)
X_test = np.array(x_test)
y_test = np.array(y_test)
paths_test = np.array(test_paths)

# === Pré-processamento dos rótulos ===
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc = le.transform(y_val)
y_test_enc = le.transform(y_test)

# === Treinamento do modelo Random Forest ===
clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=42
)
clf.fit(X_train, y_train_enc)

# === Gráficos de importância das features ===
importances = clf.feature_importances_
plt.figure(figsize=(8,3))
plt.plot(importances)
plt.title('Random Forest - Importância das Features (Histograma de Cores)')
plt.xlabel('Índice da Feature')
plt.ylabel('Importância')
plt.tight_layout()
plt.show()

# === Avaliação: Treinamento ===
y_pred_train = clf.predict(X_train)
y_pred_train_proba = clf.predict_proba(X_train)[:,1]
print("\n=== Avaliação no Conjunto de Treinamento ===")
print("Classification Report:\n", classification_report(y_train_enc, y_pred_train, target_names=le.classes_))
fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_predictions(y_train_enc, y_pred_train, display_labels=le.classes_, cmap=plt.cm.Blues, ax=ax)
plt.title('Random Forest - Matriz de Confusão - Treinamento')
plt.show()
if len(le.classes_) == 2:
    auc_train = roc_auc_score(y_train_enc, y_pred_train_proba)
    print("AUC (Treinamento):", auc_train)
    fpr, tpr, _ = roc_curve(y_train_enc, y_pred_train_proba)
    plt.plot(fpr, tpr, label=f'Treinamento (AUC = {auc_train:.2f})')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Random Forest - Curva ROC - Treinamento')
    plt.legend()
    plt.show()

# === Avaliação: Validação ===
y_pred_val = clf.predict(X_val)
y_pred_val_proba = clf.predict_proba(X_val)[:,1]
print("\n=== Avaliação no Conjunto de Validação ===")
print("Classification Report:\n", classification_report(y_val_enc, y_pred_val, target_names=le.classes_))
fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_predictions(y_val_enc, y_pred_val, display_labels=le.classes_, cmap=plt.cm.Blues, ax=ax)
plt.title('Random Forest - Matriz de Confusão - Validação')
plt.show()
if len(le.classes_) == 2:
    auc_val = roc_auc_score(y_val_enc, y_pred_val_proba)
    print("AUC (Validação):", auc_val)
    fpr, tpr, _ = roc_curve(y_val_enc, y_pred_val_proba)
    plt.plot(fpr, tpr, label=f'Validação (AUC = {auc_val:.2f})')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Random Forest - Curva ROC - Validação')
    plt.legend()
    plt.show()

# === Avaliação: Teste ===
y_pred_test = clf.predict(X_test)
y_pred_test_proba = clf.predict_proba(X_test)[:,1]
print("\n=== Avaliação no Conjunto de Teste ===")
print("Classification Report:\n", classification_report(y_test_enc, y_pred_test, target_names=le.classes_))
g, ax = plt.subplots()
ConfusionMatrixDisplay.from_predictions(y_test_enc, y_pred_test, display_labels=le.classes_, cmap=plt.cm.Blues, ax=ax)
plt.title('Random Forest - Matriz de Confusão - Teste')
plt.show()
if len(le.classes_) == 2:
    auc_test = roc_auc_score(y_test_enc, y_pred_test_proba)
    print("AUC (Teste):", auc_test)
    fpr, tpr, _ = roc_curve(y_test_enc, y_pred_test_proba)
    plt.plot(fpr, tpr, label=f'Teste (AUC = {auc_test:.2f})')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Random Forest - Curva ROC - Teste')
    plt.legend()
    plt.show()

# === Bloco de comparação dos resultados ===
# Gráfico comparativo das principais métricas
from sklearn.metrics import f1_score
import numpy as np

# Calcula métricas
f1_train = f1_score(y_train_enc, y_pred_train)
f1_val = f1_score(y_val_enc, y_pred_val)
f1_test = f1_score(y_test_enc, y_pred_test)
acc_train = accuracy_score(y_train_enc, y_pred_train)
acc_val = accuracy_score(y_val_enc, y_pred_val)
acc_test = accuracy_score(y_test_enc, y_pred_test)
auc_train_val = auc_train if len(le.classes_)==2 else np.nan
auc_val_val = auc_val if len(le.classes_)==2 else np.nan
auc_test_val = auc_test if len(le.classes_)==2 else np.nan

labels = ['Treinamento', 'Validação', 'Teste']
accs = [acc_train, acc_val, acc_test]
f1s = [f1_train, f1_val, f1_test]
aucs = [auc_train_val, auc_val_val, auc_test_val]

x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(8,5))
rects1 = ax.bar(x - width, accs, width, label='Acurácia')
rects2 = ax.bar(x, f1s, width, label='F1-score')
rects3 = ax.bar(x + width, aucs, width, label='AUC')

ax.set_ylabel('Pontuação')
ax.set_title('Random Forest - Comparação das principais métricas por conjunto')
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

# === Exemplos de classificação da rede neural (Random Forest) ===
# Mostra algumas imagens do conjunto de teste, o rótulo real e o previsto
print("\nRandom Forest - Exemplos de classificação no conjunto de teste:")
num_examples = 5
for i in range(num_examples):
    idx = np.random.randint(0, len(X_test))
    true_label = le.inverse_transform([y_test_enc[idx]])[0]
    pred_label = le.inverse_transform([y_pred_test[idx]])[0]
    img_path = paths_test[idx]
    img = cv2.imread(img_path)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Real: {true_label} | Previsto: {pred_label} | {'ACERTOU' if true_label == pred_label else 'ERROU'}")
    plt.axis('off')
    plt.show()
