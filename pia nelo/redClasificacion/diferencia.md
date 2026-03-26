# 📊 Comparación: Red Neuronal Binaria vs Multiclase (PyTorch)

Este repositorio contiene dos implementaciones de redes neuronales usando PyTorch aplicadas al dataset MNIST:

* `classBinaria.py` → Clasificación binaria (¿es un 3 o no?)
* `classMulticlase.py` → Clasificación multiclase (dígitos del 0 al 9)

---

## 🧠 Diferencias Conceptuales

| Característica     | Clasificación Binaria     | Clasificación Multiclase          |
| ------------------ | ------------------------- | --------------------------------- |
| Objetivo           | Distinguir entre 2 clases | Distinguir entre múltiples clases |
| Ejemplo            | ¿Es un 3? (Sí / No)       | ¿Qué número es? (0–9)             |
| Salida del modelo  | 1 neurona                 | 10 neuronas                       |
| Activación final   | Sigmoid                   | Softmax                           |
| Función de pérdida | BCEWithLogitsLoss         | CrossEntropyLoss                  |

---

## ⚙️ Diferencias en la Arquitectura

### 🔹 Red Binaria

```python
self.hidden = nn.Linear(inputDim,150)
self.output = nn.Linear(150,1)
```

* Solo **1 salida**
* Representa la probabilidad de pertenecer a una clase

### 🔹 Red Multiclase

```python
self.hidden = nn.Linear(inputDim,150)
self.output = nn.Linear(150,10)
```

* **10 salidas** (una por cada dígito)
* Cada salida representa una clase

---

## 📊 Procesamiento de Datos

### Binaria

```python
y_3 = (y == '3')
```

* Convierte el problema en:

  * `1` → es un 3
  * `0` → no es un 3
* Se usa `.unsqueeze(1)` para ajustar dimensiones

### Multiclase

```python
y = pd.to_numeric(y)
```

* Se mantienen las etiquetas originales (0–9)
* No se modifica la estructura del problema

---

## 📉 Función de Pérdida

### Binaria

```python
criterion = nn.BCEWithLogitsLoss()
```

* Combina Sigmoid + Binary Cross Entropy
* Requiere etiquetas en formato float

### Multiclase

```python
criterion = nn.CrossEntropyLoss()
```

* Incluye Softmax internamente
* Usa etiquetas como enteros (clases)

---

## 🔄 Predicción

### Binaria

```python
yPred = torch.sigmoid(yPred)
yPredEtiquetas = (yPred >= 0.5).float()
```

* Umbral de decisión: **0.5**

### Multiclase

```python
yPred = F.softmax(yPred, dim=1)
yPredEtiquetas = torch.argmax(yPred, dim=1)
```

* Se elige la clase con mayor probabilidad

---

## 📈 Evaluación

### Binaria

* Matriz de confusión
* Curva ROC (AUC)

### Multiclase

* Matriz de confusión
* No usa ROC (no es directa en multiclase)

---

## ⚠️ Diferencias Clave en Código

| Aspecto          | Binaria           | Multiclase         |
| ---------------- | ----------------- | ------------------ |
| Output layer     | `Linear(150,1)`   | `Linear(150,10)`   |
| Loss             | BCEWithLogitsLoss | CrossEntropyLoss   |
| Activación final | Sigmoid manual    | Softmax en predict |
| Labels           | 0 / 1             | 0–9                |
| Shape de y       | `(N,1)`           | `(N,)`             |

---

## 🧩 Conclusión

* La red **binaria** simplifica el problema a una decisión (sí/no), útil para detección específica.
* La red **multiclase** es más general y permite clasificar múltiples categorías.
* La principal diferencia está en:

  * Número de salidas
  * Función de pérdida
  * Interpretación de resultados

---

## 🚀 Recomendación

* Usa **clasificación binaria** cuando te interesa detectar una clase concreta.
* Usa **clasificación multiclase** cuando necesitas distinguir entre varias categorías.

---
