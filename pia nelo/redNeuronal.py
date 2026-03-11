import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

from sklearn.datasets import fetch_california_housing

# Cargar el dataset de California Housing donde alamcena datos como ingreso medio, edad de la casa, número de habitaciones, ubicación
data = fetch_california_housing()
X = data.data  # Features (variables de entrada)
y = data.target.reshape(-1, 1)  # Valores objetivo

# Convertir a tensores de PyTorch
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Dividir en conjunto de entrenamiento y validación.El 0.2 se reserva para validar el modelo con datos no usados antes
train_inputs, val_inputs, train_targets, val_targets = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Crear DataLoader (opcional, pero útil si se usa en batches)
train_dataset = TensorDataset(train_inputs, train_targets)
val_dataset = TensorDataset(val_inputs, val_targets)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Definir la red neuronal para regresión
class RegressionNN(nn.Module):
    # input_dim = Numero de variables de entrada
    def __init__(self, input_dim):
        super(RegressionNN, self).__init__()

        self.hidden = nn.Linear(input_dim, 16)  # Crea una capa oculta con 16 neuronas
        self.output = nn.Linear(16, 1)  # 16 neuronas con 1 salida  que es el precio de la casa
        self.activation = nn.ReLU()  # Función de activación

    def forward(self, x):
        x = self.activation(self.hidden(x))
        x = self.output(x)
        return x

# Crear modelo basado en la cantidad de features en este caso 1, En california data housing son 8 features
model = RegressionNN(input_dim=X.shape[1])

criterion = nn.MSELoss()
# Función de pérdida y optimizador Adam es muy usado porque converge rapido y ajusta automaticamente el learning rate
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Listas para almacenar métricas
# Perdida del entrenamiento
train_losses = []
# Perdidad de validación
val_losses = []
# Precisión de entrenamiento
train_accuracies = []
# Precisión de validadación
val_accuracies = []

# Definir umbral de precisión en regresión (±0.1)
threshold = 0.1

epochs = 20
# Ciclo de entrenamiento cada epoca es pasar por el dataset cada vez
for epoch in range(epochs):
    model.train()
    epoch_train_loss = 0
    epoch_train_accuracy = 0

    for batch_inputs, batch_targets in train_loader:
        optimizer.zero_grad()
        train_outputs = model(batch_inputs)
        train_loss = criterion(train_outputs, batch_targets)
        train_loss.backward()
        optimizer.step()

        # Cálculo de precisión en regresión
        batch_accuracy = ((torch.abs(train_outputs - batch_targets) < threshold).float().mean()).item()
        epoch_train_loss += train_loss.item()
        epoch_train_accuracy += batch_accuracy

    # Promediar la pérdida y precisión en el epoch
    epoch_train_loss /= len(train_loader)
    epoch_train_accuracy /= len(train_loader)

    # Evaluación en validación
    model.eval()
    epoch_val_loss = 0
    epoch_val_accuracy = 0
    with torch.no_grad():
        for batch_inputs, batch_targets in val_loader:
            val_outputs = model(batch_inputs)
            val_loss = criterion(val_outputs, batch_targets)
            batch_accuracy = ((torch.abs(val_outputs - batch_targets) < threshold).float().mean()).item()
            epoch_val_loss += val_loss.item()
            epoch_val_accuracy += batch_accuracy

    # Promediar la pérdida y precisión en validación
    epoch_val_loss /= len(val_loader)
    epoch_val_accuracy /= len(val_loader)

    # Almacenar métricas
    train_losses.append(epoch_train_loss)
    val_losses.append(epoch_val_loss)
    train_accuracies.append(epoch_train_accuracy)
    val_accuracies.append(epoch_val_accuracy)

    # Imprimir cada 50 épocas

    print(f"Época [{epoch + 1}/{epochs}] - "
            f"MSE: {epoch_train_loss:.4f}, Accuracy: {epoch_train_accuracy:.4f} - "
            f"Val MSE: {epoch_val_loss:.4f}, Val Accuracy: {epoch_val_accuracy:.4f}")