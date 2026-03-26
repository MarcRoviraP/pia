import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import os


PATH = os.path.dirname(__file__)
# dataframe = pd.read_csv(os.path.join(PATH,"insurance_preprocesed.txt"))
dataframe = pd.read_csv(os.path.join(PATH,"houses_dataset_final.csv"))


# Conseguir tota la data
print(dataframe.head())
X = dataframe.copy().drop(columns=["price"])


# Resultats de la data


Y = dataframe.copy()[["price"]]

# Pasar a torch
X_tensor = torch.tensor(X.values,dtype=torch.float32)
Y_tensor = torch.tensor(Y.values,dtype=torch.float32)
print(X_tensor.shape)
print(Y_tensor.shape)

trainInputs, valInputs, trainTargets, valTargets = train_test_split(X_tensor,Y_tensor,train_size=0.8,random_state=42)

trainDataset = TensorDataset(trainInputs,trainTargets)
validDataset = TensorDataset(valInputs,valTargets)

trainLoader = DataLoader(trainDataset,batch_size=64,shuffle=True)
validLoader = DataLoader(validDataset,batch_size=64,shuffle=True)

class RegresionNN(nn.Module):
    def __init__(self, inputDim,neuronas):
        super(RegresionNN,self).__init__()
        

        self.network = nn.Sequential(
            nn.Linear(inputDim,neuronas),
            nn.ReLU(),
            nn.Linear(neuronas,neuronas//2),
            nn.ReLU(),
            nn.Linear(neuronas//2,1),
            
        )

        
    def forward(self,x):
        return self.network(x)
    
neuronas = 128
model = RegresionNN(inputDim=X.shape[1],neuronas=neuronas)

criterion = nn.MSELoss()
# lr = Margen de error
optimizer = optim.Adam(model.parameters(),lr=0.001)
    
trainLosses = []
valLosses = []
trainAccuaracy = []
valAccuaracy = []

# Umbral de precision de la regresión
threshold = 0.1

# numero de epcocas que entrena el modelo
epochs = 20
activar = "RelU"
with open(f"{activar}_epocas{epochs}_capas{neuronas}.txt", "w", encoding="utf-8") as f:

    for epoch in range(epochs):
        model.train()
        perdidaEntrenamiento = 0
        precisionEntrenamiento = 0

        for entrada, objetivos in trainLoader:

            # Limplia los gradientes
            optimizer.zero_grad()
            # Entrena el modelo
            salidaEntrenamiento = model(entrada)
            # Calcula la perdida de la salida del entrenamiento, con los objetivos 
            perdida = criterion(salidaEntrenamiento, objetivos)
            perdida.backward()
            optimizer.step()

            # Cálculo de precisión en regresión
            precisionBatch = ((torch.abs(salidaEntrenamiento - objetivos) < threshold).float().mean()).item()
            perdidaEntrenamiento += perdida.item()
            precisionEntrenamiento += precisionBatch

        # Promediar la pérdida y precisión en el epoch
        perdidaEntrenamiento /= len(trainLoader)
        precisionEntrenamiento /= len(trainLoader)

        # Evaluación en validación
        model.eval()
        mediaDePerdidaEpoca = 0
        valPrecision = 0
        with torch.no_grad():
            for entrada, objetivos in validLoader:
                validOutputs = model(entrada)
                validLoss = criterion(validOutputs, objetivos)
                precisionBatch = ((torch.abs(validOutputs - objetivos) < threshold).float().mean()).item()
                mediaDePerdidaEpoca += validLoss.item()
                valPrecision += precisionBatch

        # Promediar la pérdida y precisión en validación
        mediaDePerdidaEpoca /= len(validLoader)
        valPrecision /= len(validLoader)

        # Almacenar métricas
        trainLosses.append(perdidaEntrenamiento)
        valLosses.append(mediaDePerdidaEpoca)
        trainAccuaracy.append(precisionEntrenamiento)
        valAccuaracy.append(valPrecision)


        f.write(f"Época [{epoch + 1}/{epochs}] - "
                f"Perdida entrenamiento: {perdidaEntrenamiento:.4f}, Precision: {precisionEntrenamiento:.4f} - "
                f"Perdida validador: {mediaDePerdidaEpoca:.4f}, Precision validador: {valPrecision:.4f}\n")
        print(f"Época [{epoch + 1}/{epochs}] - "
                f"Perdida entrenamiento: {perdidaEntrenamiento:.4f}, Precision: {precisionEntrenamiento:.4f} - "
                f"Perdida validador: {mediaDePerdidaEpoca:.4f}, Precision validador: {valPrecision:.4f}")
    

       
        