import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.datasets import fetch_openml
import torch
import torch.nn as nn

# Importar dataset
mnist = fetch_openml("mnist_784", version=1)

# Crear clase clasificacion binaria
class BinaryClasificationNN(nn.Module):
    def __init__(self, inputDim):
        super(BinaryClasificationNN,self).__init__()
        
        # Capa oculta con entra y 150 neuronas
        self.hidden = nn.Linear(inputDim,150)
        # Capa salida 
        self.output = nn.Linear(150,1)
        self.activation = nn.ReLU()
        
    def forward(self,x):
        x = self.activation(self.hidden(x))
        x = self.output(x)
        return x
    
    def proccesarData(self,X,y):
        y_3 = (y == '3')
        
        
        # Dividir datos entre entrenamiento y test
        self.XTrain, self.XTest, self.yTrain, self.yTest = train_test_split(X,y_3,test_size=0.2,random_state=1)
        # Dividir datos de entrenamiento entre entrenamiento y validadcion
        self.XTrain, self.XVal, self.yTrain, self.yVal = train_test_split(self.XTrain,self.yTrain,test_size=0.25,random_state=1)
        
        # Procesa los datos sucios para que sean numeros en caso contrario coerce los pasa a null
        self.yTrain = pd.to_numeric(self.yTrain,errors="coerce")
        self.yVal = pd.to_numeric(self.yVal,errors="coerce")
        
        
        # unsqueeze transforma el vector en una matriz para tener las mismas dimensiones y poder trabajar con los tensores
        self.XTrainTensor = torch.tensor(self.XTrain.values,dtype=torch.float32)
        self.yTrainTensor = torch.tensor(self.yTrain.values,dtype=torch.long).unsqueeze(1)
        
        self.XValTensor = torch.tensor(self.XVal.values,dtype=torch.float32)
        self.yValTensor = torch.tensor(self.yVal.values,dtype=torch.long).unsqueeze(1)
        
        
        # Crear DataLoader (opcional, pero útil si se usa en batches)
        self.trainDataset = TensorDataset(self.XTrainTensor, self.yTrainTensor)
        self.valDataset = TensorDataset(self.XValTensor, self.yValTensor)

        self.trainLoader = DataLoader(self.trainDataset, batch_size=8, shuffle=True)
        self.valLoader = DataLoader(self.valDataset, batch_size=8, shuffle=False)
        
    def entrenarModelo(self,epochs,criterion,optmizer):
        
        for epoch in range(epochs):
            self.train()
            perdidaPorEpoca = 0
            precisionPorEpoca = 0
            
            for batchInputs, batchObjetivos in self.trainLoader:
                # Reincia gradientes del optimizador
                optmizer.zero_grad()
                salidaEntrenamiento = self(batchInputs)
                perdidaEntrenamiento = criterion(salidaEntrenamiento,batchObjetivos.float())
                perdidaEntrenamiento.backward()
                optmizer.step()
                
                # Calculo precision del modelo
                
                perdidaPorEpoca += perdidaEntrenamiento.item()
            # Promedio precion epoca
            
            perdidaPorEpoca /= len(self.trainLoader)
            
            # Evaluacion en validacion
            
            self.eval()
            perdidaValidEpoca = 0
            perdidaPrecisionEpoca = 0
            with torch.no_grad():
                for batchInputs, objetivoBatch in self.valLoader:
                    salidaValid = self(batchInputs)
                    perdidaValid = criterion(salidaValid,objetivoBatch.float())
                    perdidaValidEpoca += perdidaValid.item()
                    
            # Promediar la precision de la validacion
            perdidaValidEpoca /= len(self.valLoader)
            print(f"Época [{epoch + 1}/{epochs}] - "
                    f"Train BCEWithLogitsLoss: {perdidaPorEpoca:.4f} - "
                    f"Val BCEWithLogitsLoss: {perdidaValidEpoca:.4f}")
            
        # Método para hacer una predicción
    def predict(self, X):
        self.eval()
        with torch.no_grad():
            yPred = self(X)  # Obtener logits
            yPred = torch.sigmoid(yPred)  # Convertir logits a probabilidades
            yPredEtiquetas = (yPred >= 0.5).float()  # Convertir a 0 o 1
            return yPredEtiquetas
        
X = mnist["data"]
y = mnist["target"]

model = BinaryClasificationNN(inputDim=X.shape[1])

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(),lr=0.0001)
epochs = 10

model.proccesarData(X,y)
model.entrenarModelo(epochs=epochs,criterion=criterion,optmizer=optimizer)
yPred = model.predict(model.XValTensor)