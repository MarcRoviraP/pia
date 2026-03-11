import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
import os

PATH = os.path.dirname(__file__)
dataframe = pd.read_csv(os.path.join(PATH,"insurance_preprocesed.txt"))

# Conseguir tota la data
X = dataframe
# print(dataframe.shape)

# Resultats de la data
Y = dataframe.iloc[:,-2:]

# Pasar a torch
X_tensor = torch.tensor(X,dtype=torch.float32)
Y_tensor = torch.tensor(Y,dtype=torch.float32)

trainInputs, valInputs, trainTargets, valTargets = train_test_split(X_tensor,Y_tensor,train_size=0.6,random_state=42)

trainDataset = TensorDataset(trainInputs,trainTargets)
validDataset = TensorDataset(valInputs,valInputs)

trainLoader = DataLoader(trainDataset,batch_size=32,shuffle=True)
validLoader = DataLoader(validDataset,batch_size=32,shuffle=True)

class RegresionNN(nn.Module):
    def __init__(self, inputDim):
        super(RegresionNN,self).__init__()
        
        self.hidden = nn.Linear(inputDim,16)
        self.output = nn.Linear(16,1)
        self.activation = nn.ReLU()
        
        
    def forawrd(self,x):
        x = self.activation(self.hidden(x))
        x = self.output(x)
        return x
    
    