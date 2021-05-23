import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset 


iris = load_iris()
X = iris['data']
y = iris['target']
names = iris['target_names']
feature_names = iris['feature_names']

df = pd.DataFrame(iris.data) 
df.columns = iris.feature_names 
df['label'] = iris.target


X = df.drop('label', axis=1).to_numpy() 
Y = df['label'].to_numpy().reshape((-1,1))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

class TensorData(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.FloatTensor(x_data)
        self.y_data = torch.LongTensor(y_data)
        self.len = self.y_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index] 

    def __len__(self):
        return self.len

train_ds = TensorData(X_train, Y_train)
trainloader = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
test_ds = TensorData(X_test, Y_test)
testloader = torch.utils.data.DataLoader(test_ds, batch_size=16, shuffle=False)

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(4, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = self.fc2(X)
        X = self.fc3(X)
        X = self.softmax(X)
        return X

model = Classifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-7)

epochs = 10
for epoch in range(epochs):
    print(f'### epoch {epoch+1} ###############')
    for i, data in enumerate(trainloader, 0): 
        input, target = data 
        optimizer.zero_grad()
        pred = model(input)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        print(f'{i+1} : Loss {loss}')

correct = 0
with torch.no_grad():
  for i, data in enumerate(test_ds):
    label = data[1].numpy()
    output = model.forward(data[0].reshape(1,-1))
    pred = output.argmax().item()
    
    if label == pred:
      correct += 1

print(f'정확도 : {correct/len(test_ds)*100:.2f}%')

