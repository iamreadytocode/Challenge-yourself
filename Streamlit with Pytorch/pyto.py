import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import streamlit as st
from sklearn import preprocessing
import numpy as np

st.set_page_config(page_title="Pytorch For Data Analysis",layout="centered",initial_sidebar_state="auto",menu_items=None)
st.title("ANALYZE YOUR DATA ")
st.header("--Pytorch--")

data = pd.read_csv("titanic_dataset.csv")

st.write(data.head(5))

# Convert X and y to Pandas DataFrames
# rem = st.text_input("Enter the number of columns to remove.", key="removal_input")
# rem=int(rem)
# for i in range(rem):
#     noc = st.text_input(f"Enter the name of column {i + 1}", key=f"column_input_{i}")
#     data.drop([noc], axis=1, inplace=True)
data.drop(["PassengerId","Name","Sex","Ticket","Cabin","Embarked"], axis=1, inplace=True)
st.write(data.head(5))

target=st.text_input("Input the target variable:")
x = data.drop(columns=[target])
y = data[target]

# Display the loaded data
st.write(data.corr())
x.dropna()


st.write("Titanic Dataset:")
st.write(x.head())


st.write("Survived")
st.write(y.head())

prepro = preprocessing.StandardScaler().fit(x)

x_trans = prepro.fit_transform(x)

y = y.to_numpy()
y = y.reshape(-1,1)

prepro2= preprocessing.StandardScaler().fit(y)

y_trans = prepro2.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=101)

st.write("Standardize the input features")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

st.write(X_train)

st.write("Convert data to PyTorch tensors")
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

st.write(X_train)

st.write("Define a simple regression model") 
class RegressionModel(nn.Module):
    def __init__(self, input_size):
        super(RegressionModel, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 128) 
        self.relu2 = nn.ReLU()        
        self.fc3 = nn.Linear(128, 64) 
        self.relu3 = nn.ReLU()        
        self.fc4 = nn.Linear(64, 32)   # Third hidden layer
        self.relu4 = nn.ReLU()         # Activation function for the third hidden layer
        self.fc5 = nn.Linear(32, 16)   # Fourth hidden layer
        self.relu5 = nn.ReLU()         # Activation function for the fourth hidden layer
        self.fc6 = nn.Linear(16, 1)  

    def forward(self, x):
       x = self.fc1(x)
       x = self.relu(x)
       x = self.fc2(x)
       x = self.relu2(x)
       x = self.fc3(x)
       x = self.relu3(x)
       x = self.fc4(x)
       x = self.relu4(x)
       x = self.fc5(x)
       x = self.relu5(x)
       x = self.fc6(x)
       return x
    
st.write("Create the model")
input_size = X_train.shape[1]
st.write(input_size)
model = RegressionModel(input_size)
st.write(model)    

st.write("Define loss and optimizer") 
Lossf = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

st.write("Training loop")
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)
    loss = Lossf(outputs, y_train.view(-1, 1))

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
st.write(" Save the trained model")
torch.save(model.state_dict(), 'Titanic_Dataset.pth')

st.write("Load the model for future use")
loaded_model = RegressionModel(input_size)
loaded_model.load_state_dict(torch.load('Titanic_Dataset.pth'))

mean_value = torch.tensor(np.nanmean(X_test), dtype=torch.float32)

# Impute NaN values in X_test
X_test_imputed = X_test.clone()  # Create a deep copy
X_test_imputed[torch.isnan(X_test)] = mean_value

st.write("Checking for null values")
st.write(np.isnan(y_test).any())


st.write("Evaluate the loaded model on the test set")
with torch.no_grad():
    y_pred = loaded_model(X_test_imputed)
    st.write(np.isnan(y_pred).any())
    y_pred[torch.isnan(y_pred)] = 0.0 
    mse = mean_squared_error(y_test.numpy(), y_pred.numpy())
    st.write(f'Mean Squared Error on Test Data (Loaded Model): {mse:.4f}')

        