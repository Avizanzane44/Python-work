import pandas as pd
import numpy as np


# Loading Dataset --> 

Data = pd.read_csv("Loan_Default(1).csv")

print("First five records : ")
print(Data.head())

print("Shape of Dataset : ")
print(Data.shape)

print("Columns are : ")
print(Data.columns)


# Missing Values Wrok

print("missing Values in each Column is : ")
print(Data.isnull().sum())

print("Missing Values in entire dataset : ")
print(Data.isnull().sum().sum())



# Checking Target Class Balanced or not

target = 'Default'   

print("\nTarget Class Counts : ")
print(Data[target].value_counts())

print("\nTarget Class Percentages : ")
print(Data[target].value_counts(normalize=True) * 100)



# Encoding Categorical Data


Data["PreviousDefault"] = Data["PreviousDefault"].map({"Yes" : 1, "No" : 0})

Data["HomeOwnership"] = Data["HomeOwnership"].map({"Own" : 0, "Rent" : 1, "Mortgage" : 2})


# Separating X n Y 

X = Data[['Age', 'Income', 'LoanAmount', 'CreditScore', 'EmploymentYears',
       'ExistingLoans', 'MonthlyDebt', 'LoanTerm', 'PreviousDefault',
       'HomeOwnership']]

Y = Data['Default']



# Splitting Data --> Training n Testing

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size= 0.3,
    random_state= 42
)

print("Training Input Shape : ",X_train.shape)
print("Testing Input Shape : ",X_test.shape)
print("Training Output Shape : ",Y_train.shape)
print("Testing Output Shape : ",Y_test.shape)



# Feature Scaling 

from sklearn.preprocessing import StandardScaler

Scaler = StandardScaler()

X_train = Scaler.fit_transform(X_train)
X_test = Scaler.transform(X_test)


# Creating Model ---> MLP

from sklearn.neural_network import MLPClassifier


model = MLPClassifier(
    hidden_layer_sizes=(22,22,11,11,5),
    activation='tanh',
    solver='adam',
    max_iter= 1000,
    random_state=42,
    learning_rate='adaptive',
    learning_rate_init=0.001
)


# Training Model 


model.fit(X_train, Y_train)

N_itr = model.n_iter_
print("No of iterations Req For training : ",N_itr)

fl = model.loss_
print("Final Loss : ",fl)



# Model Evaluation 

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


Y_pred_train = model.predict(X_train)
Y_pred_test = model.predict(X_test)

Train_Accuracy = accuracy_score(Y_train, Y_pred_train)
Test_Accuracy = accuracy_score(Y_test, Y_pred_test)

print("Trainig Accuracy : ",Train_Accuracy*100,"%")
print("Testing Accuracy : ",Test_Accuracy*100,"%")

cf = confusion_matrix(Y_test, Y_pred_test)
print("Confusion Matrix : ",cf)


cr = classification_report(Y_test, Y_pred_test)
print("Classification Report : ",cr)


# Plotting Training Loss 

import matplotlib.pyplot as plt

plt.plot(model.loss_curve_)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("MLP Training Loss")
plt.grid(True)
plt.show()


# Preserving Model 

import joblib

joblib.dump(model,"Loan_ANN_Model.pkl")
joblib.dump(Scaler,"Loan_Sclaer.pkl")





