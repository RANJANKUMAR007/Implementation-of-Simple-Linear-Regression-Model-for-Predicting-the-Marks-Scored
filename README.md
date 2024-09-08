# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn. dataset
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph. Compare the graphs and hence we obtained the linear regression for the given datas. 
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: RANJANKUMAR G
RegisterNumber:212223240138
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error

## dataset:
dataset

![Screenshot 2024-09-08 220332](https://github.com/user-attachments/assets/5818837d-ceb9-4a7f-aec1-01a0f2663315)

## Head And Tail
dataset = pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())

## output:
![Screenshot 2024-09-08 220350](https://github.com/user-attachments/assets/4983ef7c-a214-4ee1-8fe6-0f41b60e5d3e)

## x and y value
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,-1].values
print(y)

## output:
![Screenshot 2024-09-08 220406](https://github.com/user-attachments/assets/ddc73ebb-af74-43f2-b5b6-9262085f4559)
## Predictions of x and y
## program:
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=0)
x_test.shape

## output:
![Screenshot 2024-09-08 220422](https://github.com/user-attachments/assets/aab6de5f-5073-4e8c-9f87-10161a5a32bb)

## program:
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)

## output:
![Screenshot 2024-09-08 221725](https://github.com/user-attachments/assets/e6b998f7-016c-481a-b59f-71cb099f956b)

## program:
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)
print(y_pred)
print(y_test)

## output":
![Screenshot 2024-09-08 220442](https://github.com/user-attachments/assets/12f60aeb-f03d-4d8f-a620-77efc15e432f)

## MSE,MAE AND RMSE
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)

## output:
![Screenshot 2024-09-08 220455](https://github.com/user-attachments/assets/030e019b-f513-4c0e-aa5f-bb12ce10f487)

## Training Set:
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,reg.predict(x_train),color='purple')
plt.title('Training set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("scores")
plt.show()
plt.scatter(x_test,y_test,color='black')
plt.plot(x_test,reg.predict(x_test),color='yellow')
plt.title('test set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("scores")
plt.show()

*/
```

## Output:

![Screenshot 2024-09-08 220526](https://github.com/user-attachments/assets/6d7b9639-6666-4f6e-aaa3-8c9bcc57f948)
![Screenshot 2024-09-08 220546](https://github.com/user-attachments/assets/37b064a5-ec19-4f70-8317-782b5ce0a6e3)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
