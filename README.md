# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: vishal.v
RegisterNumber:24900179  
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv(r"C:\Users\admin\Downloads\student_scores.csv")
print(df)
df.head()

df.tail()
print(df.head())
print(df.tail())

X=df.iloc[:,:-1].values
print(X)

Y=df.iloc[:,1].values
print(Y)

#spilitting training and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

#displaying predicted values
print(Y_pred)

print(Y_test)

#graph plot for training data
plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,Y_test,color='green')
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print('RMSE = ',rmse)

## Output:
![Screenshot 2024-10-18 093710](https://github.com/user-attachments/assets/267d4d0d-3660-4011-a117-a3a3acc460ef)
![Screenshot 2024-10-18 093723](https://github.com/user-attachments/assets/d06ff321-38e8-4fa0-9755-93a56e8982b9)
![Screenshot 2024-10-18 093735](https://github.com/user-attachments/assets/9048c0f1-5887-48ae-a755-644b618fd2b3)





## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
