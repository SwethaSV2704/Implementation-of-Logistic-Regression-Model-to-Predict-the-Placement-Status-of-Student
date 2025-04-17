# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data. 
2. Print the placement data and salary data. 
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SWETHA S V
RegisterNumber:  212224230285
*/
import pandas as pd
data=pd.read_csv("C:\\Users\\admin\\Downloads\\Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x
y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion


from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
## Data head
![Screenshot 2025-04-17 081937](https://github.com/user-attachments/assets/c142f563-d73a-4b0f-bb76-c2c89e2c37bd)
## Data1 head
![Screenshot 2025-04-17 082042](https://github.com/user-attachments/assets/5111f36e-3e9d-463c-a102-daf8a5a6e3fe)
## isnull
![Screenshot 2025-04-17 082136](https://github.com/user-attachments/assets/0b70ca2a-2f16-4c7f-9734-13c7ff118675)
## Data duplicate
![Screenshot 2025-04-17 082220](https://github.com/user-attachments/assets/0fdf2499-711d-4510-b044-e1c42eac15e6)
## DATA
![Screenshot 2025-04-17 082338](https://github.com/user-attachments/assets/17af0fa7-e1c0-41de-a8a0-26364d315ca5)
## status
![Screenshot 2025-04-17 082437](https://github.com/user-attachments/assets/b5ede915-cc0d-4f1c-a737-86e894c8683d)
## y_pred
![Screenshot 2025-04-17 082621](https://github.com/user-attachments/assets/767c17b7-b259-4adf-97d2-1a2b62cc21eb)
## Accuracy
![Screenshot 2025-04-17 082657](https://github.com/user-attachments/assets/453b7f1e-7643-42d0-9e45-97f0829378c5)
## Confusion matrix
![Screenshot 2025-04-17 082826](https://github.com/user-attachments/assets/81a4f392-b4ff-4ee2-8111-3f50a6ab3ec7)
## Classification
![Screenshot 2025-04-17 082943](https://github.com/user-attachments/assets/cf684f11-17cd-4984-8e9e-256e9d374463)
## LR predict
![Screenshot 2025-04-17 083025](https://github.com/user-attachments/assets/e51fc194-19b2-470c-8f96-09050cbe7cde)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
