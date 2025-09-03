# BLENDED_LEARNING
# Implementation-of-Linear-and-Polynomial-Regression-Models-for-Predicting-Car-Prices

## AIM:
To write a program to predict car prices using Linear Regression and Polynomial Regression models.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. **Data Collection**:  
   - Import essential libraries such as pandas, numpy, sklearn, matplotlib, and seaborn.  
   - Load the dataset using `pandas.read_csv()`.  

2. **Data Preparation**:  
   - Address any missing values in the dataset.  
   - Select key features for training the models.  
   - Split the data into training and testing sets using `train_test_split()`.  

3. **Linear Regression**:  
   - Initialize a Linear Regression model using sklearn.  
   - Train the model on the training data with the `.fit()` method.  
   - Use the model to predict values for the test set with `.predict()`.  
   - Evaluate performance using metrics like Mean Squared Error (MSE) and R² score.  

4. **Polynomial Regression**:  
   - Generate polynomial features using `PolynomialFeatures` from sklearn.  
   - Train a Linear Regression model on the transformed polynomial dataset.  
   - Make predictions and assess the model's performance similarly to the Linear Regression approach.  

5. **Visualization**:  
   - Plot regression lines for both Linear and Polynomial models.  
   - Visualize residuals to analyze the models' performance.  


## Program:
```

Program to implement Linear and Polynomial Regression models for predicting car prices.
Developed by: LAAKSHIT D
RegisterNumber: 212222230071

```
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn. pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib. pyplot as plt
```

```python
df = pd.read_csv('encoded_car_data (1).csv')
```

```python
X = df[['enginesize','horsepower','citympg','highwaympg']]
y = df['price']
```

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
linear_model = Pipeline([
    ('scaler',StandardScaler()),
    ('model',LinearRegression())
    
])
linear_model.fit(X_train,y_train)
y_pred_linear = linear_model.predict(X_test)
```

```python
poly_model = Pipeline([
    ('poly',PolynomialFeatures(degree=2)),
    ('scaler',StandardScaler()),
    ('model',LinearRegression())
])
poly_model.fit(X_train,y_train)
y_pred_poly=poly_model.predict(X_test)
```

```python
print('AMIRTHA VARSHINI M')
print('Reg NO: 212224230017 ')
print("Linear Regression Model:")
mse=mean_squared_error(y_test,y_pred_poly)
print('MSE=',mean_squared_error(y_test,y_pred_poly))
r2score=r2_score(y_test,y_pred_poly)
print('R2 score=',r2score)


print(f"MSE: {mean_squared_error(y_test, y_pred_poly):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred_poly):.2f}")

print("\nPolynomial Regression Model:")
print(f"MSE: {mean_squared_error(y_test, y_pred_poly):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred_poly):.2f}")

plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred_poly, label='Linear',alpha=0.6)
plt.scatter(y_test,y_pred_poly,label='Polynomial (degree=2)',alpha=0.6)
plt.plot([y.min(),y.max()],[y.min(),y.max()],'r--',label='Perfect Predicton')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.show()
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)

<img width="1494" height="807" alt="Screenshot 2025-09-03 093757" src="https://github.com/user-attachments/assets/275ec55d-5760-4ece-9bca-64992b717f39" />


## Result:
Thus, the program to implement Linear and Polynomial Regression models for predicting car prices was written and verified using Python programming.
