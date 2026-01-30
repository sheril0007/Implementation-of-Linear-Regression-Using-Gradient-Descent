# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start
2. Initialize data, parameters, and learning rate
3. Repeat for given iterations: • Predict output • Compute loss • Update weight and bias
4. Plot results and display parameters
5. Stop 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: SHERIL P
RegisterNumber:  25012800
*/
```
```
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("ex3.csv")

x = data['R&D Spend'].values
y = data['Profit'].values

import numpy as np
import matplotlib.pyplot as plt

w = 0.0
b = 0.0
alpha = 0.0000000001
epochs = 100
n = len(x)

losses = []


for _ in range(epochs):
    y_hat = w * x + b
    loss = np.mean((y_hat - y) ** 2)
    losses.append(loss)

    dw = (2/n) * np.sum((y_hat - y) * x)
    db = (2/n) * np.sum(y_hat - y)

    w -= alpha * dw
    b -= alpha * db

plt.figure(figsize=(15,5))
plt.subplot(1, 2, 1)
plt.plot(losses, color="blue")
plt.xlabel("No of Iterations")
plt.ylabel("Loss")
plt.title("LOSS VS ITERATIONS")

plt.figure(figsize=(11, 5))
plt.subplot(1, 2, 2)
plt.scatter(x, y, color="red", label="Data")
plt.plot(x, w * x + b, color="green", label="Regression Line")
plt.xlabel("R&D Spend")
plt.ylabel("Profit")
plt.title("PROFIT VS R&D SPEND")
plt.legend()

plt.tight_layout()
plt.show()

print("Final weight (w):", w)
print("Final bias (b):", b)
```

## Output:
<img width="825" height="519" alt="Screenshot 2026-01-30 084840" src="https://github.com/user-attachments/assets/79cbb07d-c574-4bc5-96cb-3af0e05edcdf" />

<img width="798" height="624" alt="Screenshot 2026-01-30 084856" src="https://github.com/user-attachments/assets/32a7fd06-1610-4211-89e7-79ad0f8a718a" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
