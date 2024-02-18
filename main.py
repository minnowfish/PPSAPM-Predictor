import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('MINNOWFISH_PPSAPM.csv')

pps = data['PPS'].values
apm = data['APM'].values

plt.scatter(pps, apm, marker ='o', color ='blue')
plt.title('MINNOWFISH')
plt.xlabel('Pieces Per Second (PPS)')
plt.ylabel('Attack Per Minute (APM)')
plt.show()

#training model
x_train, x_test, y_train, y_test = train_test_split(pps, apm, test_size=0.4, random_state=42) #42! The answer to the ultimate question of life

x_train = x_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)

model = LinearRegression()
model.fit(x_train, y_train)

predictions = model.predict(x_test)

plt.scatter(x_test, y_test, marker='o', color='blue', label='Actual APM')
plt.plot(x_test, predictions, color='red', linewidth=2, label='Predicted APM')
plt.title('MINNOWFISH')
plt.xlabel('Pieces Per Second (PPS)')
plt.ylabel('Attack Per Minute (APM)')
plt.legend()
plt.show()
