import sys
import numpy as np
import matplotlib.pyplot as plt
import random as rnd
from random import randint
from numpy import array
from numpy import argmax
from pandas import concat
from pandas import DataFrame
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from sklearn.model_selection import train_test_split

#DATA

#Random Linear Increments per data sample
random_increments= []
for i in range(200):
    random_increments.append(rnd.randint(0,10))


#Randomized Sequential Normalized Integer Vectors
Data = []
for i in range(200):
    random1 = rnd.randint(0,100)
    random2 = rnd.randint(0,100)
    Data.append([[(random1+(random_increments[i]*j))/100, (random2+(random_increments[i]*j))/100] for j in range(20)])

Data1 = []
for i in range(200):
    Data1.append([Data[i][-1]])

print(Data)

target = []
for i in range(200):
    add = (Data[i][0][0] + Data[i][0][1] + Data[i][-1][0] + Data[i][-1][1] + (2*random_increments[i]/100))
    target.append([add])


print(target)




data = np.array(Data1, dtype=float)
target = np.array(target, dtype=float)

#Split Data into Test and Train

x_train,x_test,y_train,y_test=train_test_split(data,target,test_size=0.2,random_state=4)


#Model Inititation

choose = input("Type 1 for the LSTM Model or 2 for the DNN Model:")
if choose == "1":
    print("LSTM was chosen")
    data = np.array(Data, dtype=float)
    target = np.array(target, dtype=float)
    x_train,x_test,y_train,y_test=train_test_split(data,target,test_size=0.2,random_state=4)
    model=Sequential()
    model.add(LSTM((1),activation="tanh", recurrent_activation="sigmoid", batch_input_shape=(None,None,2),return_sequences=False))
    model.add(Dense((1)))

elif choose == "2":
    print("DNN was chosen")
    data = np.array(Data1, dtype=float)
    target = np.array(target, dtype=float)
    x_train,x_test,y_train,y_test=train_test_split(data,target,test_size=0.2,random_state=4)
    model=Sequential()
    model.add(Dense((5),activation="relu", batch_input_shape=(None,None,2)))
    model.add(Dense((1)))

#Model Compilation

model.compile(loss='mean_absolute_error', optimizer='adam',metrics=['accuracy'])
model.summary()

history = model.fit(x_train,y_train,epochs=1000,validation_data=(x_test,y_test))
results = model.predict(x_test)

#Result Plotting, Loss function and Scatter Plot of Test samples

plot1 = plt.figure(1)
plt.scatter(range(40), results,c='r', label='Predicted Value')
plt.scatter(range(40), y_test,c='g', label='Actual Value')
plt.title('Predicted Data Points')
plt.xlabel('Test Point')
plt.ylabel('Target Output')
plt.legend(loc='upper center', ncol=2)
plt.ylim((-0.5,7))

plot2 = plt.figure(2)
plt.plot(history.history['loss'])
plt.title('Mean Absolute Error Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')


plt.show()

print("Training Done")
