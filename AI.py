#import libraries

[1] #Description: This program classification patients as having kidney disease (ckd) or using Artificial Inteligece Neural Network(ANN). This project havebuilt by Bhavesh Mali, Parul Patel, Amandeep Kaur & Shubham Kumar.

import glob
from keras.models import Sequential, load_model
import numpy as np
import numpy as pd
from keras.layers import Dense
from Sklearn.model_selection import train_test_split
from Sklearn.preprocessing import LabelEncoder, MinMaxScaler
import keras as K

[2] #Load the data
from google.colab import files
uploaded = files.upload()
df = pd.read_csv('kidney_disease.csv')

[3]#print the first 5 rows
df.head ()

[4]#Get the shape of the data (the number of rows & cols)
df.shape
(400, 26)

[5]#Create a list of column names to keep
columns_to_retain = ['sg', 'al', 'sc', 'hemo', 'pcv', 'wbcc', 'rbcc', 'htn', 'classification']

[6]#Drop the columns that are not in columns_to_retain
df = df.drop([col for col in df.columns if not columns_to_retain], axis=1)

[7]#Drop the rows with na or missing values
df = df.dropna(axis=0)

[8]#Transform the non-numeric data in the columns
for columns in df.columns:
  if df[column].dtype == np.number:
   continue
   df[column] =LabelEncoder().fit_transform(df[column])

[9]#Print the first 5 rows of the new cleaned data
df.head()

[10]#Split the data into independent (X) data set (the features) and dependent (y) data set  (the target)
y = df['classification'], axis=1)
y = df['classification']

[11]#Feature Scaling
#min-max scaler method scales the data set so that all the input features lie btw 0 and 1
x_scaler = MinMaxScaler()
x_scaler.fit(X)
column_names = X.columns
X[column_names] = x_scaler.transform[x]

[12]#Split the data into 80% training and 20% testing 
X_train, X_test, Y_train, Y_test= tarin_test_split(X, y, test_size = 0.2, shuffle=True)

[13]#Buld the model
model = Sequential
model.add( Dense (256, input_dim= len(X.columns), kernel_initializer=k.initializers.random_normal(seed=13), activation='relu'))
model.add(Dense(1, activation='hard_sigmoid'))

[14]#Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

[15]#Train the model 
history = model.fit(X_train, y_train, epochs = 2000, batch_size= X_train.shape[0])

[16]#Save the model
model.save('ckd.model')

[17}#Visualize the models loss and accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['loss'])
plt.title('model accuracy & loss')
plt.ylabel('accuarcy and loss ')
plt.xlabel('epoch')


[18]#Get the shape of the training and testing data set
print('shape of training data:', X_train.shape )
print('shape of test data:', X_test.shape)


[19]#Show the actual and predicated values
pred = model.predict(X_test)
pred = [1 if y>=0.5 else 0 for y in pred]
pred('Original : {0}.format(",".join(str(x) for x in y_test))
pred('Predicated : {0}.format(",".join(str(x) for x inpred))


[20]#Show the actual values
y_test





  