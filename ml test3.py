import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from tensorflow import keras
from keras.layers import Input, Concatenate, Conv2D, Flatten, Dense
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from matplotlib import pyplot
import csv

#traville sur les dones
directory='/Users/jean/Desktop/tipe2/donne/brut/201851000.csv'
file = open(directory, "r")
windD,windS,taillevague,periode,pression,temperature=[],[],[],[],[],[]
entre=[]
reader = csv.reader(file)
for row in reader:
   a=str(row).split()
   #print('wind dir(deg)=',a[5],'wind speed(m/s)=',a[6],'taille vague(m)=',a[8],'periode(sec)=',a[10],'pression(hpa)=',a[12])
   windD.append((a[5]))
   windS.append((a[6]))
   taillevague.append((a[8]))
   periode.append((a[10]))
   pression.append((a[12]))
   temperature.append((a[14]))
file.close()

xt=np.array([windD[4:]], dtype=float)
xp = np.array([pression[4:]], dtype=float)
xd = np.array([periode[4:]], dtype=float)
xs = np.array([taillevague[4:]], dtype=float)
yt= np.array( [windS[3:8713]], dtype=float)#vent n-1
yt2= np.array( [windS[2:8712]], dtype=float)#vent n-2

y = np.array(windS[4:], dtype=float)

entre1 = np.concatenate((xd,xs),axis=0)
X= np.transpose(entre1) #matrice de 2cologne et bcp ligne


###,nrmaliser X
##scale = StandardScaler()
##X = scale.fit_transform(X_pasnormalisé)
##print (X)


#figure en3d 

##fig = plt.figure()
##ax = fig.add_subplot(1,2,1, projection='3d')
##ax.scatter(xs, xp, y, c='r', marker='^')
## 
##ax.set_xlabel('hauteur vague')
##ax.set_ylabel('periode')
##ax.set_zlabel('vitesse vent')
## 
##plt.show()


# split into train and test
n_train = len(y)//2
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]

# define the keras model
model = Sequential()
model.add(Dense(5, input_dim=2, activation='elu',kernel_initializer='lecun_normal'))
#model.add(Dense(4, activation='elu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(1, activation='linear'))
# compile the keras model
#opt = SGD(lr=0.01, momentum=0.9)
model.compile(loss='mean_squared_error', optimizer='adam',)
# fit the keras model on the dataset
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=200, verbose=0,batch_size=39)
# evaluate the keras model
train_mse = model.evaluate(trainX, trainy, verbose=0)
test_mse = model.evaluate(testX, testy, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_mse, test_mse))

# plot loss during training
pyplot.title('Loss / MeanSquared Error')
#pyplot.plot(history.history['mean_absolute_percentage_error'])
pyplot.plot(history.history['loss'], label='loss')
#pyplot.plot(history.history['val_loss'], label='loss')
#pyplot.plot(history.history['accuracy'], label='accuracy')
pyplot.legend()
pyplot.show()

# save model and architecture to single file
#model.save("model3.h5")




def verif(n):
   verifx,verify = X[:n, :], y[:n]
   prediction=model.predict(verifx)
   pyplot.title('reel / prediction')
   pyplot.plot(verify, label='reel')
   pyplot.plot(prediction, label='prediction')
   pyplot.legend()
   pyplot.show()


    
