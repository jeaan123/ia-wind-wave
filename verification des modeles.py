# load and evaluate a saved model
from keras.models import load_model
import numpy as np
from matplotlib import pyplot
import csv

# traville sur les dones
directory = '/Users/jean/Desktop/tipe2/donne/station3p.csv'
file = open(directory, "r")
windD, windS, taillevague, periode, pression, temperature = [], [], [], [], [], []

reader = csv.reader(file)
for a in reader:
    windD.append(a[5])
    windS.append(a[0])
    taillevague.append(a[1])
    periode.append(a[2])
    pression.append(a[3])
    temperature.append(a[4])

file.close()

xt = np.array(windD, dtype=float)
xp = np.array(pression, dtype=float)
xper = np.array(periode, dtype=float)
xh = np.array(taillevague, dtype=float)
y = np.array(windS, dtype=float)
X = np.transpose([xper, xh])  # matrice de 2cologne et bcp ligne
# load modelX
model = load_model('/Users/jean/Desktop/tipe2/model4.h5')


# summarize model.
# model.summary()


def verif(n, x):
    verifx, verify = X[:n, :], y[:n]
    prediction = model.predict(verifx)
    precedent = prediction[0]
    L = [precedent]
    for k in range(1, len(prediction)):
        L.append(prediction[k] * x + L[k - 1] * (1 - x))

    pyplot.title('reel / prediction')
    pyplot.plot(verify, label='reel')
    pyplot.plot(prediction, label='prediction')
    # pyplot.plot(L, label='filtrage')
    pyplot.legend()
    pyplot.show()


def entre():
    pyplot.scatter(xh,xper, s=1 ,label='hauteur/periode')
    #pyplot.plot(xper,label='periode')
    pyplot.show()

def erreur(n):
    verifx, verify,L = X[:n, :], y[:n],[]
    prediction = model.predict(verifx)
    for k in range(len(prediction)):
        l=abs(prediction[k]-verify[k])
        l=l*100/verify[k]
        L.append(l)
    pyplot.plot(prediction, label='erreur')
    pyplot.show()


