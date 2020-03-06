import matplotlib.pyplot as plt

import numpy as np

from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score

from sklearn.metrics import precision_recall_curve

import matplotlib.image as mpimg 

#aplanar las imagenes
imagenes = []
es = []
for i in range(1,101):
    una = plt.imread("Train/{}.jpg".format(i))[:,:,0].flatten()
    imagenes.append(una)
    sera = i
    if sera%2 ==0:
        es.append(0)
    else:
        es.append(1)


# Vamos a hacer un split training test en la mitad

scaler = StandardScaler()

x_train, x_test, y_train, y_test = train_test_split(imagenes, es, train_size=0.5)

x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test)

#Un metodo de SVC
def SVCeando(c,x_train, x_test, y_train, y_test):
    SVCeo =SVC(C= c)
    SVCeo.fit(x_train, y_train)
    elefeuno = f1_score(y_test,SVCeo.predict(x_test) )
    return elefeuno, SVCeo

#Busquemos el C
c =np.logspace(-5,5,1000)
f1 = []
for a in c:
    elefe, elfit = SVCeando(a,x_train[:,0:10], x_test[:,0:10], y_train, y_test)
    f1.append(elefe)

#se escoje el mejor C
dd = np.argmax(f1)
elbuenC = f1[dd]

#Se construye con las nuevas imagenes el CSV
#Las aplanamos
import os
lasdensasimagenes = []
entries = os.listdir('Test/')
for entrada in entries:
    una = plt.imread("Test/{}".format(entrada))[:,:,0].flatten()
    lasdensasimagenes.append(una)
    
lasdensasimagenes = scaler.fit_transform(lasdensasimagenes)

elefe, elfit = SVCeando(elbuenC,x_train[:,0:10], x_test[:,0:10], y_train, y_test)
prediccionaes = elfit.predict(lasdensasimagenes[:,0:10])
print(prediccionaes)
#Escribimos el archivo
#import csv
#c = csv.writer(open("Test/predict_test.csv", "wb"))
#c.writerow('Nombre del archivo', 'Prediccion')
#for i in range(10):
#    c.writerow([entries[i],prediccionaes[i]])

