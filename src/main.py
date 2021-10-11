from sklearn.datasets import make_regression
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats
import seaborn as sns

# Visualitzarem només 3 decimals per mostra
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Funció per llegir dades en format csv
def load_dataset_from_csv(path):
    dataset = pd.read_csv(path, header=0, delimiter=',')
    return dataset

# Càrrega del dataset
dataset = load_dataset_from_csv('../data/archive/data.csv')

# retorna un vector (x,y), on x és l'índex de la fila i y es el valor
# bk = dataset["Bankrupt?"]

data = dataset.values
x = data[:, 1:] #Variables d'entrada (característiques)
y = data[:, 0] #Bankrupcy (variable de sortida, objectiu, target)

"""
plt.figure()
ax = plt.scatter(bk, roa_c)
plt.show()
"""

"""
# mostrem atribut 0
x = data[:, :]
y = data[:, 0] #Bankrupcy
plt.figure()
plt.xlabel("xlabel")
plt.ylabel("Bankrupcy")
ax = plt.scatter(x[:, 37], y)
plt.show()
"""

#"""
# Mirem la correlació entre els atributs d'entrada per entendre millor les dades
plt.figure()
fig, axes = plt.subplots(figsize=(40, 40))
plt.title("Correlation matrix - Bankrupcy")
corr = dataset.corr()
f_row = corr.head(1) #Obté la primera fila de la matriu de correlació
axes = sns.heatmap(corr, annot=True, linewidths=.5, ax=axes)
plt.show()
#"""

# Mirem la relació entre atributs utilitzant la funció pairplot
#relacio = sns.pairplot(dataset)