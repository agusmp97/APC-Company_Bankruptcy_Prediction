from sklearn.datasets import make_regression
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats
import seaborn as sns


# +------------------+
# | CÀRREGA DE DADES |
# +------------------+
# Funció per llegir dades en format csv
def load_dataset_from_csv(path):
    dataset = pd.read_csv(path, header=0, delimiter=',')
    return dataset

dataset = load_dataset_from_csv('../data/archive/data.csv')

data = dataset.values  # Conté les dades del dataset sense les capçaleres
x = data[:, 1:]  # Variables d'entrada (característiques)
y = data[:, 0]  # Bankrupcy (variable de objectiu, target)


# +--------------------------+
# | VISUALITZACIÓ INFORMACIÓ |
# +--------------------------+
# Funció que mostra els primers 5 registres del Dataset
def print_head():
    print("Dataset first 5 rows:")
    print(dataset.head())
    print("------------------------------------")

#print_head()


# Funció que mostra els tipus de dades de les característiques del Dataset.
def print_data_types():
    print("------------------------------------")
    print("Dataset data types:")
    print(dataset.dtypes)
    print("------------------------------------")

#print_data_types()


# Funció que mostra la dimensionalitat del Dataset
def df_dimensionality(dataset):
    data = dataset.values
    # separa l'atribut objectiu Y de les caracterísitques X
    x_data = data[:, :-1]  # Característiques d'entrada
    y_data = data[:, -1]  # Variable objectiu (target)
    print("DataFrame dimensionality: {}:".format(dataset.shape))
    print("Features (X) dimensionality: {}".format(x_data.shape))
    print("Target (Y) dimensionality: {}".format(y_data.shape))
    print("------------------------------------")

#df_dimensionality(dataset)


# Funció que calcula si les dades estan balancejades.
# És a dir, si el nombre de mostres de les dues classes és semblant.
# Guarda un plot amb aquesta informació.
def y_balance(dataset):
    ax = sns.countplot(x="Bankrupt?", data=dataset, palette={0: 'cornflowerblue', 1: "firebrick"})
    plt.suptitle("Target attribute distribution (Company banrupcy)")
    label = ["No bankrupt", "Bankrupt"]
    ax.bar_label(container=ax.containers[0], labels=label)
    plt.xlabel('Bankrupt')
    plt.ylabel('Number of samples')
    plt.savefig("../figures/distribucio_atribut_objectiu.png")
    plt.show()

    data = dataset.values
    bk = data[:, 0]
    bk_perc = (len(bk[bk == 1]) / len(bk)) * 100
    print('Percentage of companies that go bankrupt: {:.2f}%'.format(bk_perc))

#y_balance(dataset)


# +-----------------------+
# | CORRELACIÓ D'ATRIBUTS |
# +-----------------------+
# Funció que genera la matriu de correlació de Pearson d'un DataFrame i genera el plot
def pearson_correlation(dataset):
    plt.figure()
    fig, ax = plt.subplots(figsize=(100, 40))  # figsize controla l'amplada i alçada de les cel·les de la matriu
    plt.title("Correlation matrix - Company Bankrupcy")
    sns.heatmap(dataset.corr(), annot=True, linewidths=.5, ax=ax)
    plt.savefig("../figures/pearson_correlation_matrix_.png")
    plt.show()

#pearson_correlation(dataset)


def make_histograms(dataset):
    plt.figure()
    plt.title("Pairwise relationships - Company Bankrupcy")
    sns.pairplot(dataset)
    plt.savefig("../figures/histograms_matrix.png")
    plt.show()

#make_histograms(dataset)

#x=3

"""
# Mirem la correlació entre els atributs d'entrada per entendre millor les dades
plt.figure()
fig, axes = plt.subplots(figsize=(40, 40))
plt.title("Correlation matrix - Bankrupcy")
corr = dataset.corr()
f_row = corr.head(1) #Obté la primera fila de la matriu de correlació
axes = sns.heatmap(corr, annot=True, linewidths=.5, ax=axes)
plt.show()
"""

# Mirem la relació entre atributs utilitzant la funció pairplot
#relacio = sns.pairplot(dataset)

x = 3

# +-----------------------+
# | TRACTAMENT D'ATRIBUTS |
# +-----------------------+



# +----------------------------------------+
# | PCA - transformació de dimensionalitat |
# +----------------------------------------+


# +-------------------------------+
# | SVM - Support Vectors Machine |
# +-------------------------------+



# +------------------------------------+
# | ALTRES (ESBORRAR ABANS D'ENTREGAR) |
# +------------------------------------+

# retorna un vector (x,y), on x és l'índex de la fila i y es el valor
# bk = dataset["Bankrupt?"]


"""
plt.figure()
ax = plt.scatter(data[:, 0], data[:, 1])
plt.show()
"""

x=3

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