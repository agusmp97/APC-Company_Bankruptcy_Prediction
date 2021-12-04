from sklearn import svm
from sklearn.datasets import make_regression
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


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

def print_data_statistics():
    print("------------------------------------")
    print("Dataset statistics:")
    print(dataset.describe())
    print("------------------------------------")

#print_data_statistics()


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


# +-----------------------+
# | TRACTAMENT D'ATRIBUTS |
# +-----------------------+
# Funció que elimina els espais en blanc del nom dels atributs
def remove_spaces(dataset):
    dataset.columns = dataset.columns.str.replace(' ', '')
    return dataset

dataset = remove_spaces(dataset)


# Eliminació d'atributs no necessaris dels DataFrames
def remove_columns(dataset):
    dataset = dataset.drop('NetIncomeFlag', axis=1)
    return dataset

dataset = remove_columns(dataset)


# Funció que substitueix els valors nuls del dataset pel valor numèric '0'.
def nan_treatment(dataset):
    print("------------------------------------")
    print("Dataset 'NaN' values treatment:")
    any_nan = dataset.isnull().values.any()  # Retorna True si hi ha algun valor NaN al dataset, sino retorna False

    if (any_nan):
        nan_count = dataset.isnull().sum().sum()  # Retorna el resultat numèric de comptar tots els valor NaN
        print("There is {} NaN values on this Dataset!".format(nan_count))
        dataset.fillna(0)
    else:
        print("There is no NaN values on this Dataset!")

    print("------------------------------------")

    return dataset

dataset = nan_treatment(dataset)


# Funció que transforma (escala) els valors del DataFrame, per tal de permetre fer que les diferents
# característiques siguin comparables entre elles.
def standardize_mean(dataset):
    return MinMaxScaler().fit_transform(dataset)

dataset_norm = standardize_mean(dataset)


def split_data(dataset):
    x_data = dataset[:, 1:]  # Característiques
    y_data = dataset[:, 0]  # Variable objectiu (target)
    # Fa el split de les dades d'entrenament i validació.
    x_t, x_v, y_t, y_v = train_test_split(x_data, y_data, train_size=0.8)
    return x_t, x_v, y_t, y_v

x_t, x_v, y_t, y_v = split_data(dataset_norm)



def Logistic_Regressor():
    logireg = LogisticRegression(C=2.0, fit_intercept=True, penalty='l2', tol=0.001, solver='lbfgs', max_iter=1000)
    logireg.fit(x_t, y_t)  # Entrena el model
    probs = logireg.predict_proba(x_v)  # Calcula la probabilitat de que X pertanyi a Y=1
    print("Correct classification Logistic Regression      ", 0.8, "% of the data: ", logireg.score(x_v, y_v))

Logistic_Regressor()
x = 3


def SVM():
    svc = svm.SVC(C=10.0, kernel='rbf', gamma=0.9, probability=True)
    svc.fit(x_t, y_t)  # Entrena el model
    probs = svc.predict_proba(x_v)  # Calcula la probabilitat de que X pertanyi a Y=1

    print("SVM: percentage of samples classified correctly: {}".format(svc.score(x_v, y_v)))


SVM()
x=3

# +----------------------------------------+
# | PCA - transformació de dimensionalitat |
# +----------------------------------------+
# Funció que genera el regressor multivariable entrenat amb un dataset al qual se li ha aplicat una transformació
# de dimensionalitat (amb el mètode PCA).
def make_pca(dataset, player_name, atributes, print_plot):
    dataset__norm = standardize_mean(dataset[atributes])
    x_norm = dataset__norm[atributes[0:-1]]
    y_norm = dataset__norm[atributes[-1]]  # Aquest és l'atribut a predir

    # Separa les dades entre el conjunt d'entrenament i de validació
    x_train_norm, x_val_norm, y_train_norm, y_val_norm = train_test_split(x_norm, y_norm, test_size=0.2)

    # Vectors que emmagatzemaran les dades per generar els gràfics de l'evolució de l'error.
    mse_vect = []
    i_vect = []
    r2_vect = []

    # Fa la transformació de dimensionalitat per un nombre incremental de components principals.
    for i in range(1, len(atributes)):
        pca = PCA(i)
        x_train_norm_pca = pca.fit_transform(x_train_norm.values)  # Transformació de les dades de training.
        x_test_norm_pca = pca.transform(x_val_norm.values)  # Transformació de les dades de validació.

        total_var = pca.explained_variance_ratio_.sum() * 100  # Variança total
        lab = {str(j): f"PC {j + 1}" for j in range(i)}
        lab['color'] = 'Game_score'

        fig = px.scatter_matrix(
            x_test_norm_pca,
            color=y_val_norm,
            dimensions=range(i),
            labels=lab,
            title=f'Total Explained Variance: {total_var:.2f}%',
        )
        fig.update_traces(diagonal_visible=False)
        fig.show()

        linear_model = LogisticRegression()  # Crea el model regressor
        linear_model.fit(x_train_norm_pca, y_train_norm)  # Entrena el regressor amb les dades d'entrenament
        preds = linear_model.predict(x_test_norm_pca)  # Fa la predicció sobre les dades de validació

        mse_result = mse(y_val_norm, preds)
        i_vect.append(i)
        mse_vect.append(mse_result)

        r2 = r2_score(y_val_norm, preds)
        r2_vect.append(r2)
        print("PCA %s: %d - MSE: %f - R2: %f" % (player_name, i, mse_result, r2))

    if print_plot:
        plt.figure()
        ax = plt.scatter(x_test_norm_pca[:, 0], y_val_norm)
        plt.plot(x_test_norm_pca[:, 0], preds, 'r')
        plt.show()

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(i_vect, r2_vect, 'b', label='R2_Score')
    ax.plot(i_vect, mse_vect, 'r', label='MSE')
    ax.legend(bbox_to_anchor=(1, 0.8))
    plt.title("Error per dimensionalitat PCA")
    plt.show()


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