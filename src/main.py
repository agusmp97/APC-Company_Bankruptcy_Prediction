from sklearn import svm
from sklearn.datasets import make_regression
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

from sklearn.utils import resample


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


# print_head()


# Funció que mostra els tipus de dades de les característiques del Dataset.
def print_data_types():
    print("------------------------------------")
    print("Dataset data types:")
    print(dataset.info())
    print("------------------------------------")


print_data_types()


def print_data_statistics():
    print("------------------------------------")
    print("Dataset statistics:")
    print(dataset.describe())
    print("------------------------------------")


# print_data_statistics()


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


# df_dimensionality(dataset)


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


# y_balance(dataset)


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


# pearson_correlation(dataset)


def make_histograms(dataset):
    plt.figure()
    plt.title("Pairwise relationships - Company Bankrupcy")
    sns.pairplot(dataset)
    plt.savefig("../figures/histograms_matrix.png")
    plt.show()


# make_histograms(dataset)


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


def split_data(dt):
    x_data = dt[:, 1:]  # Característiques
    y_data = dt[:, 0]  # Variable objectiu (target)
    # Fa el split de les dades d'entrenament i validació.
    x_t, x_v, y_t, y_v = train_test_split(x_data, y_data, train_size=0.8)
    return x_t, x_v, y_t, y_v


x_t, x_v, y_t, y_v = split_data(dataset.values)


# Funció que transforma (escala) els valors del dataset de training i de test, per tal de permetre fer que les diferents
# característiques siguin comparables entre elles.
def standardize_data(dt_training, dt_test):
    scaler = MinMaxScaler()
    training_scaled = scaler.fit_transform(dt_training)
    test_scaled = scaler.transform(dt_test)
    return training_scaled, test_scaled
    # return MinMaxScaler().fit_transform(dt_training)


x_t_norm, x_v_norm = standardize_data(x_t, x_v)


# w=3


# Funció que calcula les diferents mètriques d'avaluació per comprovar el funcionament dels classificadors.
def model_scorings(ground_truth, preds, model_name):
    f1 = f1_score(ground_truth, preds)
    precision = precision_score(ground_truth, preds)
    recall = recall_score(ground_truth, preds)
    accuracy = accuracy_score(ground_truth, preds)

    print("------------------------------------")
    print("{} scorings:".format(model_name))
    print("    Precision: {}".format(precision))
    print("    Recall: {}".format(recall))
    print("    F1: {}".format(f1))
    print("    Accuracy: {}".format(accuracy))
    print("------------------------------------")


# +------------------------+
# | CLASSIFICATION METHODS |
# +------------------------+

def logistic_regression():
    lr = LogisticRegression(fit_intercept=True, tol=0.001, max_iter=1000000)

    lr_params = {
        'C': [0.1, 1, 10, 100, 1000],
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'solver': ['lbfgs', 'sag', 'saga']
    }

    # Best params
    # lr_params = {'C': [1], 'penalty': ['l2'], 'solver': ['lbfgs']}

    lr_gs = GridSearchCV(estimator=lr, param_grid=lr_params, n_jobs=-1)  # Busca els millors hiperparàmetres pel LR
    lr_gs.fit(x_t_norm, y_t)  # Entrena el model

    print("LR Best Params: {}".format(lr_gs.best_params_))
    print("LR Training score with best params: {}".format(lr_gs.best_estimator_.score(x_t_norm, y_t)))
    print("LR Test score with best params: {}".format(lr_gs.best_estimator_.score(x_v_norm, y_v)))

    lr_preds = lr_gs.best_estimator_.predict(x_v_norm)
    print("LR prediction metrics: {}".format(metrics.classification_report(y_true=y_v, y_pred=lr_preds)))

    model_scorings(list(y_v), list(lr_preds), "Logistic Regressor")

# logistic_regression()
# x = 3


def svm_classifier():
    svc = svm.SVC(probability=True)

    svc_params = {
        'C': [0.1, 1, 10, 100, 1000],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': [3, 4, 5]
    }

    svc_gs = GridSearchCV(estimator=svc, param_grid=svc_params, n_jobs=-1)  # Busca els millors hiperparàmetres pel SVC
    svc_gs.fit(x_t_norm, y_t)  # Entrena el model

    print("SVC Best Params: {}".format(svc_gs.best_params_))
    print("SVC Training score with best params: {}".format(svc_gs.best_estimator_.score(x_t_norm, y_t)))
    print("SVC Test score with best params: {}".format(svc_gs.best_estimator_.score(x_v_norm, y_v)))

    svc_preds = svc_gs.best_estimator_.predict(x_v_norm)
    print("SVC prediction metrics: {}".format(metrics.classification_report(y_true=y_v, y_pred=svc_preds)))

# svm_classifier()


def knn_classifier():
    knn = KNeighborsClassifier(algorithm='auto')

    knn_params = {
        'n_neighbors': [2, 5, 10, 20, 40, 80],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }

    knn_gs = GridSearchCV(estimator=knn, param_grid=knn_params, n_jobs=-1)  # Busca els millors hiperparàmetres pel KNN
    knn_gs.fit(x_t_norm, y_t)  # Entrena el model

    print("KNN Best Params: {}".format(knn_gs.best_params_))
    print("KNN Training score with best params: {}".format(knn_gs.best_estimator_.score(x_t_norm, y_t)))
    print("KNN Test score with best params: {}".format(knn_gs.best_estimator_.score(x_v_norm, y_v)))

    knn_preds = knn_gs.best_estimator_.predict(x_v_norm)
    print("KNN prediction metrics: {}".format(metrics.classification_report(y_true=y_v, y_pred=knn_preds)))

    """
    knn.fit(x_t, y_t)  # Entrena el model
    probs = knn.predict_proba(x_v)  # Calcula la probabilitat de que X pertanyi a Y=1
    print("Correct classification KNN     ", 0.8, "% of the data: ", knn.score(x_v, y_v))
    """

#knn_classifier()
#x = 3


def random_forest_classifier():
    rfc = RandomForestClassifier()

    rfc_params = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [4, 5, 6, 7, 8],
        'criterion': ['gini', 'entropy']
    }

    rfc_gs = GridSearchCV(estimator=rfc, param_grid=rfc_params, n_jobs=-1)  # Busca els millors hiperparàmetres pel RF
    rfc_gs.fit(x_t_norm, y_t)  # Entrena el model

    print("RFC Best Params: {}".format(rfc_gs.best_params_))
    print("RFC Training score with best params: {}".format(rfc_gs.best_estimator_.score(x_t_norm, y_t)))
    print("RFC Test score with best params: {}".format(rfc_gs.best_estimator_.score(x_v_norm, y_v)))

    rfc_preds = rfc_gs.best_estimator_.predict(x_v_norm)
    print("RFC prediction metrics: {}".format(metrics.classification_report(y_true=y_v, y_pred=rfc_preds)))

    """
    rfc.fit(x_t_norm, y_t)
    rfc_preds = rfc.predict(x_v_norm)
    model_scorings(list(y_v), list(rfc_preds), "Random Forest Classifier")
    """


random_forest_classifier()
x=3


df_majority = dataset[dataset.balance==0]
df_minority = dataset[dataset.balance==1]
w=0