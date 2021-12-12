import pandas as pd
import seaborn as sns
import time
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from joblib import dump
from random import randint
import warnings
warnings.filterwarnings('ignore')



# +------------------+
# | CÀRREGA DE DADES |
# +------------------+
# Funció per llegir dades en format csv
def load_dataset_from_csv(path):
    dataset = pd.read_csv(path, header=0, delimiter=',')
    return dataset

#dataset = load_dataset_from_csv('../data/archive/data.csv')
dataset = load_dataset_from_csv('data/archive/data.csv')


# Funció que selecciona 10 registres aleatòriament del dataset per utilitzar-les com a dades de demo.
def export_demo_data(dt):
    dataset_shape = dt.shape[0]
    df_columns = dt.columns
    df_data = []
    aux_dataset = dt

    for i in range(93):
        rnd_val = randint(0, dataset_shape)
        df_data.append(dt.iloc[rnd_val])
        aux_dataset = aux_dataset.drop([rnd_val])
        aux_dataset = aux_dataset.reset_index(drop=True)
        dataset_shape -= 1

    # Per agafar sempre els 5 primers registres que són Bankrupt?=1
    df_data.append(dt.iloc[0])
    df_data.append(dt.iloc[1])
    df_data.append(dt.iloc[2])
    df_data.append(dt.iloc[3])
    df_data.append(dt.iloc[4])
    aux_dataset = aux_dataset.drop([0])
    aux_dataset = aux_dataset.drop([1])
    aux_dataset = aux_dataset.drop([2])
    aux_dataset = aux_dataset.drop([3])
    aux_dataset = aux_dataset.drop([4])
    aux_dataset = aux_dataset.reset_index(drop=True)
    dataset_shape -= 1


    demo_df = pd.DataFrame(df_data, columns=df_columns)
    demo_df = demo_df.reset_index(drop=True)
    #demo_df.to_csv('../data/demo_data.csv', index=False)
    demo_df.to_csv('data/demo_data.csv', index=False)
    return aux_dataset

dataset = export_demo_data(dataset)



# +--------------------------+
# | VISUALITZACIÓ INFORMACIÓ |
# +--------------------------+
# Funció que mostra els primers 5 registres del Dataset
def print_head(dt):
    print("Dataset first 5 rows:")
    print(dt.head())
    print("------------------------------------")

# print_head(dataset)


# Funció que mostra els tipus de dades de les característiques del Dataset.
def print_data_types(dt):
    print("------------------------------------")
    print("Dataset data types:")
    print(dt.info())
    print("------------------------------------")

# print_data_types(dataset)


def print_data_statistics(dt):
    print("------------------------------------")
    print("Dataset statistics:")
    print(dt.describe())
    print("------------------------------------")

# print_data_statistics(dataset)


# Funció que mostra la dimensionalitat del Dataset
def df_dimensionality(dt):
    data = dt.values
    # separa l'atribut objectiu Y de les caracterísitques X
    x_data = data[:, :-1]  # Característiques d'entrada
    y_data = data[:, 0]  # Variable objectiu (target)
    print("DataFrame dimensionality: {}:".format(dt.shape))
    print("Features (X) dimensionality: {}".format(x_data.shape))
    print("Target (Y) dimensionality: {}".format(y_data.shape))
    print("------------------------------------")

# df_dimensionality(dataset)


# Funció que calcula si les dades estan balancejades.
# És a dir, si el nombre de mostres de les dues classes és semblant.
# Guarda un plot amb aquesta informació.
def y_balance(dt):
    ax = sns.countplot(x="Bankrupt?", data=dt, palette={0: 'cornflowerblue', 1: "firebrick"})
    plt.suptitle("Data distribution (Company bankruptcy)")
    label = ["No bankrupt", "Bankrupt"]
    ax.bar_label(container=ax.containers[0], labels=label)
    plt.xlabel('Bankrupt')
    plt.ylabel('Number of samples')
    plt.savefig("../figures/distribucio_atribut_objectiu.png")
    plt.show()

    data = dt.values
    bk = data[:, 0]
    bk_perc = (len(bk[bk == 1]) / len(bk)) * 100
    print('Percentage of companies that go bankrupt: {:.2f}%'.format(bk_perc))


# y_balance(dataset)


# Funció que genera la matriu de correlació de Pearson d'un DataFrame i genera el plot
def pearson_correlation(dt):
    plt.figure()
    fig, ax = plt.subplots(figsize=(100, 40))  # figsize controla l'amplada i alçada de les cel·les de la matriu
    plt.title("Correlation matrix - Company Bankrupcy")
    sns.heatmap(dt.corr(), annot=True, linewidths=.5, ax=ax)
    plt.savefig("../figures/pearson_correlation_matrix_.png")
    plt.show()

# pearson_correlation(dataset)



# +-----------------------+
# | TRACTAMENT D'ATRIBUTS |
# +-----------------------+

# Funció que elimina els espais en blanc del nom dels atributs
def remove_spaces(dt):
    dt.columns = dt.columns.str.replace(' ', '')
    return dt

# dataset = remove_spaces(dataset)


# Eliminació d'atributs no necessaris dels DataFrames
def remove_columns(dt):
    dt = dt.drop('NetIncomeFlag', axis=1)
    return dt

# dataset = remove_columns(dataset)


# Funció que substitueix els valors nuls del dataset pel valor numèric '0'.
def nan_treatment(dt):
    print("------------------------------------")
    print("Dataset 'NaN' values treatment:")
    any_nan = dt.isnull().values.any()  # Retorna True si hi ha algun valor NaN al dataset, sino retorna False

    if not any_nan:
        print("There is no NaN values on this Dataset!")
    else:
        nan_count = dt.isnull().sum().sum()  # Retorna el resultat numèric de comptar tots els valor NaN
        print("There is {} NaN values on this Dataset!".format(nan_count))
        dt.fillna(0)

    print("------------------------------------")

    return dt

# dataset = nan_treatment(dataset)


def split_data(dt):
    x_data = dt[:, 1:]  # Característiques
    y_data = dt[:, 0]  # Variable objectiu (target)
    # Fa el split de les dades d'entrenament i validació.
    x_t, x_v, y_t, y_v = train_test_split(x_data, y_data, train_size=0.8)
    return x_t, x_v, y_t, y_v

# x_t, x_v, y_t, y_v = split_data(dataset.values)


# Funció que transforma (escala) els valors del dataset de training i de test, per tal de permetre fer que les diferents
# característiques siguin comparables entre elles.
def standardize_data(dt_training, dt_test):
    scaler = MinMaxScaler()
    training_scaled = scaler.fit_transform(dt_training)
    test_scaled = scaler.transform(dt_test)
    return training_scaled, test_scaled
    # return MinMaxScaler().fit_transform(dt_training)

# x_t_norm, x_v_norm = standardize_data(x_t, x_v)


def create_confusion_matrix(ground_truth, preds, model_name):
    conf_mat = confusion_matrix(y_true=ground_truth, y_pred=preds)
    print('{} confusion matrix:\n{}'.format(model_name, conf_mat))

    labels = ['Class 0', 'Class 1']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.title('{} Confusion Matrix'.format(model_name))
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    model_filename = model_name.replace(' ', '_')
    plt.savefig("../figures/{}_confusion_matrix.png".format(model_filename))
    plt.show()


# +------------------------+
# | CLASSIFICATION METHODS |
# +------------------------+

def logistic_regression(x_training, y_training, x_test, y_test):
    lr = LogisticRegression(fit_intercept=True, tol=0.001, max_iter=10000000)

    lr_params = {
        'C': [0.1, 1, 10, 100, 1000],
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'solver': ['lbfgs', 'sag', 'saga']
    }

    lr_gs = GridSearchCV(estimator=lr, param_grid=lr_params, n_jobs=-1)  # Busca els millors hiperparàmetres pel LR
    start = time.time()
    lr_model = lr_gs.fit(x_training, y_training)  # Entrena el model
    end = time.time()
    print("LR training time: {}".format(end-start))
    dump(lr_model, '../models/logistic_regressor.joblib')

    print("LR Best Params: {}".format(lr_gs.best_params_))
    print("LR Training score with best params: {}".format(lr_gs.best_estimator_.score(x_training, y_training)))
    print("LR Test score with best params: {}".format(lr_gs.best_estimator_.score(x_test, y_test)))

    lr_preds = lr_gs.best_estimator_.predict(x_test)
    print("LR prediction metrics: \n{}".format(metrics.classification_report(y_true=y_test, y_pred=lr_preds)))

    create_confusion_matrix(y_test, lr_preds, "Logistic Regressor")

# logistic_regression(x_t_norm, y_t, x_v_norm, y_v)


def svm_classifier(x_training, y_training, x_test, y_test):
    svc = svm.SVC(probability=True)

    svc_params = {
        'C': [0.1, 1, 10, 100, 1000],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': [3, 4, 5]
    }

    svc_gs = GridSearchCV(estimator=svc, param_grid=svc_params, n_jobs=-1)  # Busca els millors hiperparàmetres pel LR
    start = time.time()
    svc_model = svc_gs.fit(x_training, y_training)  # Entrena el model
    end = time.time()
    print("SVC training time: {}".format(end - start))
    dump(svc_model, '../models/support_vectors_classifier.joblib')

    print("SVC Best Params: {}".format(svc_gs.best_params_))
    print("SVC Training score with best params: {}".format(svc_gs.best_estimator_.score(x_training, y_training)))
    print("SVC Test score with best params: {}".format(svc_gs.best_estimator_.score(x_test, y_test)))

    svc_preds = svc_gs.best_estimator_.predict(x_test)
    print("SVC prediction metrics: \n{}".format(metrics.classification_report(y_true=y_test, y_pred=svc_preds)))

    create_confusion_matrix(y_test, svc_preds, "Support Vectors Classifier")

# svm_classifier(x_t_norm, y_t, x_v_norm, y_v)


def knn_classifier(x_training, y_training, x_test, y_test):
    knn = KNeighborsClassifier(algorithm='auto')

    knn_params = {
        'n_neighbors': [2, 5, 10, 20, 40, 80],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }

    knn_gs = GridSearchCV(estimator=knn, param_grid=knn_params, n_jobs=-1)  # Busca els millors hiperparàmetres pel LR
    start = time.time()
    knn_model = knn_gs.fit(x_training, y_training)  # Entrena el model
    end = time.time()
    print("KNN training time: {}".format(end - start))
    dump(knn_model, '../models/k_nearest_neighbors_classifier.joblib')

    print("KNN Best Params: {}".format(knn_gs.best_params_))
    print("KNN Training score with best params: {}".format(knn_gs.best_estimator_.score(x_training, y_training)))
    print("KNN Test score with best params: {}".format(knn_gs.best_estimator_.score(x_test, y_test)))

    knn_preds = knn_gs.best_estimator_.predict(x_test)
    print("KNN prediction metrics: \n{}".format(metrics.classification_report(y_true=y_test, y_pred=knn_preds)))

    create_confusion_matrix(y_test, knn_preds, "K Nearest Neighbors Classifier")

# knn_classifier(x_t_norm, y_t, x_v_norm, y_v)


def random_forest_classifier(x_training, y_training, x_test, y_test):
    rfc = RandomForestClassifier()

    rfc_params = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [4, 5, 6, 7, 8],
        'criterion': ['gini', 'entropy']
    }

    rfc_gs = GridSearchCV(estimator=rfc, param_grid=rfc_params, n_jobs=-1)  # Busca els millors hiperparàmetres pel LR
    start = time.time()
    rfc_model = rfc_gs.fit(x_training, y_training)  # Entrena el model
    end = time.time()
    print("RFC training time: {}".format(end - start))
    dump(rfc_model, '../models/random_forest_classifier.joblib')

    print("RFC Best Params: {}".format(rfc_gs.best_params_))
    print("RFC Training score with best params: {}".format(rfc_gs.best_estimator_.score(x_training, y_training)))
    print("RFC Test score with best params: {}".format(rfc_gs.best_estimator_.score(x_test, y_test)))

    rfc_preds = rfc_gs.best_estimator_.predict(x_test)
    print("RFC prediction metrics: \n{}".format(metrics.classification_report(y_true=y_test, y_pred=rfc_preds)))

    create_confusion_matrix(y_test, rfc_preds, "Random Forest Classifier")

# random_forest_classifier(x_t_norm, y_t, x_v_norm, y_v)