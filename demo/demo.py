from joblib import load
from sklearn import metrics
import src.main as main

# ATENCIÓ: descomentar les següents 5 línies si es vol executar des del projecte amb un IDE (PyCharm o altres).
"""
demo_dataset = main.load_dataset_from_csv('../data/demo_data.csv')  # Per executar des del projecte
lr = load('../models/logistic_regressor.joblib')  # Per executar des del projecte
svc = load('../models/support_vectors_classifier.joblib')  # Per executar des del projecte
knn = load('../models/k_nearest_neighbors_classifier.joblib')  # Per executar des del projecte
rfc = load('../models/random_forest_classifier.joblib')  # Per executar des del projecte
"""

# ATENCIÓ: comentar les següents 5 línies si es vol executar des del projecte amb un IDE (PyCharm o altres).
demo_dataset = main.load_dataset_from_csv('data/demo_data.csv')  # Per executar des del terminal
lr = load('models/logistic_regressor.joblib')  # Per executar des del terminal
svc = load('models/support_vectors_classifier.joblib')  # Per executar des del terminal
knn = load('models/k_nearest_neighbors_classifier.joblib')  # Per executar des del terminal
rfc = load('models/random_forest_classifier.joblib')  # Per executar des del terminal

demo_dataset = main.remove_spaces(demo_dataset)
demo_dataset = main.remove_columns(demo_dataset)
demo_dataset = main.nan_treatment(demo_dataset)
X = demo_dataset.values[:, 1:]  # Característiques
Y = demo_dataset.values[:, 0]  # TargeT

lr_preds = lr.predict(X)
print("LR prediction metrics: {}".format(metrics.classification_report(y_true=Y, y_pred=lr_preds)))

svc_preds = svc.predict(X)
print("SVC prediction metrics: {}".format(metrics.classification_report(y_true=Y, y_pred=svc_preds)))

knn_preds = knn.predict(X)
print("KNN prediction metrics: {}".format(metrics.classification_report(y_true=Y, y_pred=knn_preds)))

rfc_preds = rfc.predict(X)
print("RFC prediction metrics: {}".format(metrics.classification_report(y_true=Y, y_pred=rfc_preds)))

# ATENCIÓ: DESCOMENTAR EL SEGÜENT BLOC SI ES VOL ENTRENAR ELS MODELS AMB LES DADES ORIGINALS DEL DATASET
"""
bkpcy_full_dataset = main.load_dataset_from_csv('../data/archive/data.csv')
main.print_head(bkpcy_full_dataset)
main.print_data_types(bkpcy_full_dataset)
main.print_data_statistics(bkpcy_full_dataset)
main.df_dimensionality(bkpcy_full_dataset)
main.y_balance(bkpcy_full_dataset)
main.pearson_correlation(bkpcy_full_dataset)

bkpcy_dt = main.remove_spaces(bkpcy_full_dataset)
bkpcy_dt = main.remove_columns(bkpcy_dt)
bkpcy_dt = main.nan_treatment(bkpcy_dt)

# Entrenament i exportació dels models
x_t, x_v, y_t, y_v = main.split_data(bkpcy_dt.values)
x_t_norm, x_v_norm = main.standardize_data(x_t, x_v)
main.logistic_regression(x_t_norm, y_t, x_v_norm, y_v)
main.svm_classifier(x_t_norm, y_t, x_v_norm, y_v)
main.knn_classifier(x_t_norm, y_t, x_v_norm, y_v)
main.random_forest_classifier(x_t_norm, y_t, x_v_norm, y_v)
"""