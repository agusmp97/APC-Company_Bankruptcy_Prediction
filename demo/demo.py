from joblib import load
from sklearn import metrics

import src.main as main

demo_dataset = main.load_dataset_from_csv('../data/demo_data.csv')

demo_dataset = main.remove_spaces(demo_dataset)
demo_dataset = main.remove_columns(demo_dataset)
demo_dataset = main.nan_treatment(demo_dataset)
X = demo_dataset.values[:, 1:]  # Característiques
Y = demo_dataset.values[:, 0]  # Target

lr = load('../models/logistic_regressor.joblib')
#print(lr.best_estimator_.coef_)
lr_preds = lr.predict(X)

print("LR prediction metrics: {}".format(metrics.classification_report(y_true=Y, y_pred=lr_preds)))

x=3
# SI ES VOL ENTRENAR EL MODEL AMB LES DADES ORIGINALS DEL DATASET, DESCOMENTAR EL SEGÜENT BLOC
"""
bkpcy_full_dataset = main.load_dataset_from_csv('../archive/data.csv')
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