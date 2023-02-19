import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer
from tensorflow.keras.datasets import mnist

# Download and reshape data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
x = np.concatenate((x_train, x_test))[:6000]
y = np.concatenate((y_train, y_test))[:6000]

# Split into datasets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=40)

# Normalize and transform features
transformer = Normalizer()
x_train_norm = transformer.transform(x_train)
x_test_norm = transformer.transform(x_test)


# function to work with multiple models
def fit_predict_eval(model, features_train, features_test, target_train, target_test, best_estimator):
    # fit the model
    model = model.fit(features_train, target_train)
    # make a prediction
    y_pred = model.predict(features_test)
    # calculate accuracy
    score = accuracy_score(target_test, y_pred)

    print(
        f"{'K-nearest neighbours algorithm' if model.__class__.__name__.startswith('K') else 'Random forest algorithm'}"
        f"\nbest estimator: {model.__class__.__name__}{str(best_estimator).replace('{', '(').replace('}', ')')}"
        f"\naccuracy: {round(score, 3)}\n")


# getting the best parameters in K-nearest neighbours model
knn_estimator = KNeighborsClassifier()
knn_param_grid = dict(n_neighbors=[3, 4], weights=['uniform', 'distance'], algorithm=['auto', 'brute'])
gs_knn = GridSearchCV(estimator=knn_estimator, param_grid=knn_param_grid, scoring='accuracy', n_jobs=-1)
gs_knn.fit(x_train_norm, y_train)
knn_param_grid = gs_knn.best_params_

# apply function to model with normalized data
fit_predict_eval(
    model=KNeighborsClassifier(n_neighbors=knn_param_grid['n_neighbors'], weights=knn_param_grid['weights'],
                               algorithm=knn_param_grid['algorithm']),
    features_train=x_train_norm,
    features_test=x_test_norm,
    target_train=y_train,
    target_test=y_test,
    best_estimator=knn_param_grid
)

# getting the best parameters in Random Forest model
rf_estimator = RandomForestClassifier(random_state=40)
rf_param_grid = dict(n_estimators=[300, 500], max_features=['auto', 'log2'],
                     class_weight=['balanced', 'balanced_subsample'])
gs_rf = GridSearchCV(estimator=rf_estimator, param_grid=rf_param_grid, scoring='accuracy', n_jobs=-1)
gs_rf.fit(x_train_norm, y_train)
rf_param_grid = gs_rf.best_params_

# apply function to model with normalized data
fit_predict_eval(
    model=RandomForestClassifier(n_estimators=rf_param_grid['n_estimators'],
                                 max_features=rf_param_grid['max_features'],
                                 class_weight=rf_param_grid['class_weight'],
                                 random_state=40),
    features_train=x_train_norm,
    features_test=x_test_norm,
    target_train=y_train,
    target_test=y_test,
    best_estimator=rf_param_grid
)
