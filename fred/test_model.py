import catboost as cb
from catboost import Pool
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform


def test_model(X_train, y_train,number):
    # Définition de la grille des hyperparamètres
    grid = {'learning_rate': [0.01, 0.1,0.001],
        'depth': [4, 6, 10,20],
        'l2_leaf_reg': [1, 3, 5,10],
        'iterations': [50, 100, 150,200]}
    model = f'model_{number}'
            
    # Utilisation de RandomizedSearchCV avec CatBoost sans Pool
    clf = RandomizedSearchCV(model, grid, random_state=0, n_iter=100, cv=3, scoring='F1')
    search = clf.fit(X_train, y_train)

    best_estimator = search.best_estimator_
    best_param = search.best_params_
    print("les meilleurs paramètres CatBoost sont : ", best_param)

         