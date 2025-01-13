from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def train_svm_model(X_train, y_train):
    param_grid = {
        'svc__C': [0.1, 1, 10],
        'svc__kernel': ['linear', 'rbf'],
        'svc__gamma': [1, 0.1, 0.01, 0.001]
    }

    pipeline = Pipeline([
        ('scaler', StandardScaler(with_mean=False)),
        ('svc', SVC())
    ])

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='f1_macro',
        verbose=2,
        n_jobs=1
    )

    print("Starting GridSearchCV for hyperparameter tuning...")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    print(f"Best parameters found: {grid_search.best_params_}")

    return best_model