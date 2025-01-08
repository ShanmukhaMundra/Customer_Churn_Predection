from RandomForest import *
from Importing_Libraries import *
param_grid = {
    'n_estimators': [100],
    'max_depth': [30],
    'min_samples_split': [2],
    'min_samples_leaf': [2],
}
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid,
                           scoring='accuracy', cv=5, verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_score = grid_search.best_score_*100
print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy:", best_score)