from RandomForest import *
from Importing_Libraries import *
param_dist = {
    'n_estimators': randint(100,200),
    'max_depth': [30],
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 5),
}
random_search = RandomizedSearchCV(estimator=rf_model, param_distributions=param_dist,
                                   n_iter=50, scoring='accuracy', cv=5,
                                   verbose=2, random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)
best_score = random_search.best_score_*100
print("Best Parameters:", random_search.best_params_)
print("Best Accuracy:", best_score)