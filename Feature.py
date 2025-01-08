from RandomForest import *
feature_importance = rf_model.feature_importances_
sns.pointplot(x=X.columns, y=feature_importance)
plt.title('Feature Importance')
plt.show()