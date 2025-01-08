from Data_Preprocessing import *
X = df.drop(columns=['Churn'])
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=27, stratify=y)
#print(f'Training Set Size: {X_train.shape}')
#print(f'Test Set Size: {X_test.shape}')
rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)*100
print('Accuracy:')
print((str(round(accuracy,2)))+'%')