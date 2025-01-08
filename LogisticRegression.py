from Data_Preprocessing import *
X = df.drop(columns=['Churn'])
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=27)
#print(X_train.shape, X_test.shape)
X_train = X_train.dropna()
y_train = y_train[X_train.index]
X_test = X_test.dropna()
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)*100
print('Accuracy:')
print((str(round(accuracy,2)))+'%')