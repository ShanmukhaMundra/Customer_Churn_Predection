from Importing_Libraries import *
data = pd.read_csv('/Users/shanmukhamundra/desktop/Telco_Churn.csv')
df = data.drop(columns=['customerID','gender', 'Contract', 'SeniorCitizen',
                        'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                        'InternetService', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                        'PaperlessBilling', 'PaymentMethod'])
df = df
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
df['tenure'] = df['tenure'].astype(float)
le = LabelEncoder()
binary_columns = ['Churn']
for column in binary_columns:
    df[column] = le.fit_transform(df[column])
print(df['Churn'].value_counts())