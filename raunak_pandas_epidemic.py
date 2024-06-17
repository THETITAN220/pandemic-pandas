import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv('your_dataset.csv')

# Data preprocessing
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek

# Feature engineering
df['cases_7day_avg'] = df['number of cases'].rolling(window=7).mean()
df['deaths_7day_avg'] = df['number of deaths'].rolling(window=7).mean()

# Drop rows with NaN values created by rolling average
df = df.dropna()

# Define the target variable (example criteria)
def classify_status(row):
    if row['number of cases'] < 50 and row['number of deaths'] < 5:
        return 'normal conditions'
    elif row['number of cases'] >= 50 and row['number of cases'] < 200:
        return 'emergence'
    elif row['number of cases'] >= 200 and row['number of cases'] < 1000:
        return 'epidemic'
    else:
        return 'pandemic'

df['status'] = df.apply(classify_status, axis=1)

# Define features and target variable
features = df[['number of cases', 'number of deaths', 'cases_7day_avg', 'deaths_7day_avg', 'month', 'day', 'dayofweek']]
target = df['status']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
