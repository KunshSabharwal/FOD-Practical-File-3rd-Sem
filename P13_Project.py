#1. DATA COLLECTION


# Import necessary libraries
import pandas as pd

# Load the dataset (for example, from a CSV file)
data = pd.read_csv('creditcard.csv')

# View the first few rows of the dataset
data.head()

#-----------------
#2. DATA CLEANING


# Check for missing values
missing_values = data.isnull().sum()

# If missing values exist, handle them (e.g., fill with mean, median, etc.)
data.fillna(data.mean(), inplace=True)

# Check for duplicates and remove them
data.drop_duplicates(inplace=True)

#------------------
#3. DATA EXPLORATION(EDA)


# Basic statistics
data.describe()

# Visualize class distribution (fraudulent vs non-fraudulent transactions)
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=data)
plt.title('Class Distribution: Fraud vs Non-Fraud')
plt.show()

# Check correlation between features
plt.figure(figsize=(12, 10))
sns.heatmap(data.corr(), cmap="coolwarm", annot=False)
plt.title('Feature Correlation Heatmap')
plt.show()

#-------------------
#4. FEATURE ENGINEERING


# Example of feature scaling (normalization) to ensure the features are on a
# similar scale
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data['Amount_scaled'] = scaler.fit_transform(data[['Amount']])

#-------------------
#5. MODEL BUILDING


from sklearn.model_selection import train_test_split

# Define features (X) and target (y)
X = data.drop(columns=['Class', 'Amount'])
y = data['Class']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Import a machine learning algorithm (e.g., Random Forest Classifier)
from sklearn.ensemble import RandomForestClassifier

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model to the training data
model.fit(X_train, y_train)


#-------------------
#6. MODEL EVALUATION


from sklearn.metrics import classification_report, confusion_matrix

# Predict the test data
y_pred = model.predict(X_test)

# Display classification report
print(classification_report(y_test, y_pred))

# Display confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.show()


#--------------------
#7. MODEL TUNING


from sklearn.model_selection import GridSearchCV

# Define hyperparameters to tune
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30]
}

# Initialize GridSearch
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')

# Fit GridSearch to the training data
grid_search.fit(X_train, y_train)

# Get best parameters
print("Best Parameters: ", grid_search.best_params_)

