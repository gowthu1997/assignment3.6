# regression_script.py

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import os
# Load the dataset
df = pd.read_csv('50_Startups.csv')
os.makedirs('plots', exist_ok=True)
state_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Define a column transformer to apply transformations to specific columns
preprocessor = ColumnTransformer(
    transformers=[
        ('state', state_transformer, ['State'])
    ],
    remainder='passthrough'
)

# Create a pipeline for preprocessing and modeling
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])
# Function to fit and plot regression with given test size
def fit_and_plot(test_size):
    # Split data into training and testing sets
    X = df[['R&D Spend', 'Administration', 'Marketing Spend', 'State']]
    y = df['Profit']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Fit a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Generate a plot
    plt.scatter(y_test, model.predict(X_test))
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Actual vs Predicted (Test Size: {test_size})')
    plt.savefig(f'plots/plot_test_size_{test_size}.png')
    plt.close()

# Fit and plot for different test set sizes
test_sizes = [0.2, 0.3, 0.4]  # Define your test set sizes here
for test_size in test_sizes:
    fit_and_plot(test_size)
