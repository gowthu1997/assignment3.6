import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load Titanic dataset
data = pd.read_csv('train.csv')
os.makedirs('plots', exist_ok=True)



# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)



num_estimators = [50, 100, 150]

for n_estimators in num_estimators:
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    # Evaluate the model and save performance metrics or plots (if needed)

    # Save the plot to the 'plots' directory
    plt.figure(figsize=(8, 6))
    sns.barplot(x=class_distribution.index, y=class_distribution.values)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title(f'Passenger Class Distribution (n_estimators={n_estimators})')
    plt.savefig(f'plots/class_distribution_{n_estimators}.png')
    plt.close()