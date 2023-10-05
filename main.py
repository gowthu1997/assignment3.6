import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Load Titanic dataset
data = pd.read_csv('train.csv')
os.makedirs('plots', exist_ok=True)

# Preprocess the dataset (feature engineering, etc.)
# For simplicity, we'll use Pclass as a feature and Survived as the label
features = data[['Pclass']]
labels = data['Survived']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Create a bar plot showing the distribution of passengers by class
class_distribution = data['Pclass'].value_counts()
plt.figure(figsize=(8, 6))
sns.barplot(x=class_distribution.index, y=class_distribution.values)
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Passenger Class Distribution')
plt.savefig('plots/class_distribution.png')

# Train the model with different learning rates (dummy example)
learning_rates = [0.01, 0.1, 0.2]

for lr in learning_rates:
    model = RandomForestClassifier(learning_rate=lr)
    model.fit(X_train, y_train)
    # Evaluate the model and save performance metrics or plots (if needed)

# Save plots
# For simplicity, let's assume you generate plots and save them in the 'plots' directory
# Adjust this based on how you generate plots in your specific use case
# Ensure the 'plots' directory exists before running the script
# ...

plt.plot([1, 2, 3], [1, 2, 3])
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Example Plot')
plt.savefig('plots/example_plot.png')
