import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os

# 1. Ensure directories exist
os.makedirs('static', exist_ok=True)
os.makedirs('dataset', exist_ok=True)

# 2. Load dataset
print("Loading dataset...")
df = pd.read_csv('dataset/Crop_recommendation.csv')

# 3. Handle missing values
print(f"Missing values:\n{df.isnull().sum()}")

# 4. EDA - Heatmap
plt.figure(figsize=(10, 8))
numeric_cols = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('static/correlation_heatmap.png')
plt.close()

# 5. Extract Features and Target
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# 6. Train-Test Split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Model Building
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42)
}

accuracies = {}
best_model = None
best_acc = 0

print("\nTraining models...")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")
    
    if acc > best_acc:
        best_acc = acc
        best_model = model

# 8. Visualization - Accuracy Comparison
plt.figure(figsize=(10, 6))
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), hue=list(accuracies.keys()), palette='viridis', legend=False)
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1.1)
for i, v in enumerate(accuracies.values()):
    plt.text(i, v + 0.02, str(round(v, 4)), ha='center')
plt.savefig('static/accuracy_comparison.png')
plt.close()

# 9. Best Model Deep Dive
y_pred_best = best_model.predict(X_test)
print("\nBest Model Classification Report:")
print(classification_report(y_test, y_pred_best))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=best_model.classes_, 
            yticklabels=best_model.classes_)
plt.title(f'Confusion Matrix ({best_model.__class__.__name__})')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('static/confusion_matrix.png')
plt.close()

# Feature Importance
if hasattr(best_model, 'feature_importances_'):
    plt.figure(figsize=(8, 5))
    sns.barplot(x=X.columns, y=best_model.feature_importances_, hue=X.columns, palette='magma', legend=False)
    plt.title('Feature Importance')
    plt.savefig('static/feature_importance.png')
    plt.close()

# 10. Save the best model
with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print(f"\nBest model ({best_model.__class__.__name__}) saved as model.pkl")
