import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
 
sns.set_style("whitegrid")
 
 
iris = load_iris()
 
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target                                   # 0, 1, 2
df["species_name"] = df["species"].map(dict(enumerate(iris.target_names)))  # setosa/versicolor/virginica
 
print("First 5 rows of the dataset:")
print(df.head())
 
print("\nShape (rows, columns):", df.shape)
print("\nColumn names:", list(df.columns))
print("\nData types:\n", df.dtypes)
 
print("\nSpecies classes:", list(iris.target_names))
print("\nClass distribution:\n", df["species_name"].value_counts())
 
print("\nSummary statistics:\n", df.describe())
 
 
 
X = df[iris.feature_names]   # input: sepal/petal length & width
y = df["species"]            # target: species label (0/1/2)
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
 
print(f"\nTraining samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
 
# Train a classification model (Logistic Regression -- simple and effective baseline)
model = LogisticRegression(max_iter=200)
model.fit(X_train_scaled, y_train)
 
print("\nModel trained successfully:", model)
 
 
 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
 
models = {
    "Logistic Regression": model,  # reuse the one trained above
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Support Vector Machine": SVC(kernel="rbf"),
}
 
# Fit the remaining models
for name, m in models.items():
    if name != "Logistic Regression":
        m.fit(X_train_scaled, y_train)
 
 
print("\n" + "=" * 60)
print("MODEL EVALUATION ON TEST DATA")
print("=" * 60)
 
results = []
best_name, best_acc, best_model = None, 0, None
 
for name, m in models.items():
    y_pred = m.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
 
    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
 
    results.append({"Model": name, "Accuracy": round(acc, 4)})
 
    if acc > best_acc:
        best_acc, best_name, best_model = acc, name, m
 
results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
print("\nModel comparison:\n", results_df)
 
# Confusion matrix for the best-performing model
y_pred_best = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred_best)
 
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title(f"Confusion Matrix - {best_name} (Best Model)")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.show()
 
 
sns.pairplot(df, hue="species_name", vars=iris.feature_names, palette="husl")
plt.suptitle("Pairwise Feature Relationships by Species", y=1.02)
plt.show()
 
plt.figure(figsize=(6, 5))
sns.heatmap(df[iris.feature_names].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()
 
print(f"\nBest Model: {best_name} with Test Accuracy: {best_acc:.4f}")
print("""
Key classification concepts demonstrated:
- Features (X) vs. Target/Labels (y)
- Train/test split to evaluate generalization
- Feature scaling for distance/margin-based models
- Multiple algorithms compared on the same data
- Accuracy, precision, recall, F1-score as evaluation metrics
- Confusion matrix to see which classes get confused
""")