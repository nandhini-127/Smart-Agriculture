import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_excel("Smart_agriculture_dataset.xlsx")

# Features and target
X = df[["Temperature", "Humidity", "SoilMoisture", "Motion_Detected"]]
y = df["Irrigation_Needed"]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (optional but recommended for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------ Logistic Regression ------------------
log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)
y_pred_log = log_model.predict(X_test_scaled)

# Logistic Regression results
log_accuracy = accuracy_score(y_test, y_pred_log)
print("Logistic Regression Accuracy:", log_accuracy)
print(confusion_matrix(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

# Plot confusion matrix as a heatmap
cm_log = confusion_matrix(y_test, y_pred_log)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_log, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ------------------ Naive Bayes ------------------
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)  # NB doesn't need scaling
y_pred_nb = nb_model.predict(X_test)

# Naive Bayes results
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print(confusion_matrix(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))



# ------------------ Decision Tree ------------------
tree_model = DecisionTreeClassifier(random_state=42, max_depth=6)  # Adjust max_depth for better visualization
tree_model.fit(X_train_scaled, y_train)
y_pred_tree = tree_model.predict(X_test_scaled)

# Decision Tree results
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_tree))
print(confusion_matrix(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))

# ------------------ Plot the Decision Tree ------------------
plt.figure(figsize=(15, 10))  # Increase size for better clarity
plot_tree(tree_model, 
          filled=True, 
          feature_names=["Temperature", "Humidity", "SoilMoisture", "Motion_Detected"], 
          class_names=["No", "Yes"], 
          rounded=True, 
          fontsize=12)
plt.title("Decision Tree Visualization - Smart Agriculture", fontsize=16)
plt.show()

# ------------------ Feature Importance ------------------
print("Feature importances:", tree_model.feature_importances_)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.bar(["Temperature", "Humidity", "SoilMoisture", "Motion_Detected"], tree_model.feature_importances_, color='teal')
plt.title("Feature Importance - Decision Tree", fontsize=14)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()


# ------------------ Logistic Regression Line Plot ------------------
# To visualize the Logistic Regression accuracy, we will create a line plot
accuracy_data = {
    'Model': ['Logistic Regression'],
    'Accuracy': [log_accuracy]
}

accuracy_df = pd.DataFrame(accuracy_data)




from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ------------------ Linear Regression ------------------
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)  # Fit model
y_pred_lin = lin_reg.predict(X_test_scaled)  # Predict continuous values (probabilities)

# Optional: convert continuous predictions to 0 or 1 using threshold
y_pred_lin_class = [1 if prob >= 0.5 else 0 for prob in y_pred_lin]

# Linear Regression Evaluation
lin_mse = mean_squared_error(y_test, y_pred_lin)
lin_r2 = r2_score(y_test, y_pred_lin)

print("Linear Regression Mean Squared Error:", lin_mse)
print("Linear Regression R² Score:", lin_r2)
print("Linear Regression Accuracy (rounded):", accuracy_score(y_test, y_pred_lin_class))

# ------------------ Plot: Actual vs Predicted (Linear Regression) ------------------
plt.figure(figsize=(10, 6))
plt.plot(y_test.values[:50], label='Actual', marker='o', linestyle='-', color='green')
plt.plot(y_pred_lin[:50], label='Predicted (Probabilities)', marker='x', linestyle='--', color='orange')
plt.title("Linear Regression: Actual vs Predicted Probabilities")
plt.xlabel("Index")
plt.ylabel("Irrigation Needed (0=No, 1=Yes)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
