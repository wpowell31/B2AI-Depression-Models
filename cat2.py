from catboost import CatBoostClassifier
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
import shap
import random
from sklearn.model_selection import StratifiedKFold
import numpy as np
random.seed(31)




# load data
GeMAPS_demo_df = pd.read_csv("GeMAPS_demo_df2.csv")
print(GeMAPS_demo_df.shape)
#GeMAPS_demo_df = GeMAPS_demo_df.dropna(subset=['depression'])

# split data into features and target
X = GeMAPS_demo_df.drop(columns=["record_id", "Unnamed: 0", "depression"])
y = GeMAPS_demo_df["depression"]


# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)


# define model
model = CatBoostClassifier()

# define grid search
param_grid = {
    "n_estimators": [100, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.1, 0.01, 0.001]
}

kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# perform grid search
grid_search = GridSearchCV(model, param_grid, cv=kfold, n_jobs=-1)
grid_search.fit(
    X_train, 
    y_train
)

# get best parameters
cat_best_model = grid_search.best_estimator_

# make predictions
y_pred = cat_best_model.predict(X_test)
y_train_pred = cat_best_model.predict(X_train)


# evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# calculate auroc
y_pred_prob = cat_best_model.predict_proba(X_test)[:, 1]
auroc = roc_auc_score(y_test, y_pred_prob)
print(f"AUROC: {auroc}")

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

print("Classification Report:")
print(classification_report(y_test, y_pred))

# save accuracy, classification report and confusion matrix
with open("catboost_accuracy.txt", "w") as f:
    f.write(f"Accuracy: {accuracy}")

with open("catboost_classification_report.txt", "w") as f:
    f.write(classification_report(y_test, y_pred))


conf_matrix_df = pd.DataFrame(conf_matrix)
conf_matrix_df.to_csv("confusion_matrix.csv", index=False)


# evaluate training performance
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Train Accuracy: {train_accuracy}")

# calculate auroc
y_pred_prob_train = cat_best_model.predict_proba(X_train)[:, 1]
train_auroc = roc_auc_score(y_train, y_pred_prob_train)
print(f"Train AUROC: {train_auroc}")


# accuracy - 0.904
# AUROC - 0.9117
# F1 macro avg - 0.81
# 17, 0, 2, 2 confusion matrix

# Calculate SHAP values
explainer = shap.TreeExplainer(cat_best_model)
shap_values = explainer.shap_values(X)

# Create a summary plot of the SHAP values
shap.summary_plot(shap_values, X, max_display=10)


# Get the mean absolute SHAP values for each feature
shap_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': np.abs(shap_values).mean(axis=0)
})

# Sort by importance
shap_importance = shap_importance.sort_values(by='Importance', ascending=False)

# Print the top 10 features
print(shap_importance.head(10))