from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
import shap
import random
import numpy as np
random.seed(31)




# load data
GeMAPS_demo_df = pd.read_csv("GeMAPS_demo_df2.csv")
print(GeMAPS_demo_df.shape)
print(GeMAPS_demo_df["depression"].sum()) # There are 28 depression cases
#GeMAPS_demo_df = GeMAPS_demo_df.dropna(subset=['depression'])

# split data into features and target
X = GeMAPS_demo_df.drop(columns=["record_id", "depression"])
y = GeMAPS_demo_df["depression"]


# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Calculate the scale_pos_weight value
# Calculate the class weights
class_weights = {0: np.sum(y_train == 1) / len(y_train), 
                 1: np.sum(y_train == 0) / len(y_train)}


# define model
model = RandomForestClassifier(class_weight=class_weights, random_state=42)

# define grid search
param_grid = {
    "n_estimators": [25, 50, 75],
    "max_depth": [3],
}

# Define stratified K-Fold cross-validation
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# perform grid search
grid_search = GridSearchCV(model, param_grid, cv=kfold, n_jobs=-1)
grid_search.fit(
    X_train, 
    y_train
)

# get best parameters
rf_best_model = grid_search.best_estimator_
print(grid_search.best_params_)

# make predictions
y_pred = rf_best_model.predict(X_test)
y_train_pred = rf_best_model.predict(X_train)


# evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# calculate auroc
y_pred_prob = rf_best_model.predict_proba(X_test)[:, 1]
auroc = roc_auc_score(y_test, y_pred_prob)
print(f"AUROC: {auroc}")

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

print("Classification Report:")
print(classification_report(y_test, y_pred))

# save accuracy, classification report and confusion matrix
with open("rf_accuracy.txt", "w") as f:
    f.write(f"Accuracy: {accuracy}")

with open("rf_classification_report.txt", "w") as f:
    f.write(classification_report(y_test, y_pred))


conf_matrix_df = pd.DataFrame(conf_matrix)
conf_matrix_df.to_csv("rf_confusion_matrix.csv", index=False)

# Accuracy - 0.857, AUROC-0.9, confusion matrix - 15, 0, 3, 3


# evaluate training performance
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Train Accuracy: {train_accuracy}")

# calculate auroc
y_pred_prob_train = rf_best_model.predict_proba(X_train)[:, 1]
train_auroc = roc_auc_score(y_train, y_pred_prob_train)
print(f"Train AUROC: {train_auroc}")


# Lowering the threshold
threshold = 0.25
predicted_proba = rf_best_model.predict_proba(X_test)
predicted = (predicted_proba [:,1] >= threshold).astype('int')
threshold_accuracy = accuracy_score(y_test, predicted)
print(f"Threshold accuracy: {threshold_accuracy}")


conf_matrix = confusion_matrix(y_test, predicted)
print("Threshold Confusion Matrix:")
print(conf_matrix)

print("Threshold Classification Report:")
print(classification_report(y_test, predicted))

# Can get confusion matrix this way - 11, 4, 0, 6


# obtain shap summary plot
explainer = shap.Explainer(rf_best_model)
shap_values = explainer(X_train)
#shap.summary_plot(shap_values, X)
