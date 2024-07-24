from xgboost import XGBClassifier
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve
import shap
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
random.seed(31)




# load data
GeMAPS_demo_df = pd.read_csv("GeMAPS_demo_df2.csv")
print(GeMAPS_demo_df.shape)
print(GeMAPS_demo_df["depression"].sum()) # There are 28 depression cases
print(GeMAPS_demo_df.columns[:10])
#GeMAPS_demo_df = GeMAPS_demo_df.dropna(subset=['depression'])

# split data into features and target
X = GeMAPS_demo_df.drop(columns=["record_id", "Unnamed: 0", "depression"])
y = GeMAPS_demo_df["depression"]


# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.2, random_state=42)

# Calculate the scale_pos_weight value
negative_class_count = np.sum(y_train == 0)
positive_class_count = np.sum(y_train == 1)
scale_pos_weight = negative_class_count / positive_class_count


# define model
model = XGBClassifier(scale_pos_weight=scale_pos_weight)

# define grid search
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.1, 0.01, 0.001],
}

# Define stratified K-Fold cross-validation
stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# perform grid search
grid_search = GridSearchCV(model, param_grid, cv=stratified_kfold, n_jobs=-1)
grid_search.fit(
    X_train, 
    y_train
)

# get best parameters
xgb_best_model = grid_search.best_estimator_
print(grid_search.best_params_)

# make predictions
y_pred = xgb_best_model.predict(X_test)
y_train_pred = xgb_best_model.predict(X_train)


# evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# calculate auroc
y_pred_prob = xgb_best_model.predict_proba(X_test)[:, 1]
auroc = roc_auc_score(y_test, y_pred_prob)
print(f"AUROC: {auroc}")

# calculate roc curves
xgb_fpr, xgb_tpr, _ = roc_curve(y_test,y_pred_prob)

# plot the roc curve for the model
plt.plot(xgb_fpr, xgb_tpr, marker='.', label='XGBoost')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.savefig('xgb_auroc.pdf', dpi=300)
plt.show()

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

print("Classification Report:")
print(classification_report(y_test, y_pred))

# save accuracy, classification report and confusion matrix
with open("xgb_accuracy.txt", "w") as f:
    f.write(f"Accuracy: {accuracy}")

with open("xgb_classification_report.txt", "w") as f:
    f.write(classification_report(y_test, y_pred))


conf_matrix_df = pd.DataFrame(conf_matrix)
conf_matrix_df.to_csv("confusion_matrix.csv", index=False)


# evaluate training performance
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Train Accuracy: {train_accuracy}")

# calculate auroc
y_pred_prob_train = xgb_best_model.predict_proba(X_train)[:, 1]
train_auroc = roc_auc_score(y_train, y_pred_prob_train)
print(f"Train AUROC: {train_auroc}")


# Calculate SHAP values
explainer = shap.TreeExplainer(xgb_best_model)
shap_values = explainer.shap_values(X)

# Create a summary plot of the SHAP values
plt.figure()
shap.summary_plot(shap_values, X, max_display=10)

# Save the summary plot as a PNG file
plt.savefig('shap_summary_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# Get the mean absolute SHAP values for each feature
shap_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': np.abs(shap_values).mean(axis=0)
})

# Sort by importance
shap_importance = shap_importance.sort_values(by='Importance', ascending=False)

# Print the top 10 features
print(shap_importance.head(10))