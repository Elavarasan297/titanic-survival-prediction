# %% [markdown]
# # Titanic Survival Prediction — End-to-End Data Science Project
#
# **Goal:** Predict whether a passenger survived the Titanic disaster using their
# demographic and travel information (class, age, sex, fare, family size, etc.)
#
# **Workflow:**
# 1. Load and inspect the data
# 2. Exploratory Data Analysis (EDA)
# 3. Data cleaning (handle missing values)
# 4. Feature engineering
# 5. Train/test split
# 6. Model training (Logistic Regression + Random Forest)
# 7. Evaluation and comparison
# 8. Feature importance

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100

# %% [markdown]
# ## 1. Load and inspect the data

# %%
df = pd.read_csv("data/titanic.csv")
print("Shape:", df.shape)
df.head()

# %%
df.info()

# %%
df.isnull().sum().sort_values(ascending=False)

# %% [markdown]
# **Observations:**
# - `Cabin` is missing for the majority of passengers — too sparse to use reliably.
# - `Age` has a meaningful number of missing values — worth imputing rather than dropping.
# - `Embarked` has only a couple of missing values.

# %% [markdown]
# ## 2. Exploratory Data Analysis

# %%
overall_rate = df["Survived"].mean()
print(f"Overall survival rate: {overall_rate:.1%}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

sns.barplot(data=df, x="Sex", y="Survived", ax=axes[0])
axes[0].set_title("Survival Rate by Sex")
axes[0].set_ylabel("Survival Rate")

sns.barplot(data=df, x="Pclass", y="Survived", ax=axes[1])
axes[1].set_title("Survival Rate by Passenger Class")
axes[1].set_ylabel("Survival Rate")

sns.histplot(data=df, x="Age", hue="Survived", multiple="stack", bins=30, ax=axes[2])
axes[2].set_title("Age Distribution by Survival")

plt.tight_layout()
plt.savefig("images/eda_overview.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# **Key findings from EDA:**
# - Women survived at a much higher rate than men — consistent with "women and
#   children first" evacuation priority.
# - 1st class passengers had a notably higher survival rate than 3rd class,
#   pointing to a socioeconomic effect on survival chances.
# - Younger children show a visibly higher survival rate within the age distribution.

# %% [markdown]
# ## 3. Data Cleaning

# %%
data = df.copy()

# Age: impute with median age *within each passenger class + sex group*
# (more accurate than a single global median, since age varies by class/sex)
data["Age"] = data.groupby(["Pclass", "Sex"])["Age"].transform(
    lambda x: x.fillna(x.median())
)

# Embarked: only 2 missing values — fill with the most common port
data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])

# Cabin: too sparse to impute meaningfully — convert to a binary
# "had a recorded cabin" flag instead of dropping the signal entirely
data["HasCabin"] = data["Cabin"].notnull().astype(int)
data.drop(columns=["Cabin"], inplace=True)

# Drop columns that don't generalize as predictive features
data.drop(columns=["PassengerId", "Ticket", "Name"], inplace=True, errors="ignore")

print("Missing values remaining:")
print(data.isnull().sum().sum())

# %% [markdown]
# ## 4. Feature Engineering

# %%
# Family size and "traveling alone" flag — often more predictive than
# raw SibSp/Parch counts individually
data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
data["IsAlone"] = (data["FamilySize"] == 1).astype(int)

# Encode categorical variables
le_sex = LabelEncoder()
data["Sex"] = le_sex.fit_transform(data["Sex"])  # male=1, female=0

data = pd.get_dummies(data, columns=["Embarked"], drop_first=True)

data.head()

# %% [markdown]
# ## 5. Train/Test Split

# %%
X = data.drop(columns=["Survived"])
y = data["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}")

# %% [markdown]
# ## 6. Model Training

# %%
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)
log_preds = log_reg.predict(X_test)

rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

log_acc = accuracy_score(y_test, log_preds)
rf_acc = accuracy_score(y_test, rf_preds)

print(f"Logistic Regression accuracy: {log_acc:.3f}")
print(f"Random Forest accuracy:       {rf_acc:.3f}")

# %% [markdown]
# ## 7. Evaluation

# %%
best_model, best_preds, best_name = (
    (rf, rf_preds, "Random Forest") if rf_acc >= log_acc
    else (log_reg, log_preds, "Logistic Regression")
)

print(f"Best model: {best_name}\n")
print(classification_report(y_test, best_preds, target_names=["Did not survive", "Survived"]))

cm = confusion_matrix(y_test, best_preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Did not survive", "Survived"],
            yticklabels=["Did not survive", "Survived"])
plt.title(f"Confusion Matrix — {best_name}")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("images/confusion_matrix.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 8. Feature Importance

# %%
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(7, 5))
sns.barplot(x=importances.values, y=importances.index, hue=importances.index, palette="viridis", legend=False)
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("images/feature_importance.png", bbox_inches="tight")
plt.show()

print(importances)

# %% [markdown]
# ## Conclusion
#
# - Built an end-to-end classification pipeline: cleaning, feature engineering,
#   training, and evaluation.
# - **Sex, passenger class, and fare** were the strongest predictors of survival —
#   consistent with the historical account of the disaster.
# - Random Forest and Logistic Regression were compared directly; the better
#   performer was selected using held-out test accuracy.
#
# **Possible next steps:** hyperparameter tuning (GridSearchCV), cross-validation
# instead of a single train/test split, and trying gradient boosting models
# (XGBoost/LightGBM) for comparison.
