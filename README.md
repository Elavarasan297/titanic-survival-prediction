# Titanic Survival Prediction

An end-to-end data science project predicting Titanic passenger survival using
demographic and travel data. Built as a complete pipeline: data cleaning →
exploratory analysis → feature engineering → model training → evaluation.

## Dataset
891 passenger records with features including class, sex, age, fare, family
relationships, and port of embarkation. Source: [public Titanic dataset](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv).

## Approach
1. **EDA** — examined survival rate by sex, passenger class, and age distribution.
2. **Data cleaning** — imputed missing `Age` using group-wise medians (by class + sex,
   rather than a single global median), filled missing `Embarked` with the mode,
   and converted the sparse `Cabin` field into a binary "has cabin" flag instead of
   dropping it outright.
3. **Feature engineering** — derived `FamilySize` and `IsAlone` from sibling/spouse
   and parent/child counts; encoded categorical variables.
4. **Modeling** — trained and compared Logistic Regression and Random Forest
   classifiers on an 80/20 train/test split.
5. **Evaluation** — accuracy, precision/recall, confusion matrix, and feature
   importance ranking.

## Results
| Model | Test Accuracy |
|---|---|
| Logistic Regression | 80.4% |
| Random Forest | 78.8% |

**Top predictors of survival:** sex, fare, age, and passenger class — consistent
with the historical "women and children first" evacuation priority and the
socioeconomic disparities in access to lifeboats.

## Visuals
- `images/eda_overview.png` — survival rate by sex/class, age distribution
- `images/confusion_matrix.png` — best model's prediction breakdown
- `images/feature_importance.png` — Random Forest feature ranking

## Tools
Python, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn

## Possible next steps
- Hyperparameter tuning with GridSearchCV
- Cross-validation instead of a single train/test split
- Gradient boosting models (XGBoost/LightGBM) for comparison

## Files
- `Titanic_Survival_Prediction.ipynb` — full notebook with code, output, and charts
- `titanic_analysis.py` — plain-script version of the same pipeline
- `data/titanic.csv` — dataset
- `images/` — saved chart outputs
