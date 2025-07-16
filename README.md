# Heart Attack Risk Prediction using Machine Learning 🫀

This project uses machine learning models to analyze and predict the risk of heart attacks based on patient demographic and clinical data. The process includes data preprocessing, visualization, model training, evaluation, and hyperparameter tuning.

---

## 📁 Dataset

The dataset is called: `heart_attack_prediction_dataset.csv`  
It contains information such as:
- Age, Sex, Country, Continent
- Cholesterol, Heart Rate, BMI
- Lifestyle factors (Diet, Smoking, Exercise, Alcohol consumption)
- Medical history (Diabetes, Family History, Previous Heart Problems)

---

## 📊 Exploratory Data Analysis (EDA)

Several visualizations are created using:
- **Seaborn** and **Matplotlib**
- Distribution of age, country, and sex
- Heart attack risk vs. various features (like continent, country, and gender)

Example plots:
- Heart Attack Risk by Sex
- Risk by Continent
- Grouped bar charts by Country and Gender

---

## 🧹 Data Preprocessing

Features were engineered and grouped using domain knowledge:
- Age → Age Group (`Baby`, `Young Adult`, `Middle-aged`, `Senior`)
- BMI → BMI Group (`Underweight`, `Normal`, `Overweight`, `Obese`)
- Income → Income Group (`Low`, `Mid`, `High`)
- Exercise Hours → Physical Activity Group (`Low`, `Normal`, `High`)

Other preprocessing steps:
- Encoding categorical variables using `LabelEncoder`, `OrdinalEncoder`, `OneHotEncoder`
- Normalization using `MinMaxScaler`
- Handling missing values using `SimpleImputer`

---

## 🤖 Machine Learning Models

We trained and evaluated the following models using **scikit-learn**:

| Model                  | Description                                |
|-----------------------|--------------------------------------------|
| Logistic Regression    | Baseline linear model                      |
| K-Nearest Neighbors    | Distance-based classifier                  |
| Random Forest          | Tree-based ensemble model (best performer) |

All models are trained using pipelines and evaluated on accuracy, precision, recall, F1-score, and ROC AUC.

---

## 🧪 Model Evaluation Metrics

Function `get_metrics()` is used to compute:

- `Accuracy`
- `Precision`
- `Recall`
- `F1 Score`
- `ROC AUC`

Models are trained using an 80/20 train-test split with reproducible seeds (`random_state=42`).

---

## 🔍 Hyperparameter Tuning

Used `RandomizedSearchCV` for optimizing the Random Forest:

```python
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2]
}
