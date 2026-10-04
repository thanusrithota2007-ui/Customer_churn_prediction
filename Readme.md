# Customer Churn Prediction

A machine learning project to predict whether a telecom customer is likely to churn based on customer demographics, services, contract details, and billing information.

The project focuses not only on achieving good predictive performance, but also on **identifying as many potential churners as possible** and making the final model interpretable.

---

## 📌 Project Overview

Customer churn is a major challenge for subscription-based businesses. Identifying customers who are likely to leave can help businesses take preventive actions such as targeted offers, improved support, or retention campaigns.

In this project, multiple classification models were evaluated and compared. Logistic Regression was selected as the final model because of its strong churn recall, ROC-AUC performance, and interpretability.

The project also includes:

* Exploratory Data Analysis
* Data preprocessing
* Multiple classification models
* Hyperparameter tuning
* Classification threshold optimization
* Model evaluation
* Logistic Regression coefficient and odds-ratio interpretation

---

## 🎯 Objectives

* Understand the factors associated with customer churn.
* Explore patterns in customer demographics, services, contracts, and billing.
* Build machine learning models to predict churn.
* Compare different classification algorithms.
* Tune the selected model's hyperparameters.
* Optimize the classification threshold to improve churn detection.
* Interpret the final Logistic Regression model.

---

## 📊 Dataset

The dataset contains **7,043 customers and 21 features**.

### Target Variable

`Churn`

* `Yes` → Customer churned
* `No` → Customer did not churn

### Important Features

The dataset includes information related to:

* Customer demographics
* Tenure
* Contract type
* Internet service
* Payment method
* Monthly charges
* Online security
* Technical support
* Streaming services

`customerID` was removed because it is an identifier and does not provide predictive information.

---

## 🔍 Exploratory Data Analysis

Several important patterns were identified during EDA.

### Tenure and Churn

Customers with shorter tenure showed substantially higher churn rates.

* Churned customers had a mean tenure of approximately **18 months**.
* Non-churned customers had a mean tenure of approximately **38 months**.
* Customers with **0–12 months** of tenure had a churn rate of approximately **47.4%**.
* Customers with **61–72 months** of tenure had a churn rate of approximately **6.6%**.

### Contract Type

Month-to-month customers had considerably higher churn than customers on longer contracts.

| Contract       | Churn Rate |
| -------------- | ---------: |
| Month-to-month |     ~42.7% |
| One year       |     ~11.3% |
| Two year       |      ~2.8% |

### Monthly Charges

Churned customers had higher average monthly charges:

* Churned: **~74.44**
* Non-churned: **~61.27**

### Payment Method

Customers using electronic checks showed a relatively high churn rate of approximately **45.3%**.

### Internet Service

Fiber-optic customers showed higher churn than DSL customers.

| Internet Service    | Churn Rate |
| ------------------- | ---------: |
| Fiber optic         |     ~41.9% |
| DSL                 |     ~19.0% |
| No internet service |      ~7.4% |

Among fiber-optic customers, those without technical support showed a higher observed churn rate than those with technical support.

> These findings represent observed associations in the dataset and should not be interpreted as causal relationships.

---

## ⚙️ Machine Learning Workflow

```text
Data Loading
     ↓
Data Cleaning
     ↓
Exploratory Data Analysis
     ↓
Feature Preprocessing
     ↓
Train/Test Split
     ↓
Baseline Model Training
     ↓
Model Comparison
     ↓
Logistic Regression Selection
     ↓
Hyperparameter Tuning
     ↓
Threshold Optimization
     ↓
Final Evaluation
     ↓
Model Interpretation
```

---

## 🧹 Data Preprocessing

The preprocessing pipeline includes:

* Removing the `customerID` identifier.
* Handling the `TotalCharges` data issue.
* Separating numerical and categorical features.
* Encoding categorical variables.
* Scaling numerical features where required.
* Using a train/test split.
* Applying preprocessing through a Scikit-learn pipeline.

---

## 🤖 Models Evaluated

Three classification models were initially compared:

1. Logistic Regression
2. Random Forest
3. XGBoost

### Baseline Results

| Model               |   Accuracy |  Precision |     Recall |     F1 |    ROC-AUC |
| ------------------- | ---------: | ---------: | ---------: | -----: | ---------: |
| Logistic Regression |     75.59% |     52.49% | **82.04%** | 64.02% | **0.8605** |
| Random Forest       |     79.35% |     65.07% |     47.45% | 54.88% |     0.8396 |
| XGBoost             | **80.98%** | **67.56%** |     54.16% | 60.12% |     0.8600 |

Although XGBoost achieved higher accuracy, Logistic Regression was selected because it provided substantially better **churn recall** while maintaining a strong ROC-AUC.

For a churn-retention use case, missing an actual churner can be more costly than incorrectly flagging some customers as potential churners.

---

## 🔧 Hyperparameter Tuning

GridSearchCV with 5-fold cross-validation was used to tune the Logistic Regression model.

### Best Parameters

```text
C = 100
solver = liblinear
class_weight = None
```

### Best Cross-Validation Score

```text
ROC-AUC = 0.8393
```

The tuned model was then evaluated on the test set.

---

## 🎚️ Classification Threshold Optimization

By default, binary Logistic Regression classifies a customer as a churner when:

```text
Predicted Probability ≥ 0.50
```

However, for this project, identifying more potential churners is important.

Therefore, different probability thresholds were evaluated.

A threshold of **0.30** was selected based on the highest F1-score obtained on the validation data.

### Default vs Tuned Threshold

| Metric     | Threshold 0.50 | Threshold 0.30 |
| ---------- | -------------: | -------------: |
| Accuracy   |         81.90% |         77.57% |
| Precision  |         68.67% |         55.35% |
| **Recall** |         58.18% |     **79.09%** |
| **F1**     |         62.99% |     **65.12%** |
| ROC-AUC    |         86.01% |         86.01% |

### Interpretation

Lowering the threshold from **0.50 to 0.30** increased churn recall from **58.18% to 79.09%**.

This means the model identifies substantially more of the customers who actually churned.

The trade-off is lower precision and accuracy because more customers are classified as potential churners.

The F1-score also improved from **62.99% to 65.12%**.

ROC-AUC remains unchanged because threshold tuning changes the final classification decision, not the underlying probability ranking of the model.

---

## 🏆 Final Model

The final model is:

```text
Logistic Regression

C              = 100
Solver         = liblinear
Class Weight   = None
Threshold      = 0.30
```

### Final Test Performance

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **77.57%** |
| Precision | **55.35%** |
| Recall    | **79.09%** |
| F1-Score  | **65.12%** |
| ROC-AUC   | **86.01%** |

The final model prioritizes **recall**, making it suitable for a scenario where identifying customers at risk of churn is more important than maximizing overall accuracy.

---

## 🔎 Model Interpretability

One advantage of Logistic Regression is that its coefficients can be interpreted.

The project extracts:

* Feature coefficients
* Odds ratios

The odds ratio is calculated as:

```text
Odds Ratio = exp(Coefficient)
```

This allows the model's predictions to be connected to individual features and helps explain which factors are associated with increased or decreased churn likelihood.

---

## 💼 Business Value

A churn prediction system can help a telecom company:

* Identify customers at higher risk of leaving.
* Prioritize retention campaigns.
* Offer targeted incentives.
* Improve customer support for high-risk segments.
* Allocate retention resources more efficiently.

The threshold can also be adjusted depending on the business objective.

For example:

```text
Higher threshold
→ fewer customers flagged
→ higher precision
→ lower recall

Lower threshold
→ more customers flagged
→ lower precision
→ higher recall
```

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Jupyter Notebook / Google Colab

---

## 📁 Project Structure

```text
customer-churn-prediction/
│
├── Customer_Churn_Prediction.ipynb
├── README.md
├── data.csv
```



---

## 🚀 How to Run

### 1. Clone the repository

```bash

git clone https://github.com/thanusrithota2007-ui/Customer_churn_prediction.git
cd Customer_churn_prediction

```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

### 3. Open the notebook

```bash
jupyter notebook Customer_Churn_Prediction.ipynb
```



---

## 📌 Key Takeaways

1. **Tenure and contract type showed strong associations with churn.**
2. Month-to-month customers had substantially higher observed churn.
3. Logistic Regression provided strong churn recall and ROC-AUC.
4. Hyperparameter tuning identified `C=100` and `liblinear` as the best configuration under the selected CV objective.
5. Lowering the classification threshold from **0.50 to 0.30** increased churn recall from **58.18% to 79.09%**.
6. The final model achieved an **ROC-AUC of 0.8601**.
7. Logistic Regression provides interpretable coefficients and odds ratios.

---

## 👩‍💻 Author

**Thota Thanusri**

B.Tech — Computer Science & Engineering (Data Science)