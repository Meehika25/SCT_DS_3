## 📌 Objective  
The goal of this task is to build a **Decision Tree Classifier** that predicts whether a customer will purchase a product or service based on their **demographic and behavioral data**.  

We use the **Bank Marketing dataset** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing).  

---

## 📂 Dataset  
The dataset consists of customer information collected from a Portuguese banking institution.  
- Features include demographic and marketing data such as `age`, `job`, `marital status`, `education`, `balance`, etc.  
- The target variable is:  
  - **yes** → customer subscribes to the product  
  - **no** → customer does not subscribe  

---

## ⚙️ Steps Performed  

1. **Import Libraries**  
   - `pandas`, `numpy` for data handling  
   - `matplotlib`, `seaborn` for visualization  
   - `sklearn` for building the decision tree model  

2. **Load Dataset**  
   ```python
   import pandas as pd
   df = pd.read_csv("bank.csv")
Data Cleaning

Handled missing values (if any)

Encoded categorical features using LabelEncoder

Checked data types and distributions

Exploratory Data Analysis (EDA)

Visualized customer age distribution

Checked relationship between job type and subscription

Compared balance and previous campaign outcomes with subscription rates

Model Building

Split data into training and testing sets using train_test_split

Built a Decision Tree Classifier using sklearn.tree.DecisionTreeClassifier

Trained the model on training data

Model Evaluation

Calculated Accuracy Score

Generated Classification Report (Precision, Recall, F1-score)

Visualized the Decision Tree structure

📊 Results
The model was successfully trained on the Bank Marketing dataset.

Evaluation metrics such as accuracy, precision, recall, and F1-score were used to measure performance.

Decision tree visualization provided interpretability of the model.
