# Machine Learning Exercise 1

This folder (`ML_one`) contains the materials and datasets for the first exercise in the **Getting Started with Machine Learning** series.

## 📊 Learning Machine Learning Fundamentals

In this exercise, we explore the basics of Machine Learning by working hands-on with essential Python libraries such as **NumPy**, **Pandas**, and **Matplotlib**.  
The focus is on understanding data through cleaning, manipulation, and visualization techniques, which are critical steps before building any model.

This folder serves as a learning playground for experimenting with multiple real-world datasets and developing intuition about how Machine Learning algorithms interpret data.

## Contents

- **Ex-1.ipynb**: Main Jupyter Notebook for Exercise 1  
- **ML-1.pdf**: Problem statement and theoretical background  
- **datasets/**: Folder containing CSV datasets used in the exercise  

## Datasets

The `datasets/` directory includes:

- `Iris.csv` – Iris flower classification dataset  
- `loan_train.csv` – Loan amount prediction dataset  
- `english.csv` – Metadata for handwritten character images  
- `diabetes.csv` – Diabetes prediction dataset  
- `email.csv` – Email spam detection dataset  

**Handwritten Characters Dataset**  
The complete dataset can be downloaded from:  
https://www.kaggle.com/datasets/dhruvildave/english-handwritten-characters-dataset

## Notebook Summary (Ex-1.ipynb)

Exploratory Data Analysis (EDA) is performed on multiple datasets:

1. **Loan Amount Prediction**
   - Data inspection, missing value checks
   - Statistical summaries
   - Histograms, boxplots, scatter plots, correlation heatmaps

2. **Iris Dataset Analysis**
   - Species-wise distribution
   - Pairplots and correlation analysis

3. **Handwritten Character Recognition**
   - Label distribution
   - Sample image visualization

4. **Diabetes Prediction**
   - Feature exploration related to health metrics

5. **Email Spam Classification**
   - Initial analysis for spam vs non-spam detection

## Getting Started

1. Open `Ex-1.ipynb` using Jupyter Notebook / Google Colab / VS Code  
2. Install required libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`  
3. Run the notebook cells sequentially




# Machine Learning Exercise 2

This folder (`ML_two`) contains the materials for the second exercise in the **Getting Started with Machine Learning** series.

## 📊 Supervised Learning: Classification & Model Evaluation

This exercise focuses on building and evaluating supervised learning models.  
Using the **Spambase dataset**, a **Bernoulli Naive Bayes** classifier is implemented to detect spam emails.

The complete ML pipeline is covered, including preprocessing, model training, and evaluation using multiple performance metrics.

## Contents

- **Ex-2.ipynb**: Jupyter Notebook for model implementation  
- **ML-2.pdf**: Experiment report  
- **datasets/**: Folder containing the dataset  

## Dataset

- `spambase_csv_Kaggle.csv`  
  - 4601 instances  
  - 57 numerical features  
  - Binary labels: Spam (1), Non-Spam (0)

## Notebook Summary (Ex-2.ipynb)

1. **Data Inspection & Cleaning**
   - Shape, data quality checks
   - Statistical summaries

2. **Preprocessing**
   - Feature scaling using `StandardScaler`
   - Train-test split

3. **Model Implementation**
   - Bernoulli Naive Bayes classifier

4. **Evaluation**
   - Accuracy, Precision, Recall, F1-score
   - Confusion matrix
   - Learning curves

## Getting Started

1. Open `Ex-2.ipynb`
2. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn



---

## 📁 `ML_GIT / ML_Three / README.md`

```md
# Machine Learning Exercise 3

This folder (`ML_Three`) contains the third exercise in the **Getting Started with Machine Learning** series.

## 📈 Supervised Learning: Regression & Regularization

This exercise focuses on predicting continuous values using regression models.  
Different regularization techniques are explored to prevent overfitting and improve generalization.

## Contents

- **Ex-3.ipynb**: Regression pipeline implementation  
- **ML-3.pdf**: Experiment report  
- **dataset/**: Housing / loan-related datasets  

## Notebook Summary (Ex-3.ipynb)

1. **Data Preprocessing**
   - Missing value handling using `SimpleImputer`
   - Feature scaling and encoding using `ColumnTransformer`
   - Pipeline-based workflow

2. **Regression Models**
   - Linear Regression
   - Ridge Regression (L2)
   - Lasso Regression (L1)
   - Elastic Net

3. **Hyperparameter Tuning**
   - GridSearchCV
   - Cross-validation

4. **Evaluation**
   - MAE, MSE, RMSE, R²
   - Residual analysis
   - Coefficient comparison

## Getting Started

1. Open `Ex-3.ipynb`
2. Install required libraries
3. Run preprocessing, training, and evaluation cells


# Machine Learning Exercise 4

This folder (`ML_four`) contains the fourth exercise in the **Getting Started with Machine Learning** series.

## 📧 Binary Classification: Logistic Regression & SVM

This exercise compares **Logistic Regression** and **Support Vector Machines (SVM)** for spam email classification.  
Different kernels and regularization strategies are analyzed to study model performance and generalization.

## Contents

- **Ex-4.ipynb**: Notebook implementing Logistic Regression and SVM  
- **ML-4.pdf**: Experiment report  
- **dataset/**: Email spam dataset  

## Notebook Summary (Ex-4.ipynb)

1. **Data Preprocessing**
   - Missing value handling
   - Feature scaling
   - Train-validation-test split

2. **Models Implemented**
   - Logistic Regression (baseline & tuned)
   - SVM with Linear, Polynomial, RBF, and Sigmoid kernels

3. **Hyperparameter Tuning**
   - RandomizedSearchCV
   - Cross-validation

4. **Evaluation**
   - Accuracy, Precision, Recall, F1-score
   - Confusion matrices
   - ROC curves & AUC
   - Learning curves
   - Execution time comparison

## Getting Started

1. Open `Ex-4.ipynb`
2. Install required libraries
3. Execute all cells sequentially

