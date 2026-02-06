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



# Machine Learning Exercise 3

This folder (`ML_Three`) contains the materials and datasets for the third exercise in the
**Getting Started with Machine Learning** series.

## 📈 Supervised Learning: Regression & Regularization

In this exercise, we explore advanced regression techniques to predict continuous values.
Using a housing / loan-related dataset, multiple regression models are implemented and
compared. The focus is on understanding how regularization techniques such as
**Ridge**, **Lasso**, and **Elastic Net** help prevent overfitting and improve feature
selection.

This exercise highlights the importance of structured data pipelines, including automated
handling of missing values and encoding of categorical variables.

## Contents

- **Ex-3.ipynb**: Jupyter Notebook implementing the regression pipeline and model comparison  
- **dataset/**: Directory containing datasets used for training and testing  
- **ML-3.pdf**: Experiment report documenting methodology and results  

## Notebook Summary (Ex-3.ipynb)

The `Ex-3.ipynb` notebook demonstrates a complete workflow for regression modeling:

### 1. Data Preprocessing Pipeline
- **Handling Missing Values**: Using `SimpleImputer` for numerical and categorical data  
- **Feature Transformation**: Applying `StandardScaler` and `OneHotEncoder` through
  `ColumnTransformer`  
- **Pipelines**: Building Scikit-learn `Pipeline` objects for clean and reproducible workflows  

### 2. Regression Models Implemented
- **Linear Regression**: Baseline model  
- **Ridge Regression (L2)**: Controls multicollinearity using L2 regularization  
- **Lasso Regression (L1)**: Performs automatic feature selection  
- **Elastic Net**: Combines L1 and L2 penalties for flexible regularization  

### 3. Hyperparameter Tuning
- **GridSearchCV**: Searching optimal `alpha` and `l1_ratio` values  
- **Cross-Validation**: Ensuring model generalization using k-fold validation  

### 4. Evaluation & Performance Analysis
- **Metrics**: MAE, MSE, RMSE, and R²  
- **Coefficient Analysis**: Comparing feature importance across models  
- **Execution Time**: Evaluating computational efficiency  

## Getting Started

1. Open `Ex-3.ipynb` using JupyterLab, Google Colab, or VS Code  
2. Install required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn



---

## 📁 `ML_GIT / ML_four / README.md`

```md
# Machine Learning Exercise 4

This folder (`ML_four`) contains the materials and datasets for the fourth exercise in the
**Getting Started with Machine Learning** series.

## 📧 Binary Classification: Logistic Regression & Support Vector Machines

This exercise focuses on binary classification for spam email detection using
**Logistic Regression** and **Support Vector Machines (SVM)**.
The experiment analyzes how feature scaling, kernel selection, and hyperparameter tuning
impact classification performance.

Different SVM kernels are compared to study linear and non-linear decision boundaries,
while cross-validation is used to assess generalization ability.

## Contents

- **Ex-4.ipynb**: Jupyter Notebook implementing Logistic Regression and SVM models  
- **dataset/**: Directory containing the email dataset  
- **ML-4.pdf**: Experiment report with analysis and results  

## Notebook Summary (Ex-4.ipynb)

The `Ex-4.ipynb` notebook presents a full binary classification workflow:

### 1. Data Preprocessing
- **Handling Missing Values**: Identifying and addressing incomplete records  
- **Feature Scaling**: Standardizing numerical features using `StandardScaler`  
- **Train–Validation–Test Split**: Ensuring unbiased model evaluation  

### 2. Models Implemented
- **Baseline Logistic Regression**: Reference model  
- **Tuned Logistic Regression**: Performance improvement using regularization  
- **Support Vector Machines**: Linear, Polynomial, RBF, and Sigmoid kernels  

### 3. Hyperparameter Tuning
- **RandomizedSearchCV**: Efficient parameter exploration  
- **Cross-Validation**: 5-fold validation for stability and robustness  

### 4. Evaluation & Performance Analysis
- **Metrics**: Accuracy, Precision, Recall, and F1-score  
- **Confusion Matrices**: Visualization of classification errors  
- **ROC Curves & AUC**: Model comparison across kernels  
- **Learning Curves**: Bias–variance analysis  
- **Execution Time**: Computational cost comparison  

## Getting Started

1. Open `Ex-4.ipynb` using JupyterLab, Google Colab, or VS Code  
2. Install required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn

---

## 📁 `ML_GIT / ML_four / README.md`

```md
# Machine Learning Exercise 4

This folder (`ML_four`) contains the materials and datasets for the fourth exercise in the
**Getting Started with Machine Learning** series.

## 📧 Binary Classification: Logistic Regression & Support Vector Machines

This exercise focuses on binary classification for spam email detection using
**Logistic Regression** and **Support Vector Machines (SVM)**.
The experiment analyzes how feature scaling, kernel selection, and hyperparameter tuning
impact classification performance.

Different SVM kernels are compared to study linear and non-linear decision boundaries,
while cross-validation is used to assess generalization ability.

## Contents

- **Ex-4.ipynb**: Jupyter Notebook implementing Logistic Regression and SVM models  
- **dataset/**: Directory containing the email dataset  
- **ML-4.pdf**: Experiment report with analysis and results  

## Notebook Summary (Ex-4.ipynb)

The `Ex-4.ipynb` notebook presents a full binary classification workflow:

### 1. Data Preprocessing
- **Handling Missing Values**: Identifying and addressing incomplete records  
- **Feature Scaling**: Standardizing numerical features using `StandardScaler`  
- **Train–Validation–Test Split**: Ensuring unbiased model evaluation  

### 2. Models Implemented
- **Baseline Logistic Regression**: Reference model  
- **Tuned Logistic Regression**: Performance improvement using regularization  
- **Support Vector Machines**: Linear, Polynomial, RBF, and Sigmoid kernels  

### 3. Hyperparameter Tuning
- **RandomizedSearchCV**: Efficient parameter exploration  
- **Cross-Validation**: 5-fold validation for stability and robustness  

### 4. Evaluation & Performance Analysis
- **Metrics**: Accuracy, Precision, Recall, and F1-score  
- **Confusion Matrices**: Visualization of classification errors  
- **ROC Curves & AUC**: Model comparison across kernels  
- **Learning Curves**: Bias–variance analysis  
- **Execution Time**: Computational cost comparison  

## Getting Started

1. Open `Ex-4.ipynb` using JupyterLab, Google Colab, or VS Code  
2. Install required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
