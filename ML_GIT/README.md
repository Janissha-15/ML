# Machine Learning Exercise 1

This folder (`ML_one`) contains the materials and datasets for the first exercise in the **Getting Started with Machine Learning** series.

## Learning Machine Learning Fundamentals

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

## Supervised Learning: Classification & Model Evaluation

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

## Supervised Learning: Regression & Regularization

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
# Machine Learning Exercise 4  

This folder (`ML_four`) contains the materials and datasets for the fourth exercise in the
**Getting Started with Machine Learning** series.  

## Binary Classification: Logistic Regression & Support Vector Machines  

In this exercise, we focus on binary classification techniques to detect spam emails using
**Logistic Regression** and **Support Vector Machines (SVM)**.  
The experiment emphasizes how feature scaling, kernel selection, and hyperparameter
tuning influence classification performance.

Different SVM kernels are compared to study linear and nonlinear decision boundaries,
and cross-validation is used to evaluate generalization ability.

## Contents  

- **Ex-4.ipynb**: Jupyter Notebook implementing Logistic Regression and SVM models  
- **dataset/**: Directory containing the email dataset used for training and testing  
- **ML-4.pdf**: Experiment report documenting the methodology and results  

## Notebook Summary (Ex-4.ipynb)  

The `Ex-4.ipynb` notebook demonstrates a complete workflow for building and analyzing
binary classifiers:

### 1. Data Preprocessing  
- **Handling Missing Values**: Checking and addressing incomplete records  
- **Feature Scaling**: Standardizing numerical attributes using `StandardScaler`  
- **Train–Validation–Test Split**: Dividing the data to support tuning and unbiased evaluation  

### 2. Models Implemented  
- **Baseline Logistic Regression**: Establishing reference performance  
- **Tuned Logistic Regression**: Improving results through regularization and
  hyperparameter search  
- **Support Vector Machines**: Linear, Polynomial, RBF, and Sigmoid kernels  

### 3. Hyperparameter Tuning  
- **RandomizedSearchCV**: Efficient exploration of parameter combinations  
- **Cross-Validation**: 5-fold cross-validation to assess stability and robustness  

### 4. Evaluation & Performance Analysis  
- **Metrics**: Accuracy, Precision, Recall, and F1-score  
- **Confusion Matrices**: Visualization of classification errors  
- **ROC Curves & AUC**: Comparison of discriminative ability across models  
- **Learning Curves**: Bias–variance analysis with increasing training size  
- **Execution Time**: Computational efficiency comparison  

## Getting Started  

1. Open `Ex-4.ipynb` using JupyterLab, Google Colab, or VS Code  

2. Install the required Python libraries:  
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn


# Machine Learning Exercise 5  

This folder (`ML_five`) contains the implementation and analysis for the fifth exercise in the  
**Getting Started with Machine Learning** series.  

---

## Multi-Class Classification: Perceptron vs Multilayer Perceptron (MLP)  

In this experiment, we compare the performance of a **Single-Layer Perceptron Learning Algorithm (PLA)** and a **Multilayer Perceptron (MLP)** for multi-class handwritten character recognition.  

The objective is to understand the limitations of linear classifiers and demonstrate how deep neural networks with hidden layers and nonlinear activation functions improve classification accuracy.  

Hyperparameter tuning is performed to analyze the impact of learning rate, batch size, activation functions, optimizers, and number of hidden layers on convergence and performance.

---

## Dataset  

**English Handwritten Characters Dataset**  

- **Total Samples:** 3,410  
- **Number of Classes:** 62 (0–9, A–Z, a–z)  
- **Image Type:** Grayscale  

The dataset is preprocessed through resizing, flattening, and normalization before training.

---

## Contents  

- **ML_5_code.ipynb** – Jupyter Notebook implementing PLA and MLP models  
- **ML_5.pdf** – Detailed experiment report with results and comparison  
- *(Dataset sourced from Kaggle – not included in this repository)*  

---

## Notebook Summary (ML_5_code.ipynb)  

The notebook demonstrates a complete workflow for multi-class neural network classification:

### Data Preprocessing  

- Image resizing and flattening  
- Pixel value normalization  
- Train–Validation–Test split  

### Models Implemented  

#### Model A: Perceptron Learning Algorithm (PLA)  

- Step activation function  
- Weight update rule implementation  
- Extended to multi-class using **One-vs-Rest (OvR)** strategy  
- Suitable only for linearly separable data  

#### Model B: Multilayer Perceptron (MLP)  

- One or more hidden layers  
- Nonlinear activation functions (ReLU, Tanh, Sigmoid)  
- Trained using backpropagation  
- Capable of learning nonlinear decision boundaries  

---

### Hyperparameter Tuning  

- Learning Rate  
- Batch Size  
- Optimizer (SGD, Adam)  
- Activation Function  
- Number of Hidden Layers  

Tuning is performed to improve convergence stability and overall accuracy.

---

### Evaluation Metrics  

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  
- ROC Curves (Micro/Macro Average)  
- Training Loss vs Epochs Curve  

---

## Performance Comparison  

The experiment highlights:

- Final tuned hyperparameters for MLP  
- Convergence behavior comparison  
- Strengths and weaknesses of PLA vs MLP  
- Effect of hyperparameter tuning on model performance  

---

## Key Observations  

- PLA struggles with nonlinear decision boundaries.  
- MLP significantly improves accuracy through hidden layers and nonlinear activations.  
- Optimizer and learning rate strongly affect convergence speed.  
- Increasing hidden layers improves representation power but may cause overfitting.  

---

## Conclusion  

This experiment demonstrates the limitations of linear classifiers and shows how deep neural networks provide superior performance for complex multi-class classification tasks such as handwritten character recognition.


---

# Machine Learning Exercise 6

This folder (`ML_six`) contains the materials and implementation for the sixth exercise in the
**Getting Started with Machine Learning** series.

## Decision Tree vs Random Forest Classification

In this experiment, we implement and compare **Decision Tree** and **Random Forest** models for binary classification using the breast cancer dataset.

The focus is on understanding how ensemble learning improves performance over a single model and how hyperparameter tuning affects accuracy and generalization. 

## Contents

* **Ex-6.ipynb** – Notebook implementing Decision Tree and Random Forest
* **ML-6.pdf** – Experiment report
* **dataset/** – Breast cancer dataset

## Dataset

* Wisconsin Diagnostic Breast Cancer Dataset
* 569 samples
* 30 features
* Classes: Benign (B), Malignant (M) 

## Notebook Summary (Ex-6.ipynb)

### Data Preprocessing

* Handling missing values
* Feature scaling
* Train-test split

### Models Implemented

* Decision Tree classifier
* Random Forest classifier

### Hyperparameter Tuning

* Grid search with 5-fold cross-validation
* Parameters: max depth, criterion, number of estimators

### Evaluation

* Accuracy, Precision, Recall, F1-score
* Confusion matrix
* ROC curve

## Key Observations

* Random Forest achieved higher accuracy than Decision Tree
* Ensemble learning reduces overfitting
* Cross-validation improves model reliability

## Conclusion

Random Forest outperformed Decision Tree due to better generalization and reduced variance, making it more suitable for classification tasks.

---

# Machine Learning Exercise 7

This folder (`ML_seven`) contains the materials for the seventh exercise in the
**Getting Started with Machine Learning** series.

## Ensemble Learning: Bagging, Boosting & Stacking

In this experiment, we implement advanced ensemble techniques including **Bagging**, **Boosting**, and **Stacking** to improve classification performance.

The goal is to analyze how combining models reduces bias and variance and enhances prediction accuracy. 

## Contents

* **Ex-7.ipynb** – Notebook implementing ensemble models
* **ML-7.pdf** – Experiment report
* **dataset/** – Breast cancer dataset

## Dataset

* Wisconsin Breast Cancer Dataset
* 569 samples
* 30 features
* Classes: Benign, Malignant 

## Notebook Summary (Ex-7.ipynb)

### Data Preprocessing

* Feature scaling
* Train-test split

### Models Implemented

* Bagging classifier
* Boosting (AdaBoost)
* Stacked ensemble (multiple base models + meta learner)

### Hyperparameter Tuning

* Number of estimators
* Learning rate
* Sampling strategies

### Evaluation

* Accuracy, Precision, Recall, F1-score
* Confusion matrix
* ROC curve

## Key Observations

* Bagging reduces variance and improves stability
* Boosting reduces bias and increases accuracy
* Stacking combines strengths of multiple models

## Conclusion

Ensemble methods significantly improved performance and generalization compared to individual models.

---

# Machine Learning Exercise 8

This folder (`ML_eight`) contains the materials for the eighth exercise in the
**Getting Started with Machine Learning** series.

## Dimensionality Reduction using PCA

In this experiment, we study the effect of **Principal Component Analysis (PCA)** on both regression and classification models.

The focus is on reducing dimensionality while retaining maximum variance and analyzing its impact on model performance. 

## Contents

* **Ex-8.ipynb** – Notebook implementing PCA and models
* **ML-8.pdf** – Experiment report
* **dataset/** – Academic dataset

## Dataset

* 1000 samples
* Academic features
* Targets:

  * Regression → Final Score
  * Classification → Performance Level 

## Notebook Summary (Ex-8.ipynb)

### Data Preprocessing

* Feature scaling
* Data splitting

### PCA Analysis

* Scree plot
* Explained variance
* Component selection

### Models Implemented

#### Regression

* Linear Regression
* Random Forest

#### Classification

* Logistic Regression
* Support Vector Machine (SVM)

### Evaluation

* Regression → MSE, R²
* Classification → Accuracy, F1-score
* Comparison: With PCA vs Without PCA

## Key Observations

* PCA reduces dimensionality and multicollinearity
* Linear models show slight improvement
* Tree-based models show moderate improvement
* Classification models show minimal change

## Conclusion

PCA improves performance by reducing noise and redundancy, but excessive dimensionality reduction may lead to information loss.

---

# Machine Learning Exercise 9

This folder (`ML_nine`) contains the materials for the ninth exercise in the
**Getting Started with Machine Learning** series.

## Unsupervised Learning: Clustering Techniques

In this experiment, we implement and compare clustering algorithms including **K-Means**, **DBSCAN**, and **Hierarchical Clustering** on human activity recognition data.

The objective is to analyze clustering performance and visualize high-dimensional data using dimensionality reduction techniques. 

## Contents

* **Ex-9.ipynb** – Notebook implementing clustering algorithms
* **ML-9.pdf** – Experiment report
* **dataset/** – HAR dataset

## Dataset

* Human Activity Recognition dataset
* Activities: Walking, Sitting, Standing, etc.
* Data from accelerometer and gyroscope sensors
* Collected from multiple subjects 

## Notebook Summary (Ex-9.ipynb)

### Data Preprocessing

* Feature scaling
* Data normalization

### Dimensionality Reduction

* PCA
* t-SNE visualization

### Models Implemented

* K-Means clustering
* DBSCAN
* Hierarchical clustering (HAC)

### Evaluation

* Silhouette score
* Davies-Bouldin index
* ARI and NMI metrics
* Cluster visualization

## Key Observations

* K-Means performs well but depends on k
* DBSCAN is sensitive to parameter selection
* Hierarchical clustering gives better clustering quality
* PCA and t-SNE improve visualization

## Conclusion

Hierarchical clustering performed best overall, while K-Means required tuning and DBSCAN struggled with parameter sensitivity.


