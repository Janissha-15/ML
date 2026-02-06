machine_learning_repository:
  name: ML_GIT
  series: Getting Started with Machine Learning

  exercises:

    - exercise: Machine Learning Exercise 1
      folder: ML_one
      description: >
        This exercise introduces the fundamentals of Machine Learning
        through hands-on Exploratory Data Analysis (EDA). The focus is on
        understanding datasets using Python libraries before building
        any models.

      learning_focus:
        - Data cleaning and preprocessing
        - Data visualization and pattern discovery
        - Building intuition about datasets

      tools_used:
        - NumPy
        - Pandas
        - Matplotlib
        - Seaborn

      contents:
        - Ex-1.ipynb
        - ML-1.pdf
        - datasets/

      datasets:
        - Iris.csv
        - loan_train.csv
        - english.csv
        - diabetes.csv
        - email.csv

      notebook_summary:
        loan_amount_prediction:
          - Data inspection and missing value analysis
          - Statistical summaries
          - Visualizations including histograms, boxplots, and heatmaps

        iris_dataset_analysis:
          - Species-wise analysis
          - Pairplots and correlation analysis

        handwritten_character_recognition:
          - Label distribution analysis
          - Visualization of sample handwritten characters

        diabetes_prediction:
          - Health metric analysis
          - Feature exploration

        email_spam_classification:
          - Text dataset inspection
          - Initial spam vs non-spam exploration

    - exercise: Machine Learning Exercise 2
      folder: ML_two
      description: >
        This exercise focuses on supervised learning techniques and
        introduces basic Machine Learning algorithms. Emphasis is placed
        on understanding model behavior and evaluation metrics.

      learning_focus:
        - Supervised learning concepts
        - Model training and testing
        - Performance evaluation

      algorithms_covered:
        - Linear Regression
        - Logistic Regression
        - K-Nearest Neighbors (KNN)

      contents:
        - Ex-2.ipynb
        - ML-2.pdf
        - datasets/

      notebook_summary:
        - Dataset loading and preprocessing
        - Train-test split
        - Model training
        - Accuracy, precision, recall, and F1-score analysis
        - Visualization of results and predictions

    - exercise: Machine Learning Exercise 3
      folder: ML_Three
      description: >
        This exercise explores classification techniques and kernel-based
        learning methods. The goal is to understand complex decision
        boundaries and the role of feature transformation.

      learning_focus:
        - Classification problems
        - Kernel methods
        - Bias–variance tradeoff

      algorithms_covered:
        - Support Vector Machines (SVM)
        - Kernel functions (Linear, Polynomial, RBF)

      contents:
        - Ex-3.ipynb
        - ML-3.pdf
        - datasets/

      notebook_summary:
        - Understanding linear vs non-linear separability
        - Applying different kernel functions
        - Hyperparameter tuning
        - Visualization of decision boundaries

    - exercise: Machine Learning Exercise 4
      folder: ML_four
      description: >
        This exercise covers advanced Machine Learning concepts including
        model optimization and performance improvement strategies.

      learning_focus:
        - Advanced evaluation techniques
        - Model optimization
        - Case-study based analysis

      topics_covered:
        - Cross-validation
        - Overfitting and underfitting
        - Feature selection
        - Model comparison

      contents:
        - Ex-4.ipynb
        - ML-4.pdf
        - datasets/

      notebook_summary:
        - Comparative study of multiple models
        - Optimization using validation techniques
        - Final performance analysis and conclusions

  getting_started:
    steps:
      - Open the notebook files using Jupyter Notebook, JupyterLab, VS Code, or Google Colab
      - Install required libraries (numpy, pandas, matplotlib, seaborn, scikit-learn)
      - Follow instructions inside each notebook for execution

  author:
    name: Jani
    role: Engineering Student
    institution: SSN College of Engineering

  note: >
    This repository is maintained for academic learning and coursework
    submission purposes only.
