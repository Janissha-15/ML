#!/usr/bin/env python
# coding: utf-8

# # Experiment 8: Effect of PCA on Regression and Classification Models
# 

# ## 1. Import Libraries

# In[16]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, make_scorer

print('All libraries imported successfully!')

# ## 2. Load Dataset

# In[17]:


# Row 0 of the CSV contains column names, so read with header=None first
df_raw = pd.read_csv(r"C:\Users\janis\ML\Dataset.csv", header=None)

# Set row 0 as the header and drop it from data
df_raw.columns = df_raw.iloc[0].str.strip()
df = df_raw.drop(index=0).reset_index(drop=True)

# Correct target column names
regression_target     = 'Final_Score_Regression'
classification_target = 'Performance_Level_Classification'

# Feature columns = all except the two targets
feature_cols = [col for col in df.columns if col not in [regression_target, classification_target]]

# Convert features and regression target to numeric
df[feature_cols]      = df[feature_cols].apply(pd.to_numeric, errors='coerce')
df[regression_target] = pd.to_numeric(df[regression_target], errors='coerce')

print('Dataset loaded successfully!')
print(f'Shape         : {df.shape}')
print(f'Samples       : {df.shape[0]}')
print(f'Total Columns : {df.shape[1]}')
print(f'Feature Cols  : {len(feature_cols)}')
print(f'\nFeature columns: {feature_cols}')
print(f'\nRegression Target    : {regression_target}')
print(f'Classification Target: {classification_target}')
df.head()

# ## 3. Dataset Overview

# In[18]:


print('=== Dataset Info ===')
df.info()
print('\n=== Statistical Summary ===')
df[feature_cols + [regression_target]].describe()

# ## 4. Target Distribution

# In[19]:


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Regression target
axes[0].hist(df[regression_target].astype(float), bins=30, color='steelblue', edgecolor='black')
axes[0].set_title(f'Distribution of Regression Target:\n{regression_target}')
axes[0].set_xlabel(regression_target)
axes[0].set_ylabel('Frequency')

# Classification target
class_counts = df[classification_target].value_counts()
axes[1].bar(class_counts.index, class_counts.values, color='coral', edgecolor='black')
axes[1].set_title(f'Distribution of Classification Target:\n{classification_target}')
axes[1].set_xlabel(classification_target)
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('target_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

print('\nRegression Target - Final_Score_Regression:')
print(df[regression_target].astype(float).describe())
print('\nClassification Target - Performance_Level_Classification:')
print(df[classification_target].value_counts())

# ## 5. Preprocessing Steps

# In[20]:


# Step 1: Handle missing values
print('=== Step 1: Missing Value Check ===')
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else 'No missing values found!')
df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
print(f'Missing values after handling: {df.isnull().sum().sum()}')

# In[21]:


# Step 2: Encode categorical target for classification
print('=== Step 2: Encode Categorical Target ===')
le    = LabelEncoder()
y_cls = le.fit_transform(df[classification_target])
print(f'Original classes : {le.classes_}')
print(f'Encoded values   : {np.unique(y_cls)}')
print(f'Label mapping    : {dict(zip(le.classes_, le.transform(le.classes_)))}')

# In[22]:


# Step 3: Extract X and y
X     = df[feature_cols].values.astype(float)
y_reg = df[regression_target].values.astype(float)

print(f'Features matrix shape       : {X.shape}')
print(f'Regression target shape     : {y_reg.shape}')
print(f'Classification target shape : {y_cls.shape}')

# In[23]:


# Step 4: Standardize features — VERY IMPORTANT before PCA
print('=== Step 4: Standardize Features ===')
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f'Mean of scaled features (should be ~0): {X_scaled.mean(axis=0).round(3)}')
print(f'Std  of scaled features (should be ~1): {X_scaled.std(axis=0).round(3)}')
print('Standardization complete!')

# ## 6. PCA Implementation

# In[24]:


# Full PCA to study explained variance
pca_full            = PCA()
pca_full.fit(X_scaled)
explained_var_ratio = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_var_ratio)

print('Explained variance per component:')
for i, (v, c) in enumerate(zip(explained_var_ratio, cumulative_variance)):
    print(f'  PC{i+1:02d}: {v*100:.2f}%  |  Cumulative: {c*100:.2f}%')

# In[25]:


# Scree Plot
n_total = len(explained_var_ratio)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(range(1, n_total + 1), explained_var_ratio * 100, color='steelblue', edgecolor='black')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Explained Variance (%)')
axes[0].set_title('Scree Plot - Individual Explained Variance')
axes[0].set_xticks(range(1, n_total + 1))

axes[1].plot(range(1, n_total + 1), cumulative_variance * 100,
             marker='o', markersize=6, color='coral', linewidth=2)
axes[1].axhline(y=95, color='green', linestyle='--', linewidth=1.5, label='95% Threshold')
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Explained Variance (%)')
axes[1].set_title('Cumulative Explained Variance')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('scree_plot.png', dpi=150, bbox_inches='tight')
plt.show()

n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f'Components needed for 95% variance : {n_components_95}')
print(f'Actual variance retained           : {cumulative_variance[n_components_95-1]*100:.2f}%')

# In[26]:


# Apply PCA retaining 95% variance
pca   = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

n_components_chosen = pca.n_components_
explained_var_total = pca.explained_variance_ratio_.sum() * 100

print(f'Original features    : {X_scaled.shape[1]}')
print(f'PCA components chosen: {n_components_chosen}')
print(f'Explained variance   : {explained_var_total:.2f}%')
print(f'Reduced shape        : {X_pca.shape}')

# ## Table 1: PCA Summary

# In[27]:


pca_summary = pd.DataFrame({
    'Components Chosen'     : [n_components_chosen],
    'Explained Variance (%)': [f'{explained_var_total:.2f}%'],
    'Justification'         : [
        f'Retain 95% explained variance; reduced from {X_scaled.shape[1]} '
        f'to {n_components_chosen} components, minimizing information loss '
        f'while reducing dimensionality and multicollinearity.'
    ]
})

print('\n' + '='*80)
print('TABLE 1: PCA Summary')
print('='*80)
print(pca_summary.to_string(index=False))
print('='*80)
pca_summary

# ## 7. Regression Models
# ### Setup: 5-Fold Cross Validation

# In[28]:


kf         = KFold(n_splits=5, shuffle=True, random_state=42)
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
r2_scorer  = make_scorer(r2_score)
print('5-Fold CV for Regression ready.')

# ### 7.1 Linear Regression

# In[29]:


lr = LinearRegression()

# Without PCA
cv_lr_nopca  = cross_validate(lr, X_scaled, y_reg, cv=kf, scoring={'mse': mse_scorer, 'r2': r2_scorer})
lr_mse_nopca = -cv_lr_nopca['test_mse']
lr_r2_nopca  =  cv_lr_nopca['test_r2']

# With PCA
cv_lr_pca  = cross_validate(lr, X_pca, y_reg, cv=kf, scoring={'mse': mse_scorer, 'r2': r2_scorer})
lr_mse_pca = -cv_lr_pca['test_mse']
lr_r2_pca  =  cv_lr_pca['test_r2']

print('LINEAR REGRESSION RESULTS')
print('-' * 65)
print(f"{'Fold':<8} {'MSE (No PCA)':>14} {'MSE (PCA)':>12} {'R2 (No PCA)':>13} {'R2 (PCA)':>11}")
print('-' * 65)
for i in range(5):
    print(f"Fold {i+1:<3} {lr_mse_nopca[i]:>14.4f} {lr_mse_pca[i]:>12.4f} "
          f"{lr_r2_nopca[i]:>13.4f} {lr_r2_pca[i]:>11.4f}")
print('-' * 65)
print(f"{'Avg':<8} {lr_mse_nopca.mean():>14.4f} {lr_mse_pca.mean():>12.4f} "
      f"{lr_r2_nopca.mean():>13.4f} {lr_r2_pca.mean():>11.4f}")
print(f"{'Std':<8} {lr_mse_nopca.std():>14.4f} {lr_mse_pca.std():>12.4f} "
      f"{lr_r2_nopca.std():>13.4f} {lr_r2_pca.std():>11.4f}")

# ### 7.2 Random Forest Regressor

# In[30]:


rf_param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10]}

# Tune Without PCA
rf_grid_nopca = GridSearchCV(RandomForestRegressor(random_state=42),
                              rf_param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
rf_grid_nopca.fit(X_scaled, y_reg)
print(f'Best Params (No PCA)  : {rf_grid_nopca.best_params_}')

# Tune With PCA
rf_grid_pca = GridSearchCV(RandomForestRegressor(random_state=42),
                            rf_param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
rf_grid_pca.fit(X_pca, y_reg)
print(f'Best Params (With PCA): {rf_grid_pca.best_params_}')

# In[31]:


cv_rf_nopca  = cross_validate(rf_grid_nopca.best_estimator_, X_scaled, y_reg, cv=kf,
                               scoring={'mse': mse_scorer, 'r2': r2_scorer})
rf_mse_nopca = -cv_rf_nopca['test_mse']
rf_r2_nopca  =  cv_rf_nopca['test_r2']

cv_rf_pca  = cross_validate(rf_grid_pca.best_estimator_, X_pca, y_reg, cv=kf,
                             scoring={'mse': mse_scorer, 'r2': r2_scorer})
rf_mse_pca = -cv_rf_pca['test_mse']
rf_r2_pca  =  cv_rf_pca['test_r2']

print('RANDOM FOREST REGRESSOR RESULTS')
print('-' * 65)
print(f"{'Fold':<8} {'MSE (No PCA)':>14} {'MSE (PCA)':>12} {'R2 (No PCA)':>13} {'R2 (PCA)':>11}")
print('-' * 65)
for i in range(5):
    print(f"Fold {i+1:<3} {rf_mse_nopca[i]:>14.4f} {rf_mse_pca[i]:>12.4f} "
          f"{rf_r2_nopca[i]:>13.4f} {rf_r2_pca[i]:>11.4f}")
print('-' * 65)
print(f"{'Avg':<8} {rf_mse_nopca.mean():>14.4f} {rf_mse_pca.mean():>12.4f} "
      f"{rf_r2_nopca.mean():>13.4f} {rf_r2_pca.mean():>11.4f}")
print(f"{'Std':<8} {rf_mse_nopca.std():>14.4f} {rf_mse_pca.std():>12.4f} "
      f"{rf_r2_nopca.std():>13.4f} {rf_r2_pca.std():>11.4f}")

# ## Table 2: 5-Fold CV Results – Regression

# In[32]:


def obs_reg(metric, v_no, v_pca):
    diff = v_pca - v_no
    if metric == 'MSE':
        return (f"PCA {'reduces' if diff < 0 else 'increases'} MSE "
                f"({v_no:.4f} -> {v_pca:.4f}); "
                f"{'improved prediction error' if diff < 0 else 'slight info loss'}")
    else:
        return (f"PCA {'improves' if diff > 0 else 'slightly reduces'} R2 "
                f"({v_no:.4f} -> {v_pca:.4f}); "
                f"{'better fit' if diff > 0 else 'minor variance loss'}")

reg_table = pd.DataFrame({
    'Model'             : ['Linear Regression', 'Linear Regression', 'Random Forest', 'Random Forest'],
    'Metric'            : ['MSE', 'R2', 'MSE', 'R2'],
    'Fold Avg (No PCA)' : [
        f'{lr_mse_nopca.mean():.4f} +/- {lr_mse_nopca.std():.4f}',
        f'{lr_r2_nopca.mean():.4f} +/- {lr_r2_nopca.std():.4f}',
        f'{rf_mse_nopca.mean():.4f} +/- {rf_mse_nopca.std():.4f}',
        f'{rf_r2_nopca.mean():.4f} +/- {rf_r2_nopca.std():.4f}'
    ],
    'Fold Avg (With PCA)': [
        f'{lr_mse_pca.mean():.4f} +/- {lr_mse_pca.std():.4f}',
        f'{lr_r2_pca.mean():.4f} +/- {lr_r2_pca.std():.4f}',
        f'{rf_mse_pca.mean():.4f} +/- {rf_mse_pca.std():.4f}',
        f'{rf_r2_pca.mean():.4f} +/- {rf_r2_pca.std():.4f}'
    ],
    'Observation': [
        obs_reg('MSE', lr_mse_nopca.mean(), lr_mse_pca.mean()),
        obs_reg('R2',  lr_r2_nopca.mean(),  lr_r2_pca.mean()),
        obs_reg('MSE', rf_mse_nopca.mean(), rf_mse_pca.mean()),
        obs_reg('R2',  rf_r2_nopca.mean(),  rf_r2_pca.mean())
    ]
})

print('\n' + '='*100)
print('TABLE 2: 5-Fold CV Results - Regression')
print('='*100)
print(reg_table.to_string(index=False))
print('='*100)
reg_table

# In[33]:


# Regression Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
x, w = np.arange(5), 0.35
folds = [f'Fold {i+1}' for i in range(5)]

pairs = [
    (axes[0,0], lr_mse_nopca, lr_mse_pca, 'Linear Regression - MSE',  'MSE'),
    (axes[0,1], lr_r2_nopca,  lr_r2_pca,  'Linear Regression - R2',   'R2 Score'),
    (axes[1,0], rf_mse_nopca, rf_mse_pca, 'Random Forest - MSE',      'MSE'),
    (axes[1,1], rf_r2_nopca,  rf_r2_pca,  'Random Forest - R2',       'R2 Score'),
]
for ax, nopca, wpca, title, ylabel in pairs:
    ax.bar(x - w/2, nopca, w, label='No PCA',   color='steelblue', edgecolor='black')
    ax.bar(x + w/2, wpca,  w, label='With PCA', color='coral',     edgecolor='black')
    ax.set_title(title); ax.set_xticks(x); ax.set_xticklabels(folds)
    ax.set_ylabel(ylabel); ax.legend()

plt.suptitle('Regression: No PCA vs With PCA (5-Fold CV)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('regression_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ## 8. Classification Models
# ### Setup: Stratified 5-Fold CV

# In[34]:


skf       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1_scorer = make_scorer(f1_score, average='weighted')
print('Stratified 5-Fold CV for Classification ready.')
print(f'Classes: {le.classes_}')

# ### 8.1 Logistic Regression

# In[35]:


log_param_grid = {'C': [0.01, 0.1, 1, 10, 100]}

log_grid_nopca = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42),
                               log_param_grid, cv=3, scoring='accuracy', n_jobs=-1)
log_grid_nopca.fit(X_scaled, y_cls)
print(f'Best C (No PCA)  : {log_grid_nopca.best_params_["C"]}')

log_grid_pca = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42),
                             log_param_grid, cv=3, scoring='accuracy', n_jobs=-1)
log_grid_pca.fit(X_pca, y_cls)
print(f'Best C (With PCA): {log_grid_pca.best_params_["C"]}')

# In[36]:


cv_log_nopca  = cross_validate(log_grid_nopca.best_estimator_, X_scaled, y_cls, cv=skf,
                                scoring={'accuracy': 'accuracy', 'f1': f1_scorer})
log_acc_nopca = cv_log_nopca['test_accuracy']
log_f1_nopca  = cv_log_nopca['test_f1']

cv_log_pca  = cross_validate(log_grid_pca.best_estimator_, X_pca, y_cls, cv=skf,
                              scoring={'accuracy': 'accuracy', 'f1': f1_scorer})
log_acc_pca = cv_log_pca['test_accuracy']
log_f1_pca  = cv_log_pca['test_f1']

print('LOGISTIC REGRESSION RESULTS')
print('-' * 65)
print(f"{'Fold':<8} {'Acc (No PCA)':>14} {'Acc (PCA)':>12} {'F1 (No PCA)':>13} {'F1 (PCA)':>11}")
print('-' * 65)
for i in range(5):
    print(f"Fold {i+1:<3} {log_acc_nopca[i]:>14.4f} {log_acc_pca[i]:>12.4f} "
          f"{log_f1_nopca[i]:>13.4f} {log_f1_pca[i]:>11.4f}")
print('-' * 65)
print(f"{'Avg':<8} {log_acc_nopca.mean():>14.4f} {log_acc_pca.mean():>12.4f} "
      f"{log_f1_nopca.mean():>13.4f} {log_f1_pca.mean():>11.4f}")
print(f"{'Std':<8} {log_acc_nopca.std():>14.4f} {log_acc_pca.std():>12.4f} "
      f"{log_f1_nopca.std():>13.4f} {log_f1_pca.std():>11.4f}")

# ### 8.2 Support Vector Machine (SVM)

# In[37]:


svm_param_grid = {
    'kernel': ['linear', 'rbf'],
    'C'     : [0.1, 1, 10],
    'gamma' : ['scale', 'auto']
}

svm_grid_nopca = GridSearchCV(SVC(random_state=42), svm_param_grid,
                               cv=3, scoring='accuracy', n_jobs=-1)
svm_grid_nopca.fit(X_scaled, y_cls)
print(f'Best Params (No PCA)  : {svm_grid_nopca.best_params_}')

svm_grid_pca = GridSearchCV(SVC(random_state=42), svm_param_grid,
                             cv=3, scoring='accuracy', n_jobs=-1)
svm_grid_pca.fit(X_pca, y_cls)
print(f'Best Params (With PCA): {svm_grid_pca.best_params_}')

# In[38]:


cv_svm_nopca  = cross_validate(svm_grid_nopca.best_estimator_, X_scaled, y_cls, cv=skf,
                                scoring={'accuracy': 'accuracy', 'f1': f1_scorer})
svm_acc_nopca = cv_svm_nopca['test_accuracy']
svm_f1_nopca  = cv_svm_nopca['test_f1']

cv_svm_pca  = cross_validate(svm_grid_pca.best_estimator_, X_pca, y_cls, cv=skf,
                              scoring={'accuracy': 'accuracy', 'f1': f1_scorer})
svm_acc_pca = cv_svm_pca['test_accuracy']
svm_f1_pca  = cv_svm_pca['test_f1']

print('SVM RESULTS')
print('-' * 65)
print(f"{'Fold':<8} {'Acc (No PCA)':>14} {'Acc (PCA)':>12} {'F1 (No PCA)':>13} {'F1 (PCA)':>11}")
print('-' * 65)
for i in range(5):
    print(f"Fold {i+1:<3} {svm_acc_nopca[i]:>14.4f} {svm_acc_pca[i]:>12.4f} "
          f"{svm_f1_nopca[i]:>13.4f} {svm_f1_pca[i]:>11.4f}")
print('-' * 65)
print(f"{'Avg':<8} {svm_acc_nopca.mean():>14.4f} {svm_acc_pca.mean():>12.4f} "
      f"{svm_f1_nopca.mean():>13.4f} {svm_f1_pca.mean():>11.4f}")
print(f"{'Std':<8} {svm_acc_nopca.std():>14.4f} {svm_acc_pca.std():>12.4f} "
      f"{svm_f1_nopca.std():>13.4f} {svm_f1_pca.std():>11.4f}")

# ## Table 3: 5-Fold CV Results – Classification

# In[39]:


def obs_cls(metric, v_no, v_pca):
    diff = v_pca - v_no
    if diff > 0.01:
        return f'PCA improves {metric} ({v_no:.4f} -> {v_pca:.4f}); dimensionality reduction helps'
    elif diff < -0.01:
        return f'PCA reduces {metric} ({v_no:.4f} -> {v_pca:.4f}); slight info loss'
    else:
        return f'Minimal change in {metric} ({v_no:.4f} -> {v_pca:.4f}); PCA has little effect'

cls_table = pd.DataFrame({
    'Model'             : ['Logistic Regression', 'Logistic Regression', 'SVM', 'SVM'],
    'Metric'            : ['Accuracy', 'F1-score', 'Accuracy', 'F1-score'],
    'Fold Avg (No PCA)' : [
        f'{log_acc_nopca.mean():.4f} +/- {log_acc_nopca.std():.4f}',
        f'{log_f1_nopca.mean():.4f} +/- {log_f1_nopca.std():.4f}',
        f'{svm_acc_nopca.mean():.4f} +/- {svm_acc_nopca.std():.4f}',
        f'{svm_f1_nopca.mean():.4f} +/- {svm_f1_nopca.std():.4f}'
    ],
    'Fold Avg (With PCA)': [
        f'{log_acc_pca.mean():.4f} +/- {log_acc_pca.std():.4f}',
        f'{log_f1_pca.mean():.4f} +/- {log_f1_pca.std():.4f}',
        f'{svm_acc_pca.mean():.4f} +/- {svm_acc_pca.std():.4f}',
        f'{svm_f1_pca.mean():.4f} +/- {svm_f1_pca.std():.4f}'
    ],
    'Observation': [
        obs_cls('Accuracy', log_acc_nopca.mean(), log_acc_pca.mean()),
        obs_cls('F1',       log_f1_nopca.mean(),  log_f1_pca.mean()),
        obs_cls('Accuracy', svm_acc_nopca.mean(), svm_acc_pca.mean()),
        obs_cls('F1',       svm_f1_nopca.mean(),  svm_f1_pca.mean())
    ]
})

print('\n' + '='*100)
print('TABLE 3: 5-Fold CV Results - Classification')
print('='*100)
print(cls_table.to_string(index=False))
print('='*100)
cls_table

# In[40]:


# Classification Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
x, w = np.arange(5), 0.35

pairs = [
    (axes[0,0], log_acc_nopca, log_acc_pca, 'Logistic Regression - Accuracy', 'Accuracy'),
    (axes[0,1], log_f1_nopca,  log_f1_pca,  'Logistic Regression - F1 Score', 'F1 Score'),
    (axes[1,0], svm_acc_nopca, svm_acc_pca, 'SVM - Accuracy',                 'Accuracy'),
    (axes[1,1], svm_f1_nopca,  svm_f1_pca,  'SVM - F1 Score',                 'F1 Score'),
]
for ax, nopca, wpca, title, ylabel in pairs:
    ax.bar(x - w/2, nopca, w, label='No PCA',   color='steelblue', edgecolor='black')
    ax.bar(x + w/2, wpca,  w, label='With PCA', color='coral',     edgecolor='black')
    ax.set_title(title); ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {i+1}' for i in range(5)])
    ax.set_ylabel(ylabel); ax.legend(); ax.set_ylim([0, 1])

plt.suptitle('Classification: No PCA vs With PCA (5-Fold CV)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('classification_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# In[42]:


# Final consolidated summary table
print('=' * 75)
print('EXPERIMENT 8 - FINAL CONSOLIDATED SUMMARY')
print('=' * 75)
print(f'Dataset              : {df.shape[0]} samples, {len(feature_cols)} features')
print(f'PCA Components Chosen: {n_components_chosen} (retaining {explained_var_total:.2f}% variance)')
print()
print(f"{'Model':<22} {'Metric':<10} {'No PCA':>12} {'With PCA':>12}")
print('-' * 60)
rows = [
    ('Linear Regression',   'MSE',      lr_mse_nopca.mean(),  lr_mse_pca.mean()),
    ('Linear Regression',   'R2',       lr_r2_nopca.mean(),   lr_r2_pca.mean()),
    ('Random Forest',       'MSE',      rf_mse_nopca.mean(),  rf_mse_pca.mean()),
    ('Random Forest',       'R2',       rf_r2_nopca.mean(),   rf_r2_pca.mean()),
    ('Logistic Regression', 'Accuracy', log_acc_nopca.mean(), log_acc_pca.mean()),
    ('Logistic Regression', 'F1-score', log_f1_nopca.mean(),  log_f1_pca.mean()),
    ('SVM',                 'Accuracy', svm_acc_nopca.mean(), svm_acc_pca.mean()),
    ('SVM',                 'F1-score', svm_f1_nopca.mean(),  svm_f1_pca.mean()),
]
for model, metric, no_pca, with_pca in rows:
    print(f'{model:<22} {metric:<10} {no_pca:>12.4f} {with_pca:>12.4f}')
print('=' * 75)
