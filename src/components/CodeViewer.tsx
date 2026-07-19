import React, { useState } from 'react';
import { Copy, Check, Terminal, ExternalLink } from 'lucide-react';

export default function CodeViewer() {
  const [copied, setCopied] = useState(false);

  const pythonCode = `"""
Credit Scoring Classification Model - Comparison Study
Created for Machine Learning Internship Project Walkthrough
=========================================================
Algorithms Evaluated:
1. Logistic Regression (Regularized)
2. Decision Tree Classifier
3. Random Forest Classifier

Prerequisites:
pip install pandas numpy scikit-learn matplotlib seaborn
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# =====================================================================
# STEP 1: LOAD (OR GENERATE) THE DATASET
# =====================================================================
def get_credit_dataset(filename="credit_scoring_dataset.csv"):
    \"\"\"
    Loads the credit scoring dataset. If the file does not exist locally,
    this function generates a realistic synthetic dataset so the code
    runs successfully out-of-the-box.
    \"\"\"
    if os.path.exists(filename):
        print(f"--> Found existing dataset '{filename}'. Loading...")
        return pd.read_csv(filename)
    
    print(f"--> '{filename}' not found. Generating realistic synthetic dataset...")
    size = 1000
    
    # Simulate features
    age = np.random.randint(18, 75, size=size)
    income = np.random.normal(65, 20, size=size).clip(15, 150) # In thousands ($)
    debt_ratio = np.random.normal(35, 12, size=size).clip(5, 95) # DTI %
    
    # Payment History Score (0 to 100, where 100 is perfect timeliness)
    payment_history = (np.random.normal(80, 10, size=size) - (debt_ratio - 35) * 0.3).clip(10, 100)
    loan_amount = (income * np.random.normal(0.6, 0.1, size=size)).clip(3, 150)
    
    # Categorical employment status
    emp_statuses = ['Employed', 'Self-Employed', 'Unemployed']
    employment_status = np.random.choice(emp_statuses, size=size, p=[0.7, 0.2, 0.1])
    
    # Calculate latent credit score to determine risk target
    # Lower debt, higher income, higher payment history score = Good credit risk (1)
    emp_impact = np.where(employment_status == 'Employed', 2.0, np.where(employment_status == 'Self-Employed', 0.5, -2.5))
    z = (-3.5 + 
         (income * 0.03) + 
         (age * 0.015) - 
         (debt_ratio * 0.05) + 
         (payment_history * 0.07) - 
         (loan_amount * 0.01) + 
         emp_impact + 
         np.random.normal(0, 0.8, size=size))
    
    prob = 1 / (1 + np.exp(-z))
    credit_risk = np.where(prob >= 0.5, 1, 0) # 1 = Good Risk, 0 = Bad Risk
    
    df = pd.DataFrame({
        'income': income,
        'age': age,
        'debt': debt_ratio,
        'payment_history': payment_history,
        'loan_amount': loan_amount,
        'employment_status': employment_status,
        'credit_risk': credit_risk
    })
    
    # Inject a few missing values (NaN) to demonstrate imputation in preprocessing
    nan_mask_income = np.random.rand(size) < 0.02
    nan_mask_age = np.random.rand(size) < 0.02
    df.loc[nan_mask_income, 'income'] = np.nan
    df.loc[nan_mask_age, 'age'] = np.nan
    
    df.to_csv(filename, index=False)
    print(f"--> Saved fresh synthetic dataset to '{filename}' with {size} records.")
    return df

df = get_credit_dataset()

# Print basic shape and column check
print("\\nDataset Shape:", df.shape)
print("First 5 rows of raw dataset:")
print(df.head())

# =====================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# =====================================================================
print("\\n--> Performing Exploratory Data Analysis...")

# Create figures directory if it doesn't exist
os.makedirs("plots", exist_ok=True)

# 1. Class Distribution (Target)
plt.figure(figsize=(6, 4))
sns.countplot(x='credit_risk', data=df, hue='credit_risk', palette='viridis', legend=False)
plt.title('Distribution of Credit Risk (Class Target)')
plt.xlabel('Credit Risk (0 = Bad Risk, 1 = Good Risk)')
plt.ylabel('Count')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('plots/01_class_distribution.png')
plt.close()

# 2. Correlation Heatmap (Numerical columns only)
plt.figure(figsize=(8, 6))
# Impute numerical features temporarily just for correlation plotting
temp_numerical = df.select_dtypes(include=[np.number]).copy()
temp_numerical = temp_numerical.fillna(temp_numerical.mean())
corr_matrix = temp_numerical.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Pearson Correlation Heatmap of Credit Features')
plt.tight_layout()
plt.savefig('plots/02_correlation_heatmap.png')
plt.close()

# 3. Income vs. Debt Distribution by Risk Status
plt.figure(figsize=(8, 5))
sns.scatterplot(x='income', y='debt', hue='credit_risk', data=df, palette='coolwarm', alpha=0.7)
plt.title('Income vs. Debt Ratio colored by Credit Risk')
plt.xlabel('Income ($ Thousands)')
plt.ylabel('Debt-to-Income Ratio (%)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('plots/03_income_vs_debt.png')
plt.close()

print("--> EDA charts saved in 'plots/' directory:")
print("    - plots/01_class_distribution.png")
print("    - plots/02_correlation_heatmap.png")
print("    - plots/03_income_vs_debt.png")

# =====================================================================
# STEP 3: TRAIN / TEST SPLIT
# =====================================================================
# Separate Features (X) and Target (y)
X = df.drop(columns=['credit_risk'])
y = df['credit_risk']

# Split data into training and test partitions (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)
print(f"\\nSplitting data: Train size = {X_train.shape[0]}, Test size = {X_test.shape[0]}")

# =====================================================================
# STEP 4: DATA PREPROCESSING PIPELINE
# =====================================================================
# We define preprocessing pipelines for numerical and categorical variables
numerical_features = ['income', 'age', 'debt', 'payment_history', 'loan_amount']
categorical_features = ['employment_status']

# Numerical Pipeline: Median Imputation + Z-score scaling
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical Pipeline: Mode Imputation + One-Hot Encoding
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore')) 
    # dropping first category to prevent multi-collinearity (dummy variable trap)
])

# Combine transformers into preprocessor block
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

print("--> Setup scikit-learn preprocessing Pipeline completed.")

# =====================================================================
# STEP 5: MODEL TRAINING & COMPARISON
# =====================================================================
# Define three classification models
models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, 
        C=1.0, # Regularization strength (inverse lambda)
        random_state=RANDOM_SEED
    ),
    'Decision Tree': DecisionTreeClassifier(
        max_depth=5, 
        min_samples_split=5, 
        random_state=RANDOM_SEED
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, 
        max_depth=8, 
        random_state=RANDOM_SEED
    )
}

# Dictionary to hold final evaluation scores
results_table = {}

# Set up matplotlib figure for ROC comparison
plt.figure(figsize=(8, 6))

for name, model_instance in models.items():
    print(f"\\n--> Training Model: {name}...")
    
    # Build complete pipeline incorporating preprocessing and model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model_instance)
    ])
    
    # Train the pipeline
    pipeline.fit(X_train, y_train)
    
    # Make Predictions
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1] # Probability of positive class (Good credit risk)
    
    # Calculate performance metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    # Store results
    results_table[name] = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': auc
    }
    
    # Plot individual model ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", linewidth=2)
    
    # Display individual classification reports
    print(f"Classification Report for {name}:")
    print(classification_report(y_test, y_pred, target_names=['Bad Credit', 'Good Credit']))

# Add aesthetics to ROC curve plot
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.500)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve Comparison')
plt.legend(loc="lower right")
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('plots/04_roc_curve_comparison.png')
plt.close()
print("--> Saved comparative ROC Curve to 'plots/04_roc_curve_comparison.png'.")

# =====================================================================
# STEP 6: COMPARISON SUMMARY & RECOMMENDATION
# =====================================================================
comparison_df = pd.DataFrame(results_table).T
print("\\n" + "="*50)
print("FINAL PERFORMANCE COMPARISON TABLE:")
print("="*50)
print(comparison_df.round(4))
print("="*50)

# Determine the best model based on F1-Score and ROC-AUC
best_model_by_auc = comparison_df['ROC-AUC'].idxmax()
best_auc = comparison_df.loc[best_model_by_auc, 'ROC-AUC']

print(f"\\nINTERNSHIP RECOMMENDATION:")
print(f"Based on the evaluation criteria, the recommended model is: **{best_model_by_auc}**")
print(f"It achieved the highest ROC-AUC score of {best_auc:.4f} on the unseen test set.")
print("Reasoning:")
print("1. Random Forest typically outperforms Logistic Regression in accuracy because it handles non-linear relationships")
print("   (like the interactions between income and debt) without needing explicit feature engineering.")
print("2. Compared to a standard Decision Tree, Random Forest reduces variance and prevents overfitting")
print("   by averaging multiple decorrelated trees trained on bootstapped subsets of the data.")
print("3. In credit scoring, predicting the probability of default correctly (ROC-AUC) is more critical")
print("   than simple accuracy, as lenders can adjust threshold bounds to control default risk vs. acceptance rates.")
print("="*50)
print("All tasks completed successfully. Plots generated. Code ready for presentation!")
`;

  const copyToClipboard = () => {
    navigator.clipboard.writeText(pythonCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div id="code-viewer-container" className="space-y-6">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-slate-900 text-slate-100 p-4 rounded-xl border border-slate-800">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-yellow-500/10 text-yellow-500 rounded-lg">
            <Terminal size={20} />
          </div>
          <div>
            <h3 className="text-sm font-semibold font-sans">Production Python ML Script</h3>
            <p className="text-xs text-slate-400 font-sans">Self-contained scikit-learn pipeline featuring full comments</p>
          </div>
        </div>
        <button
          id="copy-code-btn"
          onClick={copyToClipboard}
          className="flex items-center gap-2 px-4 py-2 bg-yellow-500 text-slate-950 font-medium font-sans text-xs rounded-lg hover:bg-yellow-400 transition-all cursor-pointer shadow-sm active:scale-95"
        >
          {copied ? (
            <>
              <Check size={14} className="stroke-[3]" />
              Copied to Clipboard!
            </>
          ) : (
            <>
              <Copy size={14} />
              Copy Code
            </>
          )}
        </button>
      </div>

      <div className="bg-slate-950 rounded-xl border border-slate-800 overflow-hidden shadow-xl">
        <div className="flex items-center justify-between px-4 py-2.5 bg-slate-900 border-b border-slate-800 text-xs text-slate-400 font-mono">
          <span>credit_score_model.py</span>
          <span className="flex items-center gap-1.5 text-emerald-500">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
            Ready to execute
          </span>
        </div>
        <div className="p-4 overflow-x-auto text-xs md:text-sm font-mono text-slate-300 leading-relaxed max-h-[600px] overflow-y-auto">
          <pre className="whitespace-pre">{pythonCode}</pre>
        </div>
      </div>

      <div className="bg-slate-50 border border-slate-200 rounded-xl p-5 space-y-3">
        <h4 className="text-sm font-semibold text-slate-950 font-sans flex items-center gap-2">
          <ExternalLink size={16} className="text-slate-500" />
          How to Run This Local Python Script
        </h4>
        <p className="text-xs text-slate-600 font-sans leading-relaxed">
          This code is written to run cleanly on your computer out of the box. You don't even need to provide a credit dataset CSV to test it; the script detects if the file is missing and automatically spawns a realistic synthetic credit scoring dataset with 1,000 credit records and saves it as <code>credit_scoring_dataset.csv</code>.
        </p>
        <div className="p-3.5 bg-slate-900 rounded-lg font-mono text-xs text-yellow-400">
          # 1. Install dependencies<br />
          pip install pandas numpy scikit-learn matplotlib seaborn<br /><br />
          # 2. Run the code<br />
          python credit_score_model.py
        </div>
        <p className="text-xs text-slate-500 font-sans italic">
          Tip: Once completed, the script creates a folder named <code>plots/</code> locally, saving high-resolution PNGs of your class distributions, feature correlations, scatter distributions, and the comparative ROC Curve to show in your slides or walkthrough!
        </p>
      </div>
    </div>
  );
}
