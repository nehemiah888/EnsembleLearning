#!/usr/bin/env python3
"""
Employee Turnover Prediction using Decision Trees and Random Forest
使用决策树和随机森林预测员工离职率

This script helps HR departments understand why employees leave and predict
the likelihood of an employee leaving the company.
Data source: https://www.kaggle.com/ludobenistant/hr-analytics

Adapted for Python 3.12 compatibility.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplot
import seaborn as sns
import scipy.stats as stats
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report, precision_score, 
                           recall_score, confusion_matrix, precision_recall_curve, 
                           roc_auc_score, roc_curve)
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from io import StringIO  # Replaced sklearn.externals.six.StringIO
from IPython.display import Image
import pydotplus
import warnings

# Suppress specific warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)


class EmployeeTurnoverPredictor:
    """
    A class to predict employee turnover using Decision Trees and Random Forest.
    """
    
    def __init__(self, data_file='HR_comma_sep.csv'):
        """Initialize the predictor with data file."""
        self.data_file = data_file
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.dtree = None
        self.rf = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess the HR data."""
        print("Loading and preprocessing data...")
        
        # Read data into Pandas DataFrame
        self.df = pd.read_csv(self.data_file, index_col=None)
        
        # Check for missing data
        print(f"Missing data check: {self.df.isnull().any().any()}")
        
        # Rename columns for better readability
        self.df = self.df.rename(columns={
            'satisfaction_level': 'satisfaction', 
            'last_evaluation': 'evaluation',
            'number_project': 'projectCount',
            'average_montly_hours': 'averageMonthlyHours',
            'time_spend_company': 'yearsAtCompany',
            'Work_accident': 'workAccident',
            'promotion_last_5years': 'promotion',
            'sales': 'department',
            'left': 'turnover'
        })
        
        # Move prediction target 'turnover' to first column
        front = self.df['turnover']
        self.df.drop(labels=['turnover'], axis=1, inplace=True)
        self.df.insert(0, 'turnover', front)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Data types:\n{self.df.dtypes}")
        
        return self.df
        
    def analyze_data(self):
        """Perform statistical analysis on the data."""
        print("\n=== Data Analysis ===")
        
        # Turnover rate
        turnover_rate = self.df.turnover.value_counts() / len(self.df)
        print(f"Turnover rate:\n{turnover_rate}")
        
        # Statistical summary
        print(f"\nStatistical summary:\n{self.df.describe()}")
        
        # Grouped statistics by turnover (only numeric columns)
        turnover_summary = self.df.groupby('turnover')
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        print(f"\nGrouped statistics by turnover (numeric columns only):\n{turnover_summary[numeric_columns].mean()}")
        
    def analyze_correlation(self):
        """Analyze feature correlations."""
        print("\n=== Correlation Analysis ===")
        
        # Correlation matrix (only numeric columns)
        numeric_df = self.df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, 
                    xticklabels=corr.columns.values,
                    yticklabels=corr.columns.values,
                    annot=True, cmap='coolwarm')
        plt.title('Feature Correlation Matrix (Numeric Features)')
        plt.tight_layout()
        plt.show()
        
        return corr
        
    def perform_satisfaction_analysis(self):
        """Analyze satisfaction levels between turnover groups."""
        print("\n=== Satisfaction Analysis ===")
        
        # Compare satisfaction between turnover groups
        emp_population = self.df['satisfaction'][self.df['turnover'] == 0].mean()
        emp_turnover_satisfaction = self.df[self.df['turnover'] == 1]['satisfaction'].mean()
        
        print(f'未离职员工满意度: {emp_population}')
        print(f'离职员工满意度: {emp_turnover_satisfaction}')
        
        # Perform t-test
        t_stat, p_value = stats.ttest_1samp(
            a=self.df[self.df['turnover'] == 1]['satisfaction'],
            popmean=emp_population
        )
        print(f"T-test results: t-statistic={t_stat:.4f}, p-value={p_value:.2e}")
        
        # Calculate confidence intervals
        degree_freedom = len(self.df[self.df['turnover'] == 1])
        LQ = stats.t.ppf(0.025, degree_freedom)
        RQ = stats.t.ppf(0.975, degree_freedom)
        print(f'95% CI left boundary: {LQ}')
        print(f'95% CI right boundary: {RQ}')
        
    def plot_distributions(self):
        """Plot probability density functions for key features."""
        print("\n=== Plotting Distributions ===")
        
        # Set up the plotting style
        plt.style.use('default')
        
        # Work evaluation distribution
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Evaluation
        sns.kdeplot(data=self.df[self.df['turnover'] == 0], x='evaluation', 
                   color='b', fill=True, label='no turnover', ax=axes[0])
        sns.kdeplot(data=self.df[self.df['turnover'] == 1], x='evaluation', 
                   color='r', fill=True, label='turnover', ax=axes[0])
        axes[0].set_xlabel('Work Evaluation')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Work Evaluation Distribution - Turnover vs No Turnover')
        axes[0].legend()
        
        # Average monthly hours
        sns.kdeplot(data=self.df[self.df['turnover'] == 0], x='averageMonthlyHours', 
                   color='b', fill=True, label='no turnover', ax=axes[1])
        sns.kdeplot(data=self.df[self.df['turnover'] == 1], x='averageMonthlyHours', 
                   color='r', fill=True, label='turnover', ax=axes[1])
        axes[1].set_xlabel('Average Monthly Hours')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Average Monthly Hours Distribution - Turnover vs No Turnover')
        axes[1].legend()
        
        # Satisfaction
        sns.kdeplot(data=self.df[self.df['turnover'] == 0], x='satisfaction', 
                   color='b', fill=True, label='no turnover', ax=axes[2])
        sns.kdeplot(data=self.df[self.df['turnover'] == 1], x='satisfaction', 
                   color='r', fill=True, label='turnover', ax=axes[2])
        axes[2].set_xlabel('Employee Satisfaction')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Employee Satisfaction Distribution - Turnover vs No Turnover')
        axes[2].legend()
        
        plt.tight_layout()
        plt.show()
        
    def prepare_features(self):
        """Prepare features for machine learning."""
        print("\n=== Preparing Features ===")
        
        # Make a copy to avoid modifying the original data during analysis
        df_processed = self.df.copy()
        
        # Convert string categories to integers
        df_processed["department"] = df_processed["department"].astype('category').cat.codes
        df_processed["salary"] = df_processed["salary"].astype('category').cat.codes
        
        # Create feature matrix X and target vector y
        target_name = 'turnover'
        X = df_processed.drop('turnover', axis=1)
        y = df_processed[target_name]
        
        # Split data into training and testing sets
        # stratify=y ensures the same percentage of turnover in train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.15, random_state=123, stratify=y)
        
        print(f"Training set size: {self.X_train.shape}")
        print(f"Test set size: {self.X_test.shape}")
        
    def train_decision_tree(self):
        """Train a Decision Tree classifier."""
        print("\n=== Training Decision Tree ===")
        
        # Initialize Decision Tree
        self.dtree = DecisionTreeClassifier(
            criterion='entropy',
            min_weight_fraction_leaf=0.01  # Prevent overfitting
        )
        
        # Train the model
        self.dtree.fit(self.X_train, self.y_train)
        
        # Calculate metrics
        dt_predictions = self.dtree.predict(self.X_test)
        dt_roc_auc = roc_auc_score(self.y_test, dt_predictions)
        
        print(f"Decision Tree AUC = {dt_roc_auc:.2f}")
        print("Classification Report:")
        print(classification_report(self.y_test, dt_predictions))
        
        return dt_roc_auc
        
    def train_random_forest(self):
        """Train a Random Forest classifier."""
        print("\n=== Training Random Forest ===")
        
        # Initialize Random Forest
        self.rf = RandomForestClassifier(
            criterion='entropy',
            n_estimators=100,  # Increased from 3 for better performance
            max_depth=None,
            min_samples_split=10,
            random_state=123
        )
        
        # Train the model
        self.rf.fit(self.X_train, self.y_train)
        
        # Calculate metrics
        rf_predictions = self.rf.predict(self.X_test)
        rf_roc_auc = roc_auc_score(self.y_test, rf_predictions)
        
        print(f"Random Forest AUC = {rf_roc_auc:.2f}")
        print("Classification Report:")
        print(classification_report(self.y_test, rf_predictions))
        
        return rf_roc_auc
        
    def visualize_decision_tree(self, max_depth=3):
        """Visualize the decision tree (simplified version for readability)."""
        print("\n=== Visualizing Decision Tree ===")
        
        # Create a simplified tree for visualization
        simple_tree = DecisionTreeClassifier(
            criterion='entropy',
            max_depth=max_depth,
            min_weight_fraction_leaf=0.01,
            random_state=123
        )
        simple_tree.fit(self.X_train, self.y_train)
        
        try:
            # Feature names from training data
            feature_names = self.X_train.columns
            
            # Export to DOT format
            dot_data = StringIO()
            export_graphviz(simple_tree, out_file=dot_data,
                          filled=True, rounded=True,
                          special_characters=True,
                          feature_names=feature_names,
                          class_names=['Stay', 'Leave'])
            
            # Create graph
            graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
            graph.write_png('decision_tree.png')
            
            print("Decision tree visualization saved as 'decision_tree.png'")
            return Image(graph.create_png())
            
        except Exception as e:
            print(f"Could not create tree visualization: {e}")
            print("Please ensure graphviz and pydotplus are installed")
            
    def plot_feature_importance(self):
        """Plot feature importance for both models."""
        print("\n=== Feature Importance Analysis ===")
        
        if self.dtree is None or self.rf is None:
            print("Models need to be trained first!")
            return
            
        # Get feature names from training data
        feat_names = self.X_train.columns
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        
        # Decision Tree feature importance
        dt_importances = self.dtree.feature_importances_
        dt_indices = np.argsort(dt_importances)[::-1]
        
        ax1.bar(range(len(dt_indices)), dt_importances[dt_indices], 
               color='lightblue', align="center")
        ax1.step(range(len(dt_indices)), np.cumsum(dt_importances[dt_indices]), 
                where='mid', label='Cumulative')
        ax1.set_xticks(range(len(dt_indices)))
        ax1.set_xticklabels(feat_names[dt_indices], rotation='vertical', fontsize=12)
        ax1.set_title("Feature Importance - Decision Tree")
        ax1.legend()
        
        # Random Forest feature importance
        rf_importances = self.rf.feature_importances_
        rf_indices = np.argsort(rf_importances)[::-1]
        
        ax2.bar(range(len(rf_indices)), rf_importances[rf_indices], 
               color='lightgreen', align="center")
        ax2.step(range(len(rf_indices)), np.cumsum(rf_importances[rf_indices]), 
                where='mid', label='Cumulative')
        ax2.set_xticks(range(len(rf_indices)))
        ax2.set_xticklabels(feat_names[rf_indices], rotation='vertical', fontsize=12)
        ax2.set_title("Feature Importance - Random Forest")
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
    def plot_roc_curves(self):
        """Plot ROC curves for both models."""
        print("\n=== ROC Curve Analysis ===")
        
        if self.dtree is None or self.rf is None:
            print("Models need to be trained first!")
            return
            
        # Calculate ROC curves
        rf_fpr, rf_tpr, _ = roc_curve(self.y_test, self.rf.predict_proba(self.X_test)[:, 1])
        dt_fpr, dt_tpr, _ = roc_curve(self.y_test, self.dtree.predict_proba(self.X_test)[:, 1])
        
        # Calculate AUC scores
        rf_auc = roc_auc_score(self.y_test, self.rf.predict(self.X_test))
        dt_auc = roc_auc_score(self.y_test, self.dtree.predict(self.X_test))
        
        # Plot ROC curves
        plt.figure(figsize=(10, 8))
        plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.2f})', linewidth=2)
        plt.plot(dt_fpr, dt_tpr, label=f'Decision Tree (AUC = {dt_auc:.2f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.show()
        
    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        print("Starting Employee Turnover Prediction Analysis...")
        print("=" * 60)
        
        # Load and preprocess data
        self.load_and_preprocess_data()
        
        # Analyze data
        self.analyze_data()
        
        # Correlation analysis
        self.analyze_correlation()
        
        # Satisfaction analysis
        self.perform_satisfaction_analysis()
        
        # Plot distributions
        self.plot_distributions()
        
        # Prepare features
        self.prepare_features()
        
        # Train models
        dt_auc = self.train_decision_tree()
        rf_auc = self.train_random_forest()
        
        # Visualizations
        self.visualize_decision_tree()
        self.plot_feature_importance()
        self.plot_roc_curves()
        
        print("\n" + "=" * 60)
        print("Analysis Complete!")
        print(f"Decision Tree AUC: {dt_auc:.3f}")
        print(f"Random Forest AUC: {rf_auc:.3f}")
        print("=" * 60)


def main():
    """Main function to run the analysis."""
    # Create predictor instance
    predictor = EmployeeTurnoverPredictor()
    
    # Run complete analysis
    predictor.run_complete_analysis()


if __name__ == "__main__":
    main()
