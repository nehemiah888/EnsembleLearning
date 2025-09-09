#!/usr/bin/env python3
"""
XGBoost Implementation and Demonstration
XGBoost 应用实践

This script demonstrates the use of XGBoost for both classification and regression
using multiple datasets and interfaces. Adapted for Python 3.12 compatibility.
"""

import time
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class XGBoostDemo:
    """
    A comprehensive class to demonstrate XGBoost for both classification and regression.
    """
    
    def __init__(self, random_state=42):
        """Initialize the XGBoost demo with random state for reproducibility."""
        self.random_state = random_state
        self.models = {}
        self.results = {}
        
    def demo_classification_native(self):
        """Demonstrate XGBoost native interface for classification."""
        print("=" * 60)
        print("XGBoost Native Interface - Classification (Iris Dataset)")
        print("=" * 60)
        
        # Load Iris dataset
        print("\n=== Loading Iris Dataset ===")
        iris = load_iris()
        X, y = iris.data, iris.target
        feature_names = iris.feature_names
        
        print(f"Dataset shape: {X.shape}")
        print(f"Classes: {iris.target_names}")
        print(f"Features: {feature_names}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        print(f"Training set size: {X_train.shape}")
        print(f"Test set size: {X_test.shape}")
        
        # Set parameters for multi-class classification
        print("\n=== Setting Parameters ===")
        params = {
            # General parameters
            'booster': 'gbtree',        # Tree-based model
            'nthread': 4,               # Number of threads
            'verbosity': 0,             # Updated from 'silent' for XGBoost >= 1.0
            'random_state': self.random_state,
            
            # Task parameters
            'objective': 'multi:softmax',  # Multi-class classification
            'num_class': 3,               # Number of classes
            
            # Boosting parameters
            'gamma': 0.1,                 # Minimum loss reduction
            'max_depth': 6,               # Maximum tree depth
            'reg_lambda': 2,              # L2 regularization (updated from 'lambda')
            'subsample': 0.7,             # Subsample ratio
            'colsample_bytree': 0.7,      # Feature subsample ratio
            'min_child_weight': 3,        # Minimum child weight
            'eta': 0.1,                   # Learning rate
        }
        
        print("Parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        # Convert to DMatrix format
        print("\n=== Converting to DMatrix Format ===")
        dtrain = xgb.DMatrix(X_train, y_train, feature_names=feature_names)
        dtest = xgb.DMatrix(X_test, feature_names=feature_names)
        
        # Train model
        print("\n=== Training Model ===")
        num_rounds = 50
        start_time = time.time()
        
        model = xgb.train(params, dtrain, num_rounds)
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Make predictions
        print("\n=== Making Predictions ===")
        y_pred = model.predict(dtest)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")
        
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=iris.target_names))
        
        # Visualizations
        self._plot_feature_importance(model, "XGBoost Classification - Feature Importance")
        self._plot_tree_structure(model, "XGBoost Classification - Tree Structure", num_trees=2)
        
        # Save model info
        self.models['classification_native'] = model
        self.results['classification_native'] = {
            'accuracy': accuracy,
            'model': 'XGBoost Native Classification'
        }
        
        return model, accuracy
        
    def demo_regression_native(self):
        """Demonstrate XGBoost native interface for regression using synthetic data."""
        print("\n" + "=" * 60)
        print("XGBoost Native Interface - Regression (Synthetic Dataset)")
        print("=" * 60)
        
        # Create synthetic regression dataset
        print("\n=== Creating Synthetic Regression Dataset ===")
        np.random.seed(self.random_state)
        n_samples = 1000
        n_features = 10
        
        X = np.random.randn(n_samples, n_features)
        # Create a complex non-linear relationship
        y = (X[:, 0]**2 + X[:, 1] * X[:, 2] + 
             np.sin(X[:, 3]) + X[:, 4] * X[:, 5] + 
             0.1 * np.random.randn(n_samples))
        
        feature_names = [f'feature_{i}' for i in range(n_features)]
        
        print(f"Dataset shape: {X.shape}")
        print(f"Target statistics - Mean: {y.mean():.3f}, Std: {y.std():.3f}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        # Set regression parameters
        print("\n=== Setting Regression Parameters ===")
        params = {
            'booster': 'gbtree',
            'objective': 'reg:squarederror',  # Updated objective for regression
            'gamma': 0.1,
            'max_depth': 5,
            'reg_lambda': 3,                 # Updated from 'lambda'
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'min_child_weight': 3,
            'verbosity': 0,                  # Updated from 'silent'
            'eta': 0.1,
            'random_state': self.random_state,
            'nthread': 4,
        }
        
        # Convert to DMatrix
        dtrain = xgb.DMatrix(X_train, y_train, feature_names=feature_names)
        dtest = xgb.DMatrix(X_test, feature_names=feature_names)
        
        # Train model
        print("\n=== Training Regression Model ===")
        num_rounds = 100
        model = xgb.train(params, dtrain, num_rounds)
        
        # Make predictions
        y_pred = model.predict(dtest)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Test MSE: {mse:.4f}")
        print(f"Test R²: {r2:.4f}")
        
        # Plot actual vs predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'XGBoost Regression: Actual vs Predicted (R² = {r2:.3f})')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Visualizations
        self._plot_feature_importance(model, "XGBoost Regression - Feature Importance")
        
        self.models['regression_native'] = model
        self.results['regression_native'] = {
            'mse': mse,
            'r2': r2,
            'model': 'XGBoost Native Regression'
        }
        
        return model, r2
        
    def demo_classification_sklearn(self):
        """Demonstrate XGBoost sklearn interface for classification."""
        print("\n" + "=" * 60)
        print("XGBoost Sklearn Interface - Classification (Breast Cancer)")
        print("=" * 60)
        
        # Load breast cancer dataset
        print("\n=== Loading Breast Cancer Dataset ===")
        cancer = load_breast_cancer()
        X, y = cancer.data, cancer.target
        feature_names = cancer.feature_names
        
        print(f"Dataset shape: {X.shape}")
        print(f"Classes: {cancer.target_names}")
        print(f"Number of features: {len(feature_names)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        # Train XGBoost classifier using sklearn interface
        print("\n=== Training XGBoost Classifier (Sklearn Interface) ===")
        model = xgb.XGBClassifier(
            max_depth=5,
            n_estimators=100,           # Updated from deprecated parameters
            random_state=self.random_state,
            objective='binary:logistic',
            eval_metric='logloss'       # Added explicit eval_metric
        )
        
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=cancer.target_names))
        
        # Plot feature importance
        feature_importance = model.feature_importances_
        indices = np.argsort(feature_importance)[::-1][:10]  # Top 10 features
        
        plt.figure(figsize=(12, 8))
        plt.title("Top 10 Most Important Features (XGBoost Sklearn)", fontsize=14, fontweight='bold')
        bars = plt.bar(range(10), feature_importance[indices], color='steelblue', alpha=0.7)
        plt.xticks(range(10), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.ylabel('Feature Importance', fontsize=12)
        plt.xlabel('Features', fontsize=12)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # Save and show
        plt.savefig('xgboost_sklearn_classification_feature_importance.png', dpi=150, bbox_inches='tight')
        print("Feature importance plot saved as: xgboost_sklearn_classification_feature_importance.png")
        plt.show()
        
        # ROC Curve
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - XGBoost Classification')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.show()
        
        self.models['classification_sklearn'] = model
        self.results['classification_sklearn'] = {
            'accuracy': accuracy,
            'auc': roc_auc,
            'model': 'XGBoost Sklearn Classification'
        }
        
        return model, accuracy
        
    def demo_regression_sklearn(self):
        """Demonstrate XGBoost sklearn interface for regression."""
        print("\n" + "=" * 60)
        print("XGBoost Sklearn Interface - Regression")
        print("=" * 60)
        
        # Create a more complex synthetic dataset
        print("\n=== Creating Complex Synthetic Dataset ===")
        np.random.seed(self.random_state)
        n_samples = 1500
        n_features = 15
        
        X = np.random.randn(n_samples, n_features)
        # More complex relationship
        y = (2 * X[:, 0]**2 + 3 * X[:, 1] * X[:, 2] + 
             np.sin(5 * X[:, 3]) + np.cos(3 * X[:, 4]) +
             X[:, 5] * X[:, 6] * X[:, 7] + 
             np.sqrt(np.abs(X[:, 8])) +
             0.1 * np.random.randn(n_samples))
        
        print(f"Dataset shape: {X.shape}")
        print(f"Target range: [{y.min():.3f}, {y.max():.3f}]")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        # Train XGBoost regressor
        print("\n=== Training XGBoost Regressor (Sklearn Interface) ===")
        model = xgb.XGBRegressor(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=150,
            random_state=self.random_state,
            objective='reg:squarederror'
        )
        
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Test MSE: {mse:.4f}")
        print(f"Test R²: {r2:.4f}")
        
        # Visualizations
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Actual vs Predicted
        ax1.scatter(y_test, y_pred, alpha=0.6)
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax1.set_xlabel('Actual Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title(f'Actual vs Predicted (R² = {r2:.3f})')
        ax1.grid(True, alpha=0.3)
        
        # Residuals
        residuals = y_test - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.6)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Predicted Values')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residual Plot')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        self.models['regression_sklearn'] = model
        self.results['regression_sklearn'] = {
            'mse': mse,
            'r2': r2,
            'model': 'XGBoost Sklearn Regression'
        }
        
        return model, r2
        
    def _plot_feature_importance(self, model, title):
        """Helper method to plot feature importance."""
        try:
            # Get feature importance data
            importance_dict = model.get_score(importance_type='weight')
            if not importance_dict:
                print("No feature importance data available")
                return
            
            # Close any existing figures first
            plt.close('all')
            
            # Let plot_importance create its own figure (this avoids the blank figure issue)
            ax = plot_importance(model, max_num_features=10, importance_type='weight', 
                               show_values=True, height=0.8)
            
            # Customize the current figure
            fig = plt.gcf()  # Get current figure
            fig.set_size_inches(12, 8)
            
            plt.title(title, fontsize=14, fontweight='bold')
            plt.xlabel('Feature Importance (Weight)', fontsize=12)
            plt.ylabel('Features', fontsize=12)
            plt.tight_layout()
            
            # Save the plot to file and display
            safe_filename = title.replace(' ', '_').replace('-', '_').lower() + '.png'
            plt.savefig(safe_filename, dpi=150, bbox_inches='tight')
            print(f"Feature importance plot saved as: {safe_filename}")
            
            # Try to show the plot
            plt.show()
            
        except Exception as e:
            print(f"Could not plot feature importance: {e}")
            # Fallback: print importance values
            try:
                importance_dict = model.get_score(importance_type='weight')
                print(f"\n{title} - Feature Importance Values:")
                for feature, importance in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {feature}: {importance}")
            except Exception as fallback_e:
                print(f"Could not get feature importance data: {fallback_e}")
            
    def _plot_tree_structure(self, model, title, num_trees=0):
        """Helper method to plot tree structure."""
        try:
            # Check if graphviz is available
            import graphviz
            
            plt.figure(figsize=(20, 10))
            plot_tree(model, num_trees=num_trees)
            plt.title(f"{title} (Tree {num_trees})", fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save the plot
            safe_filename = f"{title.replace(' ', '_').replace('-', '_').lower()}_tree_{num_trees}.png"
            plt.savefig(safe_filename, dpi=150, bbox_inches='tight')
            print(f"Tree structure plot saved as: {safe_filename}")
            
            plt.show()
            
        except ImportError:
            print(f"Skipping tree visualization: graphviz not installed")
            print("To install graphviz: conda install python-graphviz")
            # Close any opened figure to prevent blank plots
            plt.close()
            
        except Exception as e:
            print(f"Could not plot tree structure: {e}")
            print("Tree visualization requires graphviz to be installed")
            # Close any opened figure to prevent blank plots
            plt.close()
            
    def save_models(self):
        """Save trained models to files."""
        print("\n=== Saving Models ===")
        
        for model_name, model in self.models.items():
            filename = f"xgboost_{model_name}.json"
            try:
                if hasattr(model, 'save_model'):
                    model.save_model(filename)
                    print(f"Saved {model_name} to {filename}")
                else:
                    # For sklearn interface models
                    import joblib
                    joblib.dump(model, f"xgboost_{model_name}.pkl")
                    print(f"Saved {model_name} to xgboost_{model_name}.pkl")
            except Exception as e:
                print(f"Could not save {model_name}: {e}")
                
    def print_summary(self):
        """Print summary of all experiments."""
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        
        for experiment, results in self.results.items():
            print(f"\n{results['model']}:")
            for metric, value in results.items():
                if metric != 'model':
                    print(f"  {metric.upper()}: {value:.4f}")
                    
    def run_all_demos(self):
        """Run all XGBoost demonstrations."""
        print("Starting Comprehensive XGBoost Demonstration")
        print("=" * 80)
        
        # Run all demonstrations
        self.demo_classification_native()
        self.demo_regression_native()
        self.demo_classification_sklearn()
        self.demo_regression_sklearn()
        
        # Save models and print summary
        self.save_models()
        self.print_summary()
        
        print("\n" + "=" * 80)
        print("All XGBoost Demonstrations Complete!")
        print("=" * 80)


def compare_xgboost_parameters():
    """Compare XGBoost performance with different hyperparameters."""
    print("\n" + "=" * 60)
    print("XGBoost Hyperparameter Comparison")
    print("=" * 60)
    
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Parameter combinations
    param_combinations = [
        {'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 100},
        {'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 200},
        {'max_depth': 7, 'learning_rate': 0.2, 'n_estimators': 50},
        {'max_depth': 4, 'learning_rate': 0.15, 'n_estimators': 150},
    ]
    
    results = []
    
    for i, params in enumerate(param_combinations):
        print(f"\nTesting combination {i+1}: {params}")
        
        model = xgb.XGBClassifier(random_state=42, **params)
        
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        results.append({
            'params': params,
            'accuracy': accuracy,
            'training_time': training_time
        })
        
        print(f"  Accuracy: {accuracy:.4f}, Training time: {training_time:.2f}s")
    
    # Find best combination
    best_result = max(results, key=lambda x: x['accuracy'])
    
    print(f"\n" + "=" * 40)
    print("Best Parameters:")
    print(f"  Parameters: {best_result['params']}")
    print(f"  Accuracy: {best_result['accuracy']:.4f}")
    print(f"  Training time: {best_result['training_time']:.2f}s")
    print("=" * 40)
    
    return results


def main():
    """Main function to run all XGBoost demonstrations."""
    # Create demo instance and run all demonstrations
    demo = XGBoostDemo()
    demo.run_all_demos()
    
    # Run parameter comparison
    compare_xgboost_parameters()


if __name__ == "__main__":
    main()
