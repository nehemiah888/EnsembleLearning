#!/usr/bin/env python3
"""
LightGBM Implementation and Demonstration
LightGBM 应用

This script demonstrates the use of LightGBM for binary classification
using the breast cancer dataset. Adapted for Python 3.12 compatibility.
"""

import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class LightGBMDemo:
    """
    A class to demonstrate LightGBM classification with breast cancer dataset.
    """
    
    def __init__(self, random_state=0):
        """Initialize the LightGBM demo with random state for reproducibility."""
        self.random_state = random_state
        self.X = None
        self.y = None
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.lgb_train = None
        self.lgb_eval = None
        self.model = None
        self.results = {}
        
    def load_data(self):
        """Load the breast cancer dataset."""
        print("=== Loading Breast Cancer Dataset ===")
        
        # Load the dataset
        breast = load_breast_cancer()
        self.X = breast.data
        self.y = breast.target
        self.feature_names = breast.feature_names
        
        print(f"Dataset shape: {self.X.shape}")
        print(f"Number of features: {len(self.feature_names)}")
        print(f"Number of samples: {len(self.X)}")
        print(f"Class distribution: {np.bincount(self.y)}")
        print(f"Feature names: {list(self.feature_names[:5])}...") # Show first 5 features
        
        return self.X, self.y
        
    def split_data(self, test_size=0.2):
        """Split the dataset into training and testing sets."""
        print(f"\n=== Splitting Dataset (test_size={test_size}) ===")
        
        if self.X is None or self.y is None:
            print("No data loaded. Please load data first.")
            return
            
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=self.random_state
        )
        
        print(f"Training set size: {self.X_train.shape}")
        print(f"Test set size: {self.X_test.shape}")
        print(f"Training class distribution: {np.bincount(self.y_train)}")
        print(f"Test class distribution: {np.bincount(self.y_test)}")
        
    def prepare_lgb_datasets(self):
        """Convert data to LightGBM dataset format."""
        print("\n=== Preparing LightGBM Datasets ===")
        
        if self.X_train is None:
            print("No training data available. Please split data first.")
            return
            
        # Create LightGBM datasets with feature names
        self.lgb_train = lgb.Dataset(self.X_train, self.y_train, feature_name=list(self.feature_names))
        self.lgb_eval = lgb.Dataset(self.X_test, self.y_test, reference=self.lgb_train)
        
        print("LightGBM datasets created successfully!")
        
    def set_parameters(self):
        """Set LightGBM training parameters."""
        print("\n=== Setting LightGBM Parameters ===")
        
        # Updated parameters for Python 3.12 and modern LightGBM
        self.params = {
            'boosting_type': 'gbdt',  # Gradient boosting decision tree
            'objective': 'binary',    # Binary classification (updated from 'regression')
            'metric': ['binary_logloss', 'auc'],  # Updated metrics
            'num_leaves': 31,         # Number of leaves in one tree
            'learning_rate': 0.05,    # Learning rate
            'feature_fraction': 0.9,  # Feature sampling ratio
            'bagging_fraction': 0.8,  # Data sampling ratio
            'bagging_freq': 5,        # Frequency for bagging
            'verbose': 0,             # Updated: 0 for no output, 1 for info
            'random_state': self.random_state
        }
        
        print("Parameters set:")
        for key, value in self.params.items():
            print(f"  {key}: {value}")
            
        return self.params
        
    def train_model(self, boost_round=50, early_stop_rounds=10):
        """Train the LightGBM model with early stopping."""
        print(f"\n=== Training LightGBM Model ===")
        print(f"Boost rounds: {boost_round}")
        print(f"Early stopping rounds: {early_stop_rounds}")
        
        if self.lgb_train is None or self.lgb_eval is None:
            print("LightGBM datasets not prepared. Please prepare datasets first.")
            return
            
        if not hasattr(self, 'params'):
            self.set_parameters()
            
        # Train the model with validation and early stopping
        self.results = {}
        
        print("Training started...")
        start_time = datetime.datetime.now()
        
        # Updated for LightGBM 4.x API
        callbacks = [
            lgb.log_evaluation(period=10),
            lgb.early_stopping(stopping_rounds=early_stop_rounds)
        ]
        
        self.model = lgb.train(
            self.params,
            self.lgb_train,
            num_boost_round=boost_round,
            valid_sets=[self.lgb_eval, self.lgb_train],
            valid_names=['validate', 'train'],
            callbacks=callbacks
        )
        
        end_time = datetime.datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        print(f"Training completed in {training_time:.2f} seconds!")
        print(f"Best iteration: {self.model.best_iteration}")
        print(f"Best score: {self.model.best_score}")
        
        return self.model
        
    def make_predictions(self):
        """Make predictions on the test set."""
        print("\n=== Making Predictions ===")
        
        if self.model is None:
            print("No trained model available. Please train the model first.")
            return
            
        # Make predictions
        y_pred_proba = self.model.predict(self.X_test, num_iteration=self.model.best_iteration)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        auc_score = roc_auc_score(self.y_test, y_pred_proba)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test AUC Score: {auc_score:.4f}")
        
        print(f"\nDetailed Classification Report:")
        print(classification_report(self.y_test, y_pred))
        
        return y_pred, y_pred_proba
        
    def plot_training_history(self):
        """Plot training and validation metrics."""
        print("\n=== Plotting Training History ===")
        
        if self.model is None:
            print("No trained model available. Please train the model first.")
            return
            
        # Plot training history using LightGBM's built-in function
        try:
            # Close any existing figures first
            plt.close('all')
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot training history
            lgb.plot_metric(self.model, metric='auc', ax=ax1)
            ax1.set_title('AUC Score During Training', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Iterations', fontsize=10)
            ax1.set_ylabel('AUC Score', fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            lgb.plot_metric(self.model, metric='binary_logloss', ax=ax2)
            ax2.set_title('Binary Log Loss During Training', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Iterations', fontsize=10)
            ax2.set_ylabel('Log Loss', fontsize=10)
            ax2.grid(True, alpha=0.3)
            
            plt.suptitle('LightGBM Training History', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save the plot
            plt.savefig('lightgbm_training_history.png', dpi=150, bbox_inches='tight')
            print("Training history plot saved as: lightgbm_training_history.png")
            
            plt.show()
            
        except Exception as e:
            print(f"Could not plot training history: {e}")
            print("This might be due to missing training history data.")
            
    def plot_feature_importance(self, importance_type='split', max_features=20):
        """Plot feature importance."""
        print(f"\n=== Plotting Feature Importance ({importance_type}) ===")
        
        if self.model is None:
            print("No trained model available. Please train the model first.")
            return
            
        try:
            # Close any existing figures first to avoid blank Figure 1
            plt.close('all')
            
            # Let lgb.plot_importance create its own figure (this avoids the blank figure issue)
            ax = lgb.plot_importance(
                self.model, 
                importance_type=importance_type,
                max_num_features=max_features,
                figsize=(12, 8),
                height=0.8
            )
            
            # Customize the current figure
            fig = plt.gcf()  # Get current figure
            plt.title(f'LightGBM Feature Importance ({importance_type.title()})', 
                     fontsize=14, fontweight='bold')
            plt.xlabel(f'Feature Importance ({importance_type.title()})', fontsize=12)
            plt.ylabel('Features', fontsize=12)
            plt.tight_layout()
            
            # Save the plot
            safe_filename = f'lightgbm_feature_importance_{importance_type}.png'
            plt.savefig(safe_filename, dpi=150, bbox_inches='tight')
            print(f"Feature importance plot saved as: {safe_filename}")
            
            plt.show()
            
        except Exception as e:
            print(f"Could not plot feature importance: {e}")
            
            # Alternative plotting method
            try:
                feature_importance = self.model.feature_importance(importance_type=importance_type)
                feature_names = self.model.feature_name()
                
                # Sort features by importance
                indices = np.argsort(feature_importance)[::-1][:max_features]
                
                plt.close('all')  # Close any existing figures
                plt.figure(figsize=(12, 8))
                bars = plt.barh(range(len(indices)), feature_importance[indices], 
                               color='lightblue', alpha=0.7)
                plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
                plt.xlabel(f'Feature Importance ({importance_type.title()})', fontsize=12)
                plt.ylabel('Features', fontsize=12)
                plt.title(f'Top {max_features} Most Important Features ({importance_type.title()})',
                         fontsize=14, fontweight='bold')
                plt.gca().invert_yaxis()
                
                # Add value labels on bars
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    plt.text(width + max(feature_importance[indices]) * 0.01, bar.get_y() + bar.get_height()/2.,
                           f'{width:.0f}', ha='left', va='center', fontsize=10)
                
                plt.tight_layout()
                
                # Save the plot
                safe_filename = f'lightgbm_feature_importance_{importance_type}_alternative.png'
                plt.savefig(safe_filename, dpi=150, bbox_inches='tight')
                print(f"Feature importance plot saved as: {safe_filename}")
                
                plt.show()
                
            except Exception as e2:
                print(f"Alternative plotting method also failed: {e2}")
                # Fallback: print importance values
                try:
                    feature_importance = self.model.feature_importance(importance_type=importance_type)
                    feature_names = self.model.feature_name()
                    
                    print(f"\nTop {max_features} Feature Importance Values ({importance_type}):")
                    indices = np.argsort(feature_importance)[::-1][:max_features]
                    for i, idx in enumerate(indices):
                        print(f"  {i+1:2d}. {feature_names[idx]}: {feature_importance[idx]}")
                        
                except Exception as e3:
                    print(f"Could not get feature importance data: {e3}")
                
    def analyze_predictions(self):
        """Analyze model predictions in detail."""
        print("\n=== Detailed Prediction Analysis ===")
        
        if self.model is None:
            print("No trained model available. Please train the model first.")
            return
            
        y_pred, y_pred_proba = self.make_predictions()
        
        # Confusion matrix analysis
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        print(f"Confusion Matrix:")
        print(f"  True Negative: {cm[0,0]}")
        print(f"  False Positive: {cm[0,1]}")
        print(f"  False Negative: {cm[1,0]}")
        print(f"  True Positive: {cm[1,1]}")
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title('LightGBM Confusion Matrix', fontsize=14, fontweight='bold')
        plt.colorbar()
        
        classes = ['Malignant', 'Benign']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        
        # Add text annotations
        for i in range(len(classes)):
            for j in range(len(classes)):
                plt.text(j, i, cm[i, j], ha="center", va="center", 
                        fontsize=12, fontweight='bold',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')
                
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('lightgbm_confusion_matrix.png', dpi=150, bbox_inches='tight')
        print("Confusion matrix plot saved as: lightgbm_confusion_matrix.png")
        
        plt.show()
        
        return y_pred, y_pred_proba
        
    def run_complete_demo(self):
        """Run the complete LightGBM demonstration."""
        print("Starting LightGBM Classification Demonstration")
        print("=" * 60)
        
        # Load and prepare data
        self.load_data()
        self.split_data()
        self.prepare_lgb_datasets()
        
        # Set parameters and train model
        self.set_parameters()
        self.train_model()
        
        # Make predictions and analyze results
        y_pred, y_pred_proba = self.make_predictions()
        
        # Visualizations
        self.plot_training_history()
        self.plot_feature_importance('split')
        self.plot_feature_importance('gain')
        self.analyze_predictions()
        
        print("\n" + "=" * 60)
        print("LightGBM Demonstration Complete!")
        print("=" * 60)
        
        return self.model


def compare_different_parameters():
    """Compare LightGBM performance with different parameters."""
    print("\n" + "=" * 60)
    print("Comparing Different LightGBM Parameters")
    print("=" * 60)
    
    # Parameter combinations to test
    param_combinations = [
        {'num_leaves': 15, 'learning_rate': 0.1, 'feature_fraction': 0.8},
        {'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 0.9},
        {'num_leaves': 63, 'learning_rate': 0.03, 'feature_fraction': 0.7},
        {'num_leaves': 127, 'learning_rate': 0.01, 'feature_fraction': 0.9},
    ]
    
    results = []
    
    for i, params in enumerate(param_combinations):
        print(f"\nTesting combination {i+1}: {params}")
        
        demo = LightGBMDemo(random_state=42)
        demo.load_data()
        demo.split_data()
        demo.prepare_lgb_datasets()
        
        # Update parameters
        demo.set_parameters()
        demo.params.update(params)
        
        demo.train_model(boost_round=100, early_stop_rounds=10)
        y_pred, y_pred_proba = demo.make_predictions()
        
        accuracy = accuracy_score(demo.y_test, y_pred)
        auc_score = roc_auc_score(demo.y_test, y_pred_proba)
        
        results.append({
            'params': params,
            'accuracy': accuracy,
            'auc': auc_score
        })
    
    # Find best parameters
    best_result = max(results, key=lambda x: x['auc'])
    
    print(f"\n" + "=" * 50)
    print("Parameter Comparison Results:")
    for i, result in enumerate(results):
        print(f"Combination {i+1}: {result['params']}")
        print(f"  -> Accuracy: {result['accuracy']:.4f}, AUC: {result['auc']:.4f}")
    
    print(f"\nBest Parameters: {best_result['params']}")
    print(f"Best AUC: {best_result['auc']:.4f}")
    print("=" * 50)
    
    return results


def main():
    """Main function to run LightGBM demonstration."""
    # Run basic demonstration
    demo = LightGBMDemo()
    demo.run_complete_demo()
    
    # Run parameter comparison
    compare_different_parameters()


if __name__ == "__main__":
    main()
