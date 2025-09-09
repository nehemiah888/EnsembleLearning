#!/usr/bin/env python3
"""
AdaBoost Implementation and Demonstration
AdaBoost 实践

This script demonstrates the use of AdaBoost classifier for binary classification
using synthetic Gaussian data. Adapted for Python 3.12 compatibility.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
from sklearn.metrics import accuracy_score, classification_report
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class AdaBoostDemo:
    """
    A class to demonstrate AdaBoost classification with synthetic data.
    """
    
    def __init__(self, random_state=1):
        """Initialize the AdaBoost demo with random state for reproducibility."""
        self.random_state = random_state
        self.X = None
        self.y = None
        self.clf = None
        
    def generate_data(self):
        """Generate synthetic Gaussian data for classification."""
        print("=== Generating Synthetic Data ===")
        
        # Generate first dataset: 2D normal distribution with 200 samples
        x1, y1 = make_gaussian_quantiles(
            cov=2., 
            n_samples=200, 
            n_features=2, 
            n_classes=2, 
            shuffle=True, 
            random_state=self.random_state
        )
        
        # Generate second dataset with different mean to increase complexity
        x2, y2 = make_gaussian_quantiles(
            mean=(3, 3), 
            cov=1.5, 
            n_samples=300, 
            n_features=2, 
            n_classes=2, 
            shuffle=True, 
            random_state=self.random_state
        )
        
        # Combine datasets
        self.X = np.vstack((x1, x2))
        self.y = np.hstack((y1, 1 - y2))  # Invert labels for second dataset
        
        print(f"Generated data shape: {self.X.shape}")
        print(f"Number of samples: {len(self.X)}")
        print(f"Class distribution: {np.bincount(self.y)}")
        
        return self.X, self.y
        
    def plot_data(self):
        """Plot the generated data points."""
        print("\n=== Plotting Generated Data ===")
        
        if self.X is None or self.y is None:
            print("No data to plot. Please generate data first.")
            return
            
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Class')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Generated Synthetic Data for AdaBoost Classification')
        plt.grid(True, alpha=0.3)
        plt.show()
        
    def train_adaboost(self, n_estimators=300, learning_rate=0.8, max_depth=2):
        """Train AdaBoost classifier."""
        print(f"\n=== Training AdaBoost Classifier ===")
        print(f"Parameters:")
        print(f"  - n_estimators: {n_estimators}")
        print(f"  - learning_rate: {learning_rate}")
        print(f"  - max_depth (weak classifier): {max_depth}")
        
        if self.X is None or self.y is None:
            print("No data available. Please generate data first.")
            return
            
        # Set up weak classifier (CART decision tree)
        weak_classifier = DecisionTreeClassifier(max_depth=max_depth, random_state=self.random_state)
        
        # Create and train AdaBoost classifier
        self.clf = AdaBoostClassifier(
            estimator=weak_classifier,  # Updated from deprecated base_estimator
            algorithm='SAMME',
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=self.random_state
        )
        
        self.clf.fit(self.X, self.y)
        
        # Calculate training accuracy
        y_pred_train = self.clf.predict(self.X)
        train_accuracy = accuracy_score(self.y, y_pred_train)
        
        print(f"Training completed!")
        print(f"Training accuracy: {train_accuracy:.4f}")
        
        return self.clf
        
    def plot_decision_boundary(self, resolution=0.02):
        """Plot the decision boundary and classification results."""
        print("\n=== Plotting Decision Boundary ===")
        
        if self.clf is None:
            print("No trained classifier available. Please train the model first.")
            return
            
        # Create a mesh grid
        x1_min = self.X[:, 0].min() - 1
        x1_max = self.X[:, 0].max() + 1
        x2_min = self.X[:, 1].min() - 1
        x2_max = self.X[:, 1].max() + 1
        
        x1_mesh, x2_mesh = np.meshgrid(
            np.arange(x1_min, x1_max, resolution),
            np.arange(x2_min, x2_max, resolution)
        )
        
        # Make predictions on the mesh grid
        mesh_points = np.c_[x1_mesh.ravel(), x2_mesh.ravel()]
        y_pred_mesh = self.clf.predict(mesh_points)
        y_pred_mesh = y_pred_mesh.reshape(x1_mesh.shape)
        
        # Plot the results
        plt.figure(figsize=(12, 8))
        
        # Plot decision boundary
        plt.contourf(x1_mesh, x2_mesh, y_pred_mesh, alpha=0.8, cmap='viridis')
        
        # Plot data points
        scatter = plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, 
                            cmap='viridis', edgecolors='black', linewidth=0.5)
        plt.colorbar(scatter, label='True Class')
        
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('AdaBoost Classification Results with Decision Boundary')
        plt.grid(True, alpha=0.3)
        plt.show()
        
    def evaluate_model(self):
        """Evaluate the trained model and show detailed metrics."""
        print("\n=== Model Evaluation ===")
        
        if self.clf is None:
            print("No trained classifier available. Please train the model first.")
            return
            
        # Make predictions
        y_pred = self.clf.predict(self.X)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y, y_pred)
        
        print(f"Model Performance:")
        print(f"  - Accuracy: {accuracy:.4f}")
        print(f"  - Number of estimators used: {self.clf.n_estimators}")
        print(f"  - Feature importances: {self.clf.feature_importances_}")
        
        print(f"\nDetailed Classification Report:")
        print(classification_report(self.y, y_pred))
        
        return accuracy
        
    def plot_feature_importance(self):
        """Plot feature importance if available."""
        print("\n=== Feature Importance Analysis ===")
        
        if self.clf is None:
            print("No trained classifier available. Please train the model first.")
            return
            
        feature_names = [f'Feature {i+1}' for i in range(self.X.shape[1])]
        importances = self.clf.feature_importances_
        
        plt.figure(figsize=(10, 6))
        plt.bar(feature_names, importances, color='skyblue', alpha=0.7)
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title('Feature Importance in AdaBoost Classifier')
        plt.grid(True, alpha=0.3)
        
        for i, importance in enumerate(importances):
            plt.text(i, importance + 0.01, f'{importance:.3f}', 
                    ha='center', va='bottom')
        
        plt.show()
        
    def run_complete_demo(self):
        """Run the complete AdaBoost demonstration."""
        print("Starting AdaBoost Classification Demonstration")
        print("=" * 60)
        
        # Generate data
        self.generate_data()
        
        # Plot original data
        self.plot_data()
        
        # Train AdaBoost
        self.train_adaboost()
        
        # Plot decision boundary
        self.plot_decision_boundary()
        
        # Evaluate model
        accuracy = self.evaluate_model()
        
        # Plot feature importance
        self.plot_feature_importance()
        
        print("\n" + "=" * 60)
        print("AdaBoost Demonstration Complete!")
        print(f"Final Accuracy: {accuracy:.4f}")
        print("=" * 60)
        
        return accuracy


def compare_different_parameters():
    """Compare AdaBoost performance with different parameters."""
    print("\n" + "=" * 60)
    print("Comparing Different AdaBoost Parameters")
    print("=" * 60)
    
    # Parameter combinations to test
    param_combinations = [
        {'n_estimators': 100, 'learning_rate': 0.5, 'max_depth': 1},
        {'n_estimators': 200, 'learning_rate': 0.8, 'max_depth': 2},
        {'n_estimators': 300, 'learning_rate': 1.0, 'max_depth': 3},
        {'n_estimators': 500, 'learning_rate': 0.3, 'max_depth': 2},
    ]
    
    results = []
    
    for i, params in enumerate(param_combinations):
        print(f"\nTesting combination {i+1}: {params}")
        
        demo = AdaBoostDemo(random_state=42)
        demo.generate_data()
        demo.train_adaboost(**params)
        accuracy = demo.evaluate_model()
        
        results.append({
            'params': params,
            'accuracy': accuracy
        })
    
    # Find best parameters
    best_result = max(results, key=lambda x: x['accuracy'])
    
    print(f"\n" + "=" * 40)
    print("Parameter Comparison Results:")
    for i, result in enumerate(results):
        print(f"Combination {i+1}: {result['params']} -> Accuracy: {result['accuracy']:.4f}")
    
    print(f"\nBest Parameters: {best_result['params']}")
    print(f"Best Accuracy: {best_result['accuracy']:.4f}")
    print("=" * 40)
    
    return results


def main():
    """Main function to run AdaBoost demonstration."""
    # Run basic demonstration
    demo = AdaBoostDemo()
    demo.run_complete_demo()
    
    # Run parameter comparison
    compare_different_parameters()


if __name__ == "__main__":
    main()
