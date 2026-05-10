#!/usr/bin/env python3
"""
Hybrid FFE + ML Non-Linear Equalizer for PCIe 7.0
==================================================

Combines traditional FFE for linear ISI with ML for non-linear distortion.

Architecture:
    RX Signal → FFE (Linear ISI) → ML (Non-linear) → Symbol Decision

The ML component learns to compensate for:
- Non-linear channel effects
- Crosstalk
- Pattern-dependent noise
- Residual ISI after FFE

Author: Anish Anand
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import pickle

np.random.seed(1)

class HybridMLEqualizer:
    """
    Hybrid FFE + ML equalizer.
    
    Uses FFE for linear ISI removal, then applies ML to handle:
    - Non-linear distortions
    - Pattern-dependent effects
    - Residual ISI
    """
    
    def __init__(self,
                 context_length: int = 5,
                 hidden_sizes: List[int] = [32, 64, 32],
                 learning_rate: float = 0.001,
                 pam_levels: int = 6,
                 pre_context: int = None,
                 post_context: int = None):
        """
        Initialize hybrid ML equalizer.
        
        Args:
            context_length: Number of neighboring samples to use (each side, symmetric)
            hidden_sizes: Neural network hidden layer sizes
            learning_rate: Learning rate for training
            pam_levels: Number of PAM levels (6 for PAM6)
            pre_context: Samples before cursor (overrides context_length if set)
            post_context: Samples after cursor (overrides context_length if set)
        """
        self.context_length = context_length
        self.pre_context = pre_context if pre_context is not None else context_length
        self.post_context = post_context if post_context is not None else context_length
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.pam_levels = pam_levels
        
        # Network parameters
        self.weights = []
        self.biases = []
        self.is_trained = False
        
        # Normalization
        self.input_mean = None
        self.input_std = None
        
        # PAM6 decision levels: (-5, -3, -1, 1, 3, 5)
        self.decision_levels = np.array([-5, -3, -1, 1, 3, 5])
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation."""
        return np.maximum(0, x)
    
    def _relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """ReLU derivative."""
        return (x > 0).astype(float)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation for classification."""
        # Numerical stability
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _initialize_network(self):
        """Initialize network weights."""
        input_size = self.pre_context + 1 + self.post_context
        layer_sizes = [input_size] + self.hidden_sizes + [self.pam_levels]
        
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros(layer_sizes[i+1])
            self.weights.append(w)
            self.biases.append(b)
    
    def _extract_context(self, signal: np.ndarray, index: int) -> np.ndarray:
        """
        Extract context window around a sample.
        
        Args:
            signal: Input signal
            index: Center index (cursor position)
        
        Returns:
            context: Context window (pre_context + 1 + post_context,)
        """
        start = max(0, index - self.pre_context)
        end = min(len(signal), index + self.post_context + 1)
        
        context = signal[start:end]
        
        # Pad if necessary
        if start == 0:
            left_pad = self.pre_context - index
            context = np.pad(context, (left_pad, 0), 'edge')
        if end == len(signal):
            right_pad = (index + self.post_context + 1) - len(signal)
            context = np.pad(context, (0, right_pad), 'edge')
        
        return context
    
    def _forward(self, x: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Forward pass through network."""
        activations = [x]
        
        # Hidden layers with ReLU
        for i in range(len(self.weights) - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            a = self._relu(z)
            activations.append(a)
        
        # Output layer with softmax
        z = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        a = self._softmax(z)
        activations.append(a)
        
        return activations[-1], activations
    
    def predict(self, ffe_output: np.ndarray, samples_per_symbol: int = 32) -> np.ndarray:
        """
        Predict symbols from FFE output.
        
        Args:
            ffe_output: Signal after FFE equalization
            samples_per_symbol: Samples per symbol
        
        Returns:
            decisions: Predicted symbols
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        # Sample at symbol centers
        sample_indices = np.arange(samples_per_symbol // 2, len(ffe_output), samples_per_symbol)
        n_symbols = len(sample_indices)
        
        decisions = np.zeros(n_symbols, dtype=int)
        
        for i, idx in enumerate(sample_indices):
            # Extract context
            context = self._extract_context(ffe_output, idx)
            
            # Normalize
            context_norm = (context - self.input_mean) / (self.input_std + 1e-8)
            
            # Predict probabilities
            probs, _ = self._forward(context_norm.reshape(1, -1))
            
            # Choose most likely symbol
            decisions[i] = np.argmax(probs[0])
        
        return decisions
    
    def train(self,
              ffe_output: np.ndarray,
              true_symbols: np.ndarray,
              samples_per_symbol: int = 32,
              epochs: int = 50,
              batch_size: int = 64,
              validation_split: float = 0.2,
              verbose: bool = True,
              snapshot_epochs: list = None):
        """
        Train the ML equalizer.
        
        Args:
            ffe_output: Signal after FFE equalization
            true_symbols: Ground truth symbols (0, 1, 2, 3, 4, 5 for PAM6)
            samples_per_symbol: Samples per symbol
            epochs: Training epochs
            batch_size: Batch size
            validation_split: Validation fraction
            verbose: Print progress
            snapshot_epochs: List of epoch numbers (0-indexed) at which to
                            snapshot weights/biases. Epoch 0 means after
                            initialization (before any training).
                            Snapshots are stored in history['weight_snapshots'].
        """
        # Extract training samples
        sample_indices = np.arange(samples_per_symbol // 2, len(ffe_output), samples_per_symbol)
        n_samples = min(len(sample_indices), len(true_symbols))
        
        X = []
        y = []
        
        for i in range(n_samples):
            idx = sample_indices[i]
            context = self._extract_context(ffe_output, idx)
            X.append(context)
            y.append(true_symbols[i])
        
        X = np.array(X)
        y = np.array(y, dtype=int)
        
        # Normalize inputs
        self.input_mean = np.mean(X, axis=0)
        self.input_std = np.std(X, axis=0)
        X = (X - self.input_mean) / (self.input_std + 1e-8)
        
        # One-hot encode targets
        y_onehot = np.zeros((len(y), self.pam_levels))
        y_onehot[np.arange(len(y)), y] = 1
        
        # Train/validation split
        n_val = int(len(X) * validation_split)
        indices = np.random.permutation(len(X))
        
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]
        
        X_train, y_train = X[train_idx], y_onehot[train_idx]
        X_val, y_val = X[val_idx], y_onehot[val_idx]
        y_val_labels = y[val_idx]
        
        # Initialize network
        self._initialize_network()
        
        # Snapshot tracking
        if snapshot_epochs is None:
            snapshot_epochs_set = set()
        else:
            snapshot_epochs_set = set(snapshot_epochs)
        weight_snapshots = {}  # epoch -> {'weights': [...], 'biases': [...]}
        
        # Capture epoch-0 snapshot (weights right after initialization, before training)
        if 0 in snapshot_epochs_set:
            val_output_init, _ = self._forward(X_val)
            val_loss_init = -np.mean(np.sum(y_val * np.log(val_output_init + 1e-10), axis=1))
            val_pred_init = np.argmax(val_output_init, axis=1)
            val_acc_init = float(np.mean(val_pred_init == y_val_labels))
            weight_snapshots[0] = {
                'weights': [w.copy() for w in self.weights],
                'biases': [b.copy() for b in self.biases],
                'val_acc': val_acc_init,
                'val_loss': float(val_loss_init),
            }
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        n_train = len(X_train)
        n_batches = (n_train + batch_size - 1) // batch_size
        
        for epoch in range(epochs):
            # Shuffle
            shuffle_idx = np.random.permutation(n_train)
            X_train_shuffled = X_train[shuffle_idx]
            y_train_shuffled = y_train[shuffle_idx]
            
            epoch_loss = 0.0
            epoch_correct = 0
            
            # Mini-batch training
            for batch in range(n_batches):
                start = batch * batch_size
                end = min(start + batch_size, n_train)
                
                X_batch = X_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end]
                
                # Forward pass
                output, activations = self._forward(X_batch)
                
                # Cross-entropy loss
                loss = -np.mean(np.sum(y_batch * np.log(output + 1e-10), axis=1))
                epoch_loss += loss
                
                # Accuracy
                predictions = np.argmax(output, axis=1)
                true_labels = np.argmax(y_batch, axis=1)
                epoch_correct += np.sum(predictions == true_labels)
                
                # Backward pass
                self._backward(X_batch, y_batch, activations)
            
            # Training metrics
            train_loss = epoch_loss / n_batches
            train_acc = epoch_correct / n_train
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # Validation metrics
            val_output, _ = self._forward(X_val)
            val_loss = -np.mean(np.sum(y_val * np.log(val_output + 1e-10), axis=1))
            val_predictions = np.argmax(val_output, axis=1)
            val_acc = np.mean(val_predictions == y_val_labels)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Loss: {train_loss:.4f}, Acc: {train_acc:.4f} - "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Snapshot weights at requested epochs (1-indexed epoch number)
            # epoch variable is 0-indexed, so epoch=0 means after 1st training epoch
            # We use (epoch+1) to match the user-facing epoch number:
            #   snapshot_epoch=1 -> after 1st epoch of training
            #   snapshot_epoch=8 -> after 8th epoch of training
            if (epoch + 1) in snapshot_epochs_set:
                weight_snapshots[epoch + 1] = {
                    'weights': [w.copy() for w in self.weights],
                    'biases': [b.copy() for b in self.biases],
                    'val_acc': float(val_acc),
                    'val_loss': float(val_loss),
                }
        
        self.is_trained = True
        
        # Store snapshots in history
        history['weight_snapshots'] = weight_snapshots
        
        if verbose:
            print(f"\nTraining completed!")
            print(f"Final Validation Accuracy: {history['val_acc'][-1]:.4f}")
        
        return history
    
    def _backward(self, X_batch: np.ndarray, y_batch: np.ndarray,
                  activations: List[np.ndarray]):
        """Backpropagation."""
        batch_size = len(X_batch)
        
        # Output layer gradient (cross-entropy + softmax)
        delta = (activations[-1] - y_batch) / batch_size
        
        # Backpropagate
        deltas = [delta]
        
        for i in range(len(self.weights) - 1, 0, -1):
            delta = np.dot(delta, self.weights[i].T) * self._relu_derivative(activations[i])
            deltas.insert(0, delta)
        
        # Update weights
        for i in range(len(self.weights)):
            dW = np.dot(activations[i].T, deltas[i])
            db = np.sum(deltas[i], axis=0)
            
            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db
    
    def save(self, filename: str):
        """Save model."""
        model_data = {
            'context_length': self.context_length,
            'hidden_sizes': self.hidden_sizes,
            'pam_levels': self.pam_levels,
            'weights': self.weights,
            'biases': self.biases,
            'input_mean': self.input_mean,
            'input_std': self.input_std,
            'is_trained': self.is_trained
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filename}")
    
    @classmethod
    def load(cls, filename: str) -> 'HybridMLEqualizer':
        """Load model."""
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(
            context_length=model_data['context_length'],
            hidden_sizes=model_data['hidden_sizes'],
            pam_levels=model_data['pam_levels']
        )
        
        model.weights = model_data['weights']
        model.biases = model_data['biases']
        model.input_mean = model_data['input_mean']
        model.input_std = model_data['input_std']
        model.is_trained = model_data['is_trained']
        
        print(f"Model loaded from {filename}")
        return model


if __name__ == "__main__":
    print("Hybrid FFE + ML Equalizer")
    print("=" * 70)
    print("\nCombines linear FFE with ML for non-linear compensation.")
    print("\nSee demo_hybrid_ml_eq.py for usage example.")
