#!/usr/bin/env python3
"""
Joint FFE + ML Equalizer: End-to-End Training (v2 — Fixed + Improved)
======================================================================

Trains FFE (linear FIR) taps and neural network weights simultaneously
via backpropagation through the entire chain:

    ADC samples → FFE (linear FIR) → Context extraction → NN → Softmax → Loss

Key fixes over v1:
  - FFE uses CAUSAL convention: y[n] = sum_k taps[k] * signal[n-k]
    This matches AdaptiveFFE exactly, so LMS taps transfer directly
    and alignment offsets are consistent.
  - cursor_index only controls initialization and normalization,
    not signal indexing.

Performance improvements:
  - Adam optimizer (adaptive per-parameter LR, momentum)
  - Gradient clipping for FFE taps (prevents large steps)
  - Warm-start option: freeze FFE for first N epochs, then unfreeze
  - Learning rate scheduling (cosine annealing)
  - Vectorized NumPy throughout (np.convolve, fancy indexing, np.add.at)

Author: Seema / Claude
Date: February 2026
"""

import numpy as np
from typing import Tuple, List, Optional, Dict


class AdamState:
    """Adam optimizer state for a single parameter array."""
    
    def __init__(self, shape, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = np.zeros(shape)  # first moment
        self.v = np.zeros(shape)  # second moment
        self.t = 0                # timestep
    
    def step(self, param: np.ndarray, grad: np.ndarray, lr_override: float = None) -> np.ndarray:
        """Update parameter using Adam. Returns updated parameter."""
        self.t += 1
        lr = lr_override if lr_override is not None else self.lr
        
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        param -= lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return param


class JointFFEMLEqualizer:
    """
    Joint FFE + ML Equalizer with end-to-end backpropagation.
    
    FFE convention (CAUSAL — matches AdaptiveFFE):
        y[n] = sum_k( w[k] * x[n-k] )  for k = 0..n_taps-1
    
    This means:
        - taps[0] * x[n]       (current sample)
        - taps[1] * x[n-1]     (1 sample ago)
        - ...
        - taps[cursor] * x[n-cursor]  (cursor position — "main" tap)
        - ...
        - taps[K-1] * x[n-K+1] (K-1 samples ago)
    
    The cursor_index only controls:
        - Which tap is initialized to 1.0 (pass-through)
        - Which tap is normalized to 1.0 (gain control)
    
    It does NOT shift the signal, so:
        - LMS taps from AdaptiveFFE transfer directly
        - Alignment offsets from AdaptiveFFE are consistent
        - y[n] at output position n corresponds to symbol at position n
          (with the same group delay as AdaptiveFFE)
    """
    
    def __init__(self,
                 # FFE parameters
                 n_ffe_taps: int = 8,
                 cursor_index: int = 3,
                 ffe_learning_rate: float = 0.0005,
                 ffe_normalize_interval: int = 50,
                 ffe_grad_clip: float = 1.0,
                 # NN parameters  
                 context_length: int = 11,
                 hidden_sizes: List[int] = None,
                 nn_learning_rate: float = 0.005,
                 pam_levels: int = 4,
                 # Optimizer
                 use_adam: bool = True,
                 # L2 regularization toward reference taps
                 ffe_l2_lambda: float = 0.0,
                 ffe_reference_taps: np.ndarray = None,
                 # Asymmetric context
                 pre_context: int = None,
                 post_context: int = None):
        """
        Initialize joint FFE+ML equalizer.
        
        Args:
            n_ffe_taps: Number of FFE taps
            cursor_index: Index of cursor (main) tap — for init/normalization only
            ffe_learning_rate: Learning rate for FFE taps
            ffe_normalize_interval: Normalize FFE taps every N batches (0=disable)
            ffe_grad_clip: Max L2 norm for FFE gradient (0=disable)
            context_length: Number of neighboring samples for NN context (each side)
            hidden_sizes: NN hidden layer sizes
            nn_learning_rate: Learning rate for NN weights
            pam_levels: Number of PAM levels (4, 6, or 8)
            use_adam: Use Adam optimizer (True) or plain SGD (False)
            ffe_l2_lambda: L2 regularization strength toward reference taps.
                           Adds lambda * (taps - reference_taps) to FFE gradient.
                           Acts as elastic net that keeps taps close to LMS solution.
            ffe_reference_taps: Reference taps for L2 regularization (typically LMS taps).
                                If None and l2_lambda > 0, uses initial taps as reference.
            pre_context: Samples before cursor (overrides context_length if set)
            post_context: Samples after cursor (overrides context_length if set)
        """
        # FFE parameters
        self.n_ffe_taps = n_ffe_taps
        self.cursor_index = cursor_index
        self.ffe_lr = ffe_learning_rate
        self.ffe_normalize_interval = ffe_normalize_interval
        self.ffe_grad_clip = ffe_grad_clip
        self.ffe_l2_lambda = ffe_l2_lambda
        self.ffe_reference_taps = np.array(ffe_reference_taps, dtype=float) if ffe_reference_taps is not None else None
        
        # Initialize FFE taps: cursor=1, rest=0
        self.ffe_taps = np.zeros(n_ffe_taps)
        self.ffe_taps[cursor_index] = 1.0
        
        # NN parameters
        self.context_length = context_length
        self.pre_context = pre_context if pre_context is not None else context_length
        self.post_context = post_context if post_context is not None else context_length
        self.hidden_sizes = hidden_sizes if hidden_sizes is not None else [4, 4]
        self.nn_lr = nn_learning_rate
        self.pam_levels = pam_levels
        self.use_adam = use_adam
        
        # NN weights (initialized in _initialize_network)
        self.nn_weights = []
        self.nn_biases = []
        
        # Adam states (initialized in _initialize_network)
        self.nn_w_adam = []
        self.nn_b_adam = []
        self.ffe_adam = None
        
        # Normalization
        self.input_mean = None
        self.input_std = None
        
        self.is_trained = False
        self.batch_count = 0
    
    def set_ffe_taps(self, taps: np.ndarray):
        """Set FFE taps (e.g., from LMS pre-training)."""
        if len(taps) != self.n_ffe_taps:
            raise ValueError(f"Expected {self.n_ffe_taps} taps, got {len(taps)}")
        self.ffe_taps = np.array(taps, dtype=float)
    
    def get_ffe_taps(self) -> np.ndarray:
        """Get current FFE taps."""
        return self.ffe_taps.copy()
    
    # ─── FFE Operations (Causal Convention) ─────────────────────────────
    
    def apply_ffe(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply FFE to symbol-rate signal using CAUSAL convention.
        
        y[n] = sum_k( ffe_taps[k] * signal[n-k] )
             = np.convolve(signal, taps)[n]   (first N elements)
        
        This matches AdaptiveFFE.equalize() exactly, so:
          - LMS taps transfer directly without any shift
          - Alignment offsets from _align_sequences are consistent
        
        Args:
            signal: Input signal at symbol rate
            
        Returns:
            ffe_output: Equalized signal (same length as input)
        """
        full_conv = np.convolve(signal, self.ffe_taps, mode='full')
        return full_conv[:len(signal)]
    
    # ─── Context Extraction (Vectorized) ────────────────────────────────
    
    def _build_context_matrix(self, signal: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """
        Extract context windows for all indices using vectorized fancy indexing.
        
        Returns:
            contexts: (len(indices), pre_context + 1 + post_context) context matrix
        """
        pre = self.pre_context
        post = self.post_context
        ctx_width = pre + 1 + post
        
        padded = np.pad(signal, (pre, post), mode='edge')
        col_offsets = np.arange(ctx_width)
        idx_matrix = indices[:, np.newaxis] + col_offsets[np.newaxis, :]
        
        return padded[idx_matrix]
    
    # ─── NN Operations ──────────────────────────────────────────────────
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def _relu_deriv(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _initialize_network(self):
        """Initialize NN weights with He initialization and Adam states."""
        input_size = self.pre_context + 1 + self.post_context
        layer_sizes = [input_size] + self.hidden_sizes + [self.pam_levels]
        
        self.nn_weights = []
        self.nn_biases = []
        self.nn_w_adam = []
        self.nn_b_adam = []
        
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros(layer_sizes[i+1])
            self.nn_weights.append(w)
            self.nn_biases.append(b)
            
            if self.use_adam:
                self.nn_w_adam.append(AdamState(w.shape, lr=self.nn_lr))
                self.nn_b_adam.append(AdamState(b.shape, lr=self.nn_lr))
        
        # Adam state for FFE taps
        if self.use_adam:
            self.ffe_adam = AdamState((self.n_ffe_taps,), lr=self.ffe_lr)
    
    def _nn_forward(self, x: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Forward pass through NN. Returns (output, activations_list)."""
        activations = [x]
        
        for i in range(len(self.nn_weights) - 1):
            z = np.dot(activations[-1], self.nn_weights[i]) + self.nn_biases[i]
            a = self._relu(z)
            activations.append(a)
        
        z = np.dot(activations[-1], self.nn_weights[-1]) + self.nn_biases[-1]
        a = self._softmax(z)
        activations.append(a)
        
        return activations[-1], activations
    
    def _nn_backward(self, y_batch: np.ndarray, activations: List[np.ndarray]):
        """
        Backprop through NN.
        Returns: (weight_grads, bias_grads, dL_dinput)
        """
        batch_size = len(y_batch)
        
        delta = (activations[-1] - y_batch) / batch_size
        
        deltas = [delta]
        for i in range(len(self.nn_weights) - 1, 0, -1):
            delta = np.dot(delta, self.nn_weights[i].T) * self._relu_deriv(activations[i])
            deltas.insert(0, delta)
        
        weight_grads = []
        bias_grads = []
        for i in range(len(self.nn_weights)):
            dW = np.dot(activations[i].T, deltas[i])
            db = np.sum(deltas[i], axis=0)
            weight_grads.append(dW)
            bias_grads.append(db)
        
        dL_dinput = np.dot(deltas[0], self.nn_weights[0].T)
        
        return weight_grads, bias_grads, dL_dinput
    
    # ─── FFE Gradient (Vectorized, Causal Convention) ───────────────────
    
    def _context_grad_to_ffe_grad(self, dL_d_contexts_norm: np.ndarray,
                                   symbol_indices: np.ndarray,
                                   full_signal: np.ndarray,
                                   N_ffe: int) -> np.ndarray:
        """
        Convert gradient w.r.t. normalized contexts to gradient w.r.t. FFE taps.
        
        Chain: signal → FFE → ffe_output → contexts → normalize → NN → loss
        
        Step 1: Undo normalization: dL/d(context) = dL/d(context_norm) / (std + eps)
        Step 2: Scatter-add to get dL/d(ffe_output[n]) using np.add.at
        Step 3: dL/d(taps[k]) = sum_n dL/d(ffe_output[n]) * signal[n - k]
                (CAUSAL: signal index is n-k, no cursor offset)
        
        Args:
            dL_d_contexts_norm: (batch_size, 2*L+1) gradient from NN backprop
            symbol_indices: which symbols are in this batch
            full_signal: the raw ADC signal (before FFE)
            N_ffe: length of FFE output
            
        Returns:
            dL_d_ffe_taps: (n_taps,) gradient for FFE tap update
        """
        pre = self.pre_context
        post = self.post_context
        ctx_width = pre + 1 + post
        
        # Step 1: undo normalization
        dL_d_contexts = dL_d_contexts_norm / (self.input_std + 1e-8)
        
        # Step 2: scatter-add to dL/d(ffe_output)
        dL_d_ffe = np.zeros(N_ffe)
        
        j_offsets = np.arange(ctx_width) - pre
        all_positions = symbol_indices[:, np.newaxis] + j_offsets[np.newaxis, :]
        
        valid_mask = (all_positions >= 0) & (all_positions < N_ffe)
        flat_positions = np.clip(all_positions, 0, N_ffe - 1).ravel()
        flat_values = (dL_d_contexts * valid_mask).ravel()
        
        np.add.at(dL_d_ffe, flat_positions, flat_values)
        
        # Step 3: CAUSAL convention — signal index is (n - k)
        # dL/d(taps[k]) = sum_n dL/d(ffe_output[n]) * signal[n - k]
        dL_d_taps = np.zeros(self.n_ffe_taps)
        
        nonzero_mask = dL_d_ffe != 0
        if not np.any(nonzero_mask):
            return dL_d_taps
        
        nonzero_idx = np.nonzero(nonzero_mask)[0]
        dL_nonzero = dL_d_ffe[nonzero_idx]
        
        # Pad signal for safe indexing (handle n-k < 0)
        pad = self.n_ffe_taps
        padded_signal = np.pad(full_signal, (pad, pad), mode='constant')
        
        for k in range(self.n_ffe_taps):
            # CAUSAL: signal index = n - k (+ pad for padded array)
            sig_indices = nonzero_idx - k + pad
            dL_d_taps[k] = np.dot(dL_nonzero, padded_signal[sig_indices])
        
        return dL_d_taps
    
    def normalize_ffe_taps(self):
        """Normalize FFE taps so cursor tap = 1.0."""
        cursor_val = self.ffe_taps[self.cursor_index]
        if abs(cursor_val) > 1e-10:
            self.ffe_taps /= cursor_val
    
    def _clip_ffe_grad(self, grad: np.ndarray) -> np.ndarray:
        """Clip FFE gradient by L2 norm."""
        if self.ffe_grad_clip > 0:
            norm = np.linalg.norm(grad)
            if norm > self.ffe_grad_clip:
                grad = grad * (self.ffe_grad_clip / norm)
        return grad
    
    def _cosine_lr(self, base_lr: float, epoch: int, total_epochs: int,
                   warmup_epochs: int = 5) -> float:
        """Cosine annealing with linear warmup."""
        if epoch < warmup_epochs:
            return base_lr * (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return base_lr * 0.5 * (1 + np.cos(np.pi * progress))
    
    # ─── Training ───────────────────────────────────────────────────────
    
    def train(self,
              adc_signal: np.ndarray,
              true_symbols: np.ndarray,
              epochs: int = 80,
              batch_size: int = 64,
              validation_split: float = 0.2,
              init_ffe_taps: np.ndarray = None,
              train_ffe: bool = True,
              ffe_warmup_epochs: int = 0,
              ffe_refresh_interval: int = 1,
              lr_schedule: bool = True,
              early_stopping_patience: int = 0,
              verbose: bool = True) -> Dict:
        """
        Train the joint FFE+ML equalizer.
        
        Args:
            adc_signal: Raw ADC output signal (symbol-rate)
            true_symbols: Ground truth symbol indices (0..pam_levels-1)
            epochs: Training epochs
            batch_size: Mini-batch size (larger = more stable FFE gradients)
            validation_split: Fraction for validation
            init_ffe_taps: Initial FFE taps (None = cursor-only init)
            train_ffe: If True, jointly train FFE. If False, freeze FFE.
            ffe_warmup_epochs: Freeze FFE for first N epochs while NN stabilizes.
            ffe_refresh_interval: Recompute FFE output & contexts every N epochs.
            lr_schedule: Use cosine annealing with warmup (True) or flat LR (False)
            early_stopping_patience: Stop if val_acc doesn't improve for N epochs.
                                     0 = disabled. Restores best weights on stop.
            verbose: Print progress
            
        Returns:
            history dict with train/val loss/acc and FFE tap evolution
        """
        # Initialize FFE taps
        if init_ffe_taps is not None:
            self.set_ffe_taps(init_ffe_taps)
        
        n_samples = min(len(adc_signal), len(true_symbols))
        adc_signal = adc_signal[:n_samples]
        y_all = true_symbols[:n_samples].astype(int)
        all_indices = np.arange(n_samples)
        
        # One-hot encode targets
        y_onehot = np.zeros((n_samples, self.pam_levels))
        y_onehot[np.arange(n_samples), y_all] = 1
        
        # Train/val split
        n_val = int(n_samples * validation_split)
        perm = np.random.permutation(n_samples)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        
        y_train = y_onehot[train_idx]
        y_train_labels = y_all[train_idx]
        train_symbol_indices = train_idx.copy()
        
        y_val = y_onehot[val_idx]
        y_val_labels = y_all[val_idx]
        
        # Initialize NN + optimizer states
        self._initialize_network()
        
        # Set L2 reference taps (if not explicitly provided, use init taps)
        if self.ffe_l2_lambda > 0 and self.ffe_reference_taps is None:
            self.ffe_reference_taps = self.ffe_taps.copy()
        
        # Early stopping state
        best_val_acc = -1.0
        best_epoch = 0
        best_nn_weights = None
        best_nn_biases = None
        best_ffe_taps = None
        patience_counter = 0
        
        # Initial FFE output + context extraction
        ffe_output = self.apply_ffe(adc_signal)
        X_all_norm, X_val_norm = self._refresh_contexts(
            ffe_output, all_indices, val_idx
        )
        X_train_norm = X_all_norm[train_idx]
        
        # Training history
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'ffe_taps': [self.ffe_taps.copy()],
            'ffe_tap_grad_norm': [],
            'nn_lr_history': [],
            'ffe_lr_history': [],
        }
        
        n_train = len(train_idx)
        n_batches = (n_train + batch_size - 1) // batch_size
        self.batch_count = 0
        
        for epoch in range(epochs):
            # ── Learning rate schedule ──
            if lr_schedule:
                nn_lr_epoch = self._cosine_lr(self.nn_lr, epoch, epochs, warmup_epochs=5)
                ffe_lr_epoch = self._cosine_lr(self.ffe_lr, epoch, epochs, warmup_epochs=5)
            else:
                nn_lr_epoch = self.nn_lr
                ffe_lr_epoch = self.ffe_lr
            
            # Should FFE be trained this epoch?
            ffe_active = train_ffe and (epoch >= ffe_warmup_epochs)
            
            # Shuffle
            shuffle = np.random.permutation(n_train)
            X_train_shuffled = X_train_norm[shuffle]
            y_train_shuffled = y_train[shuffle]
            train_indices_shuffled = train_symbol_indices[shuffle]
            
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_ffe_grad_norm = 0.0
            
            for batch in range(n_batches):
                start = batch * batch_size
                end = min(start + batch_size, n_train)
                
                X_batch = X_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end]
                batch_sym_indices = train_indices_shuffled[start:end]
                
                # Forward
                output, activations = self._nn_forward(X_batch)
                
                loss = -np.mean(np.sum(y_batch * np.log(output + 1e-10), axis=1))
                epoch_loss += loss
                
                predictions = np.argmax(output, axis=1)
                true_labels = np.argmax(y_batch, axis=1)
                epoch_correct += np.sum(predictions == true_labels)
                
                # Backward
                nn_w_grads, nn_b_grads, dL_dinput = self._nn_backward(y_batch, activations)
                
                # Update NN weights
                for i in range(len(self.nn_weights)):
                    if self.use_adam:
                        self.nn_weights[i] = self.nn_w_adam[i].step(
                            self.nn_weights[i], nn_w_grads[i], lr_override=nn_lr_epoch)
                        self.nn_biases[i] = self.nn_b_adam[i].step(
                            self.nn_biases[i], nn_b_grads[i], lr_override=nn_lr_epoch)
                    else:
                        self.nn_weights[i] -= nn_lr_epoch * nn_w_grads[i]
                        self.nn_biases[i] -= nn_lr_epoch * nn_b_grads[i]
                
                # FFE gradient
                if ffe_active:
                    dL_d_taps = self._context_grad_to_ffe_grad(
                        dL_dinput, batch_sym_indices, adc_signal, len(ffe_output)
                    )
                    
                    # Add L2 regularization toward reference taps
                    if self.ffe_l2_lambda > 0 and self.ffe_reference_taps is not None:
                        l2_grad = self.ffe_l2_lambda * (self.ffe_taps - self.ffe_reference_taps)
                        dL_d_taps += l2_grad
                    
                    # Clip gradient
                    dL_d_taps = self._clip_ffe_grad(dL_d_taps)
                    epoch_ffe_grad_norm += np.linalg.norm(dL_d_taps)
                    
                    # Update FFE taps
                    if self.use_adam:
                        self.ffe_taps = self.ffe_adam.step(
                            self.ffe_taps, dL_d_taps, lr_override=ffe_lr_epoch)
                    else:
                        self.ffe_taps -= ffe_lr_epoch * dL_d_taps
                    
                    # Periodic normalization
                    self.batch_count += 1
                    if (self.ffe_normalize_interval > 0 and 
                        self.batch_count % self.ffe_normalize_interval == 0):
                        self.normalize_ffe_taps()
            
            # ── End of epoch: refresh FFE output if taps changed ──
            if ffe_active and (epoch + 1) % ffe_refresh_interval == 0:
                ffe_output = self.apply_ffe(adc_signal)
                X_all_norm, X_val_norm = self._refresh_contexts(
                    ffe_output, all_indices, val_idx
                )
                X_train_norm = X_all_norm[train_idx]
            
            # Epoch metrics
            train_loss = epoch_loss / n_batches
            train_acc = epoch_correct / n_train
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            val_output, _ = self._nn_forward(X_val_norm)
            val_loss = -np.mean(np.sum(y_val * np.log(val_output + 1e-10), axis=1))
            val_pred = np.argmax(val_output, axis=1)
            val_acc = np.mean(val_pred == y_val_labels)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(float(val_acc))
            
            history['ffe_taps'].append(self.ffe_taps.copy())
            history['ffe_tap_grad_norm'].append(
                epoch_ffe_grad_norm / n_batches if ffe_active else 0.0
            )
            history['nn_lr_history'].append(nn_lr_epoch)
            history['ffe_lr_history'].append(ffe_lr_epoch if ffe_active else 0.0)
            
            # ── Early stopping ──
            if early_stopping_patience > 0:
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch
                    patience_counter = 0
                    # Save best weights
                    best_nn_weights = [w.copy() for w in self.nn_weights]
                    best_nn_biases = [b.copy() for b in self.nn_biases]
                    best_ffe_taps = self.ffe_taps.copy()
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        if verbose:
                            print(f"  Early stopping at epoch {epoch+1} "
                                  f"(best val_acc={best_val_acc:.4f} at epoch {best_epoch+1})")
                        # Restore best weights
                        self.nn_weights = best_nn_weights
                        self.nn_biases = best_nn_biases
                        self.ffe_taps = best_ffe_taps
                        break
            
            if verbose and (epoch + 1) % 10 == 0:
                ffe_str = ""
                if ffe_active:
                    ffe_str = f"  FFE ‖∇‖={epoch_ffe_grad_norm/n_batches:.4f}"
                elif train_ffe:
                    ffe_str = f"  (FFE warming up, {ffe_warmup_epochs - epoch - 1} left)"
                print(f"  Epoch {epoch+1:3d}/{epochs} - "
                      f"Loss: {train_loss:.4f}, Acc: {train_acc:.4f} - "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}{ffe_str}")
        
        self.is_trained = True
        
        # Final FFE refresh
        if train_ffe:
            ffe_output = self.apply_ffe(adc_signal)
            X_all_norm, _ = self._refresh_contexts(
                ffe_output, all_indices, val_idx
            )
        
        if verbose:
            print(f"\n  Training complete! Final Val Acc: {history['val_acc'][-1]:.4f}")
            if train_ffe:
                print(f"  Final FFE taps: [{', '.join(f'{t:.4f}' for t in self.ffe_taps)}]")
        
        return history
    
    def _refresh_contexts(self, ffe_output: np.ndarray, 
                          all_indices: np.ndarray,
                          val_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Recompute context matrix and normalization from FFE output."""
        X_all = self._build_context_matrix(ffe_output, all_indices)
        self.input_mean = np.mean(X_all, axis=0)
        self.input_std = np.std(X_all, axis=0)
        X_all_norm = (X_all - self.input_mean) / (self.input_std + 1e-8)
        X_val_norm = X_all_norm[val_idx]
        return X_all_norm, X_val_norm
    
    # ─── Prediction ─────────────────────────────────────────────────────
    
    def predict(self, adc_signal: np.ndarray) -> np.ndarray:
        """Predict symbols from raw ADC signal (applies FFE then NN)."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        ffe_output = self.apply_ffe(adc_signal)
        return self._predict_from_signal(ffe_output)
    
    def predict_from_ffe_output(self, ffe_output: np.ndarray) -> np.ndarray:
        """Predict symbols from pre-computed FFE output."""
        if not self.is_trained:
            raise RuntimeError("Model not trained.")
        return self._predict_from_signal(ffe_output)
    
    def _predict_from_signal(self, signal: np.ndarray) -> np.ndarray:
        """Predict symbols using vectorized batch inference."""
        n_symbols = len(signal)
        indices = np.arange(n_symbols)
        
        contexts = self._build_context_matrix(signal, indices)
        contexts_norm = (contexts - self.input_mean) / (self.input_std + 1e-8)
        
        chunk_size = 5000
        decisions = np.zeros(n_symbols, dtype=int)
        
        for start in range(0, n_symbols, chunk_size):
            end = min(start + chunk_size, n_symbols)
            probs, _ = self._nn_forward(contexts_norm[start:end])
            decisions[start:end] = np.argmax(probs, axis=1)
        
        return decisions
    
    def get_summary(self) -> Dict:
        """Get model summary."""
        nn_params = sum(w.size + b.size for w, b in zip(self.nn_weights, self.nn_biases))
        return {
            'n_ffe_taps': self.n_ffe_taps,
            'cursor_index': self.cursor_index,
            'ffe_taps': self.ffe_taps.tolist(),
            'nn_hidden_sizes': self.hidden_sizes,
            'nn_total_params': nn_params,
            'context_length': self.context_length,
            'pre_context': self.pre_context,
            'post_context': self.post_context,
            'pam_levels': self.pam_levels,
            'total_trainable_params': nn_params + self.n_ffe_taps,
            'optimizer': 'adam' if self.use_adam else 'sgd',
        }
