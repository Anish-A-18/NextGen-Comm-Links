#!/usr/bin/env python3
"""
Generalized PAM MLSE Implementation for SerDes
================================================

Supports PAM-4, PAM-6, and PAM-8 signaling with configurable channel memory.

Based on pam4_mlse_robust.py but extended for multi-level PAM.

Key features:
1. Proper channel estimation with correct alignment
2. Robust Viterbi decoder for L=1 and L=2
3. Correct traceback and symbol alignment
4. Numerical stability improvements
5. Support for PAM-4, PAM-6, and PAM-8

Author: Anish Anand
Date: December 2025
"""

import numpy as np
import pickle
from typing import Tuple, Dict, Optional, List

np.random.seed(42)


def get_pam_levels(n_levels: int) -> np.ndarray:
    """Get PAM levels for given number of levels."""
    if n_levels == 4:
        return np.array([-3, -1, 1, 3], dtype=float)
    elif n_levels == 6:
        return np.array([-5, -3, -1, 1, 3, 5], dtype=float)
    elif n_levels == 8:
        return np.array([-7, -5, -3, -1, 1, 3, 5, 7], dtype=float)
    else:
        raise ValueError(f"Unsupported PAM level: {n_levels}. Use 4, 6, or 8.")


class PAM_MLSE_Equalizer:
    """
    Generalized PAM MLSE Equalizer with training and prediction.
    
    Supports PAM-4, PAM-6, and PAM-8 signaling.
    
    This is a wrapper that makes MLSE easy to use, similar to ML equalizer interface.
    """
    
    def __init__(self, n_levels: int = 4, channel_memory: int = 1, 
                 traceback_depth: Optional[int] = None, verbose: bool = True):
        """
        Initialize MLSE equalizer.
        
        Parameters
        ----------
        n_levels : int
            Number of PAM levels (4, 6, or 8)
        channel_memory : int
            Channel memory length L (1 or 2)
            L=1 means the channel has 1 post-cursor tap (2 taps total).
            L=2 means the channel has 2 post-cursor taps (3 taps total).
        traceback_depth : int, optional
            Traceback depth D. If None, uses max(5*L, 10).
            This determines how far back the Viterbi algorithm looks to make a decision.
            Larger depth improves reliability but increases latency.
        verbose : bool
            Print initialization info
        """
        self.n_levels = n_levels
        self.pam_levels = get_pam_levels(n_levels)
        self.channel_memory = channel_memory
        self.num_states = n_levels ** channel_memory
        
        if traceback_depth is None:
            self.traceback_depth = max(5 * channel_memory, 10)
        else:
            self.traceback_depth = traceback_depth
        
        self.channel_taps = None
        self.viterbi = None
        self.verbose_init = verbose
        
        if verbose:
            print(f"\nPAM-{n_levels} MLSE Equalizer Initialized:")
            print(f"  Channel memory L: {self.channel_memory}")
            print(f"  Number of states: {self.num_states} ({n_levels}^{self.channel_memory})")
            print(f"  Traceback depth: {self.traceback_depth}")
    
    def train(self, ffe_output: np.ndarray, true_symbols: np.ndarray,
              samples_per_symbol: int = 1, verbose: bool = True) -> Dict:
        """
        Train MLSE by estimating channel taps from known sequence.
        
        Parameters
        ----------
        ffe_output : ndarray
            FFE output samples (one per symbol if samples_per_symbol=1)
        true_symbols : ndarray
            Known transmitted symbols (0 to n_levels-1)
        samples_per_symbol : int
            If >1, will downsample ffe_output
        verbose : bool
            Print training info
            
        Returns
        -------
        info : dict
            Training information including estimated channel taps
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"Training PAM-{self.n_levels} MLSE Equalizer (Channel Estimation)")
            print(f"{'='*70}")
        
        # Downsample if needed
        if samples_per_symbol > 1:
            ffe_output = ffe_output[::samples_per_symbol]
        
        # Ensure same length
        min_len = min(len(ffe_output), len(true_symbols))
        ffe_output = ffe_output[:min_len]
        true_symbols = true_symbols[:min_len]
        
        if verbose:
            print(f"  Training samples: {len(ffe_output)}")
            print(f"  Channel memory L: {self.channel_memory}")
        
        # Estimate channel taps using least squares
        self.channel_taps = self._estimate_channel_robust(ffe_output, true_symbols, verbose)
        
        # Create Viterbi decoder
        self.viterbi = ViterbiDecoderPAM(self.channel_taps, self.traceback_depth, 
                                          self.n_levels, self.pam_levels)
        
        info = {
            'channel_taps': self.channel_taps,
            'main_tap': self.channel_taps[0],
            'isi_taps': self.channel_taps[1:] if len(self.channel_taps) > 1 else [],
            'num_states': self.num_states,
            'traceback_depth': self.traceback_depth,
            'n_levels': self.n_levels
        }
        
        return info
    
    def _estimate_channel_robust(self, rx_samples: np.ndarray, 
                                  tx_symbols: np.ndarray, verbose: bool = True) -> np.ndarray:
        """
        Robust channel estimation using least squares with regularization.
        
        Model: r[n] = h[0]*s[n] + h[1]*s[n-1] + ... + h[L]*s[n-L] + noise
        """
        L = self.channel_memory
        N = len(tx_symbols)
        
        # Convert symbols to PAM levels
        tx_levels = self.pam_levels[tx_symbols.astype(int)]
        
        # Build design matrix S
        # Each row is [s[n], s[n-1], ..., s[n-L]]
        S = np.zeros((N - L, L + 1))
        
        for n in range(L, N):
            for k in range(L + 1):
                S[n - L, k] = tx_levels[n - k]
        
        # Corresponding received samples
        R = rx_samples[L:N]
        
        # Least squares with small regularization for stability
        # h = (S^T S + λI)^(-1) S^T R
        lambda_reg = 1e-6
        STI = S.T @ S + lambda_reg * np.eye(L + 1)
        h = np.linalg.solve(STI, S.T @ R)
        
        if verbose:
            print(f"\nEstimated Channel Taps:")
            print(f"  h[0] (main tap):     {h[0]:.6f}")
            for i in range(1, len(h)):
                print(f"  h[{i}] (ISI tap {i}):     {h[i]:.6f}")
            
            # Calculate residual MSE
            predicted = S @ h
            mse = np.mean((R - predicted) ** 2)
            print(f"  Channel fit MSE:     {mse:.6e}")
            
            # Normalize check
            if abs(h[0]) < 0.1:
                print(f"  WARNING: Main tap is small ({h[0]:.6f}). Channel estimation may be poor.")
        
        return h
    
    def print_trellis(self):
        """Print the trellis structure (states and expected outputs)."""
        if self.viterbi:
            self.viterbi.print_trellis()
        else:
            print("Viterbi decoder not initialized.")

    def predict(self, ffe_output: np.ndarray, samples_per_symbol: int = 1,
                verbose: bool = False, debug_steps: int = 0) -> np.ndarray:
        """
        Predict symbols using trained Viterbi decoder.
        
        Parameters
        ----------
        ffe_output : ndarray
            FFE output samples (input to MLSE)
        samples_per_symbol : int
            Downsampling factor (default 1)
        verbose : bool
            Print progress information
        debug_steps : int
            Number of initial steps to print detailed debug info for (default 0)
            
        Returns
        -------
        detected_symbols : ndarray
            Detected PAM symbols (0 to n_levels-1)
        """
        if self.viterbi is None:
            raise ValueError("Must train() before predict()")
        
        # Downsample if needed
        if samples_per_symbol > 1:
            ffe_output = ffe_output[::samples_per_symbol]
        
        if verbose:
            print(f"\nRunning Viterbi Decoder...")
            print(f"  Input samples: {len(ffe_output)}")
        
        # Reset Viterbi state
        self.viterbi.reset()
        
        # Process all samples
        for i, sample in enumerate(ffe_output):
            # Print debug info for the first few steps if requested
            debug = (i < debug_steps)
            if debug:
                print(f"\n--- Step {i} (Sample: {sample:.4f}) ---")
                
            self.viterbi.update(sample, debug=debug)
            
            if verbose and (i + 1) % 5000 == 0:
                print(f"  Processed {i+1}/{len(ffe_output)} samples...")
        
        # Traceback to get decisions
        detected = self.viterbi.traceback()
        
        # Trim to match input length
        detected = detected[:len(ffe_output)]
        
        if verbose:
            print(f"  Decoded {len(detected)} symbols")
        
        return np.array(detected)
    
    def save(self, filepath: str):
        """Save trained model."""
        model_data = {
            'n_levels': self.n_levels,
            'channel_memory': self.channel_memory,
            'traceback_depth': self.traceback_depth,
            'channel_taps': self.channel_taps
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"  MLSE model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load trained model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.n_levels = model_data['n_levels']
        self.pam_levels = get_pam_levels(self.n_levels)
        self.channel_memory = model_data['channel_memory']
        self.traceback_depth = model_data['traceback_depth']
        self.channel_taps = model_data['channel_taps']
        self.num_states = self.n_levels ** self.channel_memory
        
        # Recreate Viterbi decoder
        self.viterbi = ViterbiDecoderPAM(self.channel_taps, self.traceback_depth,
                                          self.n_levels, self.pam_levels)
        
        print(f"  MLSE model loaded from {filepath}")


class ViterbiDecoderPAM:
    """
    Efficient Viterbi decoder for generalized PAM with proper state transitions.
    
    Supports PAM-4, PAM-6, and PAM-8.
    """
    
    def __init__(self, channel_taps: np.ndarray, traceback_depth: int,
                 n_levels: int, pam_levels: np.ndarray):
        """
        Initialize Viterbi decoder.
        
        Parameters
        ----------
        channel_taps : ndarray
            Channel coefficients [h[0], h[1], ..., h[L]]
        traceback_depth : int
            Traceback depth D
        n_levels : int
            Number of PAM levels (4, 6, or 8)
        pam_levels : ndarray
            PAM level values (e.g., [-3, -1, 1, 3] for PAM-4)
        """
        self.h = np.array(channel_taps, dtype=float)
        self.L = len(self.h) - 1  # Channel memory
        self.D = traceback_depth
        self.buffer_size = self.D * 2  # Double buffer to prevent overwrite during traceback
        self.n_levels = n_levels
        self.pam_levels = pam_levels
        self.num_states = n_levels ** self.L if self.L > 0 else 1
        
        # Pre-compute state transition table for efficiency
        self._build_transition_table()
        
        # Initialize
        self.reset()
    
    def print_trellis(self):
        """Print all state transitions and expected outputs."""
        print(f"\nTrellis Structure (L={self.L}, {self.num_states} states, PAM-{self.n_levels}):")
        print(f"{'From State':<12} {'Symbol':<8} {'To State':<10} {'Expected Output':<15}")
        print("-" * 50)
        
        # Sort by from_state, then symbol
        sorted_trans = sorted(self.transitions, key=lambda x: (x['from_state'], x['symbol']))
        
        for t in sorted_trans:
            print(f"{t['from_state']:<12} {t['symbol']:<8} {t['to_state']:<10} {t['expected']:<15.4f}")

    def _build_transition_table(self):
        """Pre-compute valid state transitions and expected outputs."""
        self.transitions = []  # List of (from_state, symbol, to_state, expected_output)
        
        for from_state in range(self.num_states):
            from_symbols = self._state_to_symbols(from_state)
            
            for symbol in range(self.n_levels):  # Try all PAM symbols
                to_state = self._get_next_state(from_state, symbol)
                expected = self._compute_expected(symbol, from_symbols)
                
                self.transitions.append({
                    'from_state': from_state,
                    'symbol': symbol,
                    'to_state': to_state,
                    'expected': expected
                })
    
    def _state_to_symbols(self, state: int) -> np.ndarray:
        """Convert state index to symbol sequence (base-n_levels representation)."""
        if self.L == 0:
            return np.array([])
        
        symbols = np.zeros(self.L, dtype=int)
        temp = state
        for i in range(self.L - 1, -1, -1):
            symbols[i] = temp % self.n_levels
            temp //= self.n_levels
        return symbols
    
    def _get_next_state(self, prev_state: int, current_symbol: int) -> int:
        """Compute next state given previous state and current symbol."""
        if self.L == 0:
            return 0
        elif self.L == 1:
            # For L=1, next state IS the current symbol
            return current_symbol
        else:
            # For L>1, shift window: drop oldest, add current
            prev_symbols = self._state_to_symbols(prev_state)
            next_symbols = np.concatenate([prev_symbols[1:], [current_symbol]])
            
            # Convert back to state index
            state = 0
            for sym in next_symbols:
                state = state * self.n_levels + sym
            return state
    
    def _compute_expected(self, current_symbol: int, prev_symbols: np.ndarray) -> float:
        """Compute expected output for given symbol and history."""
        current_level = self.pam_levels[current_symbol]
        prev_levels = self.pam_levels[prev_symbols] if len(prev_symbols) > 0 else np.array([])
        
        expected = self.h[0] * current_level
        for k in range(1, min(self.L + 1, len(self.h))):
            if k - 1 < len(prev_levels):
                expected += self.h[k] * prev_levels[-k]
        
        return expected
    
    def reset(self):
        """Reset decoder state."""
        # Path metrics (cumulative cost)
        self.path_metrics = np.full(self.num_states, np.inf, dtype=float)
        self.path_metrics[0] = 0.0  # Start from state 0
        
        # Survivor history (circular buffer)
        self.survivor_states = np.zeros((self.num_states, self.buffer_size), dtype=int)
        self.survivor_symbols = np.zeros((self.num_states, self.buffer_size), dtype=int)
        self.time_index = 0
        
        # Store all decoded symbols
        self.all_decoded_symbols = []
    
    def update(self, received_sample: float, debug: bool = False):
        """
        Process one received sample (Add-Compare-Select operation).
        Performs fixed-lag decoding when buffer is full.
        
        Parameters
        ----------
        received_sample : float
            Received sample after FFE
        debug : bool
            Print debug info for this step
        """
        new_path_metrics = np.full(self.num_states, np.inf, dtype=float)
        new_survivor_states = np.zeros(self.num_states, dtype=int)
        new_survivor_symbols = np.zeros(self.num_states, dtype=int)
        
        # For each possible current state (to_state)
        for to_state in range(self.num_states):
            best_metric = np.inf
            best_from_state = 0
            best_symbol = 0
            
            # Try all transitions that lead to this state
            for trans in self.transitions:
                if trans['to_state'] == to_state:
                    from_state = trans['from_state']
                    symbol = trans['symbol']
                    expected = trans['expected']
                    
                    # Branch metric (Euclidean distance squared)
                    branch_metric = (received_sample - expected) ** 2
                    
                    # Path metric (cumulative cost)
                    path_metric = self.path_metrics[from_state] + branch_metric
                    
                    # Keep best (minimum cost)
                    if path_metric < best_metric:
                        best_metric = path_metric
                        best_from_state = from_state
                        best_symbol = symbol
            
            new_path_metrics[to_state] = best_metric
            new_survivor_states[to_state] = best_from_state
            new_survivor_symbols[to_state] = best_symbol
        
        if debug:
            best_state = np.argmin(new_path_metrics)
            print(f"  Best State: {best_state} (Metric: {new_path_metrics[best_state]:.4f})")
            print(f"  Survivor States (prev -> curr):")
            for s in range(min(self.num_states, 8)):  # Print first 8 states
                if new_path_metrics[s] < np.inf:
                    print(f"    State {s:2d}: from {new_survivor_states[s]:2d}, cost {new_path_metrics[s]:.4f}")
        
        # Update path metrics
        self.path_metrics = new_path_metrics
        
        # Store survivor information (circular buffer)
        buf_idx = self.time_index % self.buffer_size
        self.survivor_states[:, buf_idx] = new_survivor_states
        self.survivor_symbols[:, buf_idx] = new_survivor_symbols
        
        # Fixed-lag decoding: Once buffer is full, decode oldest symbol
        if self.time_index >= self.D:
            # Trace back from current best state
            current_state = np.argmin(self.path_metrics)
            
            # Go back D steps to find the decoded symbol
            for d in range(self.D + 1):
                back_idx = (self.time_index - d) % self.buffer_size
                
                # If this is the oldest symbol in the buffer (depth D)
                if d == self.D:
                    decoded_symbol = self.survivor_symbols[current_state, back_idx]
                    self.all_decoded_symbols.append(decoded_symbol)
                    if debug:
                        print(f"  Decoded Symbol (delayed): {decoded_symbol}")
                
                current_state = self.survivor_states[current_state, back_idx]
        
        self.time_index += 1
    
    def traceback(self) -> List[int]:
        """
        Final traceback for remaining symbols in buffer.
        
        Returns
        -------
        decoded_symbols : list
            All decoded symbols (chronological order)
        """
        # Start from state with minimum path metric
        current_state = np.argmin(self.path_metrics)
        
        final_decoded = []
        
        # Trace back remaining symbols (< D symbols at end)
        remaining = min(self.time_index, self.D)
        for d in range(remaining):
            buf_idx = (self.time_index - 1 - d) % self.buffer_size
            
            symbol = self.survivor_symbols[current_state, buf_idx]
            final_decoded.append(symbol)
            
            # Move to previous state
            current_state = self.survivor_states[current_state, buf_idx]
        
        # Combine all decoded + final traceback (reverse final part because we traced backwards)
        all_symbols = self.all_decoded_symbols + final_decoded[::-1]
        
        return all_symbols


# Utility functions
def calculate_symbol_errors(detected: np.ndarray, transmitted: np.ndarray) -> Tuple[int, float]:
    """Calculate symbol errors and SER."""
    min_len = min(len(detected), len(transmitted))
    detected = detected[:min_len]
    transmitted = transmitted[:min_len]
    
    errors = np.sum(detected != transmitted)
    ser = errors / min_len if min_len > 0 else 0.0
    
    return errors, ser


def estimate_composite_channel_pam(tx_symbols: np.ndarray, rx_samples: np.ndarray,
                                    rx_rms: float, n_levels: int, channel_memory: int = 1,
                                    verbose: bool = True) -> np.ndarray:
    """
    Estimate composite channel response for PAM signaling.
    
    Parameters
    ----------
    tx_symbols : ndarray
        Transmitted symbols (0 to n_levels-1)
    rx_samples : ndarray
        Received samples after FFE
    rx_rms : float
        RMS of received signal
    n_levels : int
        Number of PAM levels (4, 6, or 8)
    channel_memory : int
        Channel memory L
    verbose : bool
        Print estimation info
        
    Returns
    -------
    h : ndarray
        Estimated channel taps
    """
    pam_levels = get_pam_levels(n_levels)
    L = channel_memory
    N = min(len(tx_symbols), len(rx_samples))
    
    # Convert symbols to levels
    tx_levels = pam_levels[tx_symbols[:N].astype(int)]
    
    # Build design matrix
    S = np.zeros((N - L, L + 1))
    for n in range(L, N):
        for k in range(L + 1):
            S[n - L, k] = tx_levels[n - k]
    
    R = rx_samples[L:N]
    
    # Least squares with regularization
    lambda_reg = 1e-6
    STI = S.T @ S + lambda_reg * np.eye(L + 1)
    h = np.linalg.solve(STI, S.T @ R)
    
    if verbose:
        print(f"\nComposite Channel Estimation (PAM-{n_levels}, L={L}):")
        print(f"  h[0] (main): {h[0]:.6f}")
        for i in range(1, len(h)):
            print(f"  h[{i}] (ISI):  {h[i]:.6f}")
    
    return h


if __name__ == "__main__":
    """Test the implementation for PAM-4, PAM-6, and PAM-8."""
    
    print("\n" + "="*70)
    print("Testing Generalized PAM MLSE Implementation")
    print("="*70)
    
    for n_levels in [4, 6, 8]:
        print(f"\n{'='*70}")
        print(f"Testing PAM-{n_levels}")
        print(f"{'='*70}")
        
        pam_levels = get_pam_levels(n_levels)
        
        # Test L=1
        print(f"\n--- PAM-{n_levels}, L=1 ({n_levels} states) ---")
        h1 = np.array([1.0, 0.3])
        
        np.random.seed(42)
        test_symbols = np.random.randint(0, n_levels, size=100)
        test_levels = pam_levels[test_symbols]
        
        # Simulate channel
        received = np.zeros(len(test_levels))
        for n in range(len(test_levels)):
            received[n] = h1[0] * test_levels[n]
            if n > 0:
                received[n] += h1[1] * test_levels[n-1]
        received += np.random.normal(0, 0.05, len(received))
        
        # Create and train MLSE
        mlse1 = PAM_MLSE_Equalizer(n_levels=n_levels, channel_memory=1, verbose=False)
        mlse1.train(received, test_symbols, verbose=True)
        
        # Predict
        detected1 = mlse1.predict(received, verbose=True)
        errors1, ser1 = calculate_symbol_errors(detected1, test_symbols)
        print(f"\nL=1 Results: Errors={errors1}/{len(test_symbols)}, SER={ser1:.4e}")
        
        # Test L=2
        print(f"\n--- PAM-{n_levels}, L=2 ({n_levels**2} states) ---")
        h2 = np.array([1.0, 0.3, 0.15])
        
        received2 = np.zeros(len(test_levels))
        for n in range(len(test_levels)):
            received2[n] = h2[0] * test_levels[n]
            if n > 0:
                received2[n] += h2[1] * test_levels[n-1]
            if n > 1:
                received2[n] += h2[2] * test_levels[n-2]
        received2 += np.random.normal(0, 0.05, len(received2))
        
        # Create and train MLSE
        mlse2 = PAM_MLSE_Equalizer(n_levels=n_levels, channel_memory=2, verbose=False)
        mlse2.train(received2, test_symbols, verbose=True)
        
        # Predict
        detected2 = mlse2.predict(received2, verbose=True)
        errors2, ser2 = calculate_symbol_errors(detected2, test_symbols)
        print(f"\nL=2 Results: Errors={errors2}/{len(test_symbols)}, SER={ser2:.4e}")
    
    print("\n" + "="*70)
    print("All tests complete!")
    print("="*70)

