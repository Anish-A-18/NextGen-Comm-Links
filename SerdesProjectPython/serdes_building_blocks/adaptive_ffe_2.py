"""
Adaptive Feed-Forward Equalizer (FFE) for SerDes Receiver
===========================================================

Implements multi-tap FFE with adaptive algorithms:
- LMS (Least Mean Squares)
- Sign-Sign LMS
- RLS (Recursive Least Squares)

Supports 4-64 taps with configurable pre-cursor and post-cursor structure.

Author: Anish Anand
Date: November 8, 2025
"""

import numpy as np
from typing import Optional, Dict, Tuple, List
import matplotlib.pyplot as plt


class AdaptiveFFE:
    """
    Adaptive Feed-Forward Equalizer for digital equalization.
    
    Structure: [pre_n, ..., pre_1, cursor, post_1, ..., post_m]
    Default: 16 taps with pre_n=15, cursor=1, post_m=0
    
    Parameters
    ----------
    n_taps : int
        Total number of taps (4-64)
    n_precursor : int
        Number of pre-cursor taps (default: 15)
    n_postcursor : int
        Number of post-cursor taps (default: 0)
    algorithm : str
        Adaptation algorithm: 'lms', 'sign_sign_lms', 'rls'
    mu : float
        Step size for LMS algorithms (default: 0.01)
    lambda_rls : float
        Forgetting factor for RLS (default: 0.99)
    delta_rls : float
        Initialization constant for RLS (default: 1.0)
    adc_rms : float
        RMS value for ADC scaling (default: 0.5)
    normalize_interval : int
        Normalize taps every N adaptation steps (default: 100, 0 = disable)
        
    Attributes
    ----------
    taps : ndarray
        Current tap coefficients [w_0, w_1, ..., w_{N-1}]
    cursor_index : int
        Index of the cursor (main) tap
    """
    
    def __init__(self,
        n_taps: int = 16,
        n_precursor: int = 15,
        n_postcursor: int = 0,
        algorithm: str = 'lms',
        mu: float = 0.005,
        lambda_rls: float = 0.99,
        delta_rls: float = 1.0,
        adc_rms: float = 0.5,
        normalize_interval: int = 100,
        n_levels: int = 4,
        pam_alpha: float = 0.87):

        """Initialize Adaptive FFE.
        
        Parameters
        ----------
        n_levels : int
            Number of PAM levels (4, 6, or 8). Default: 4 (PAM-4)
        pam_alpha : float
            Alpha scaling factor for thresholds and levels. Default: 0.87
        """
        
        if not (4 <= n_taps <= 64):
            raise ValueError("Number of taps must be between 4 and 64")
        
        if n_precursor + n_postcursor + 1 != n_taps:
            raise ValueError(f"n_precursor ({n_precursor}) + 1 + n_postcursor ({n_postcursor}) must equal n_taps ({n_taps})")
        
        if algorithm not in ['lms', 'sign_sign_lms', 'rls']:
            raise ValueError("Algorithm must be 'lms', 'sign_sign_lms', or 'rls'")
        
        if n_levels not in [4, 6, 8]:
            raise ValueError(f"n_levels must be 4, 6, or 8, got {n_levels}")
        
        self.n_taps = n_taps
        self.n_precursor = n_precursor
        self.n_postcursor = n_postcursor
        self.cursor_index = n_precursor  # Cursor tap index
        self.algorithm = algorithm
        self.mu = mu
        self.lambda_rls = lambda_rls
        self.delta_rls = delta_rls
        self.adc_rms = adc_rms
        self.normalize_interval = normalize_interval
        self.n_levels = n_levels
        self.pam_alpha = pam_alpha
        
        # Initialize PAM levels and thresholds based on n_levels
        self._init_pam_levels()
        # Initialize taps: cursor = 1, others = 0
        self.taps = np.zeros(n_taps)
        self.taps[self.cursor_index] = 1.0
        
        # RLS-specific parameters
        if algorithm == 'rls':
            self.P = np.eye(n_taps) * delta_rls  # Inverse correlation matrix
        
        # Input buffer for tap delay line
        self.input_buffer = np.zeros(n_taps)
        
        # Adaptation history
        self.tap_history = [self.taps.copy()]
        self.error_history = []
        self.mse_history = []
        
        # Statistics
        self.n_adaptations = 0
        self.converged = False
    
    def _init_pam_levels(self):
        """Initialize PAM levels and thresholds based on n_levels."""
        if self.n_levels == 4:
            # PAM-4: levels at -1.5, -0.5, 0.5, 1.5 (normalized)
            base_levels = np.array([-1.5, -0.5, 0.5, 1.5])
            base_thresholds = np.array([-1.0, 0.0, 1.0])
        elif self.n_levels == 6:
            # PAM-6: levels at -5/3, -1, -1/3, 1/3, 1, 5/3 (normalized to similar RMS as PAM-4)
            base_levels = np.array([-5, -3, -1, 1, 3, 5]) / 3.0
            base_thresholds = np.array([-4, -2, 0, 2, 4]) / 3.0
        elif self.n_levels == 8:
            # PAM-8: levels at -7/4, -5/4, -3/4, -1/4, 1/4, 3/4, 5/4, 7/4
            base_levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7]) / 4.0
            base_thresholds = np.array([-6, -4, -2, 0, 2, 4, 6]) / 4.0
        
        # Apply adc_rms scaling and pam_alpha
        self.pam_levels = base_levels * self.adc_rms * self.pam_alpha
        self.thresholds = base_thresholds * self.adc_rms * self.pam_alpha
        
    def reset_taps(self):
        """Reset taps to initial condition (cursor only)."""
        self.taps = np.zeros(self.n_taps)
        self.taps[self.cursor_index] = 1.0
        self.input_buffer = np.zeros(self.n_taps)
        
        if self.algorithm == 'rls':
            self.P = np.eye(self.n_taps) * self.delta_rls
            
        self.tap_history = [self.taps.copy()]
        self.error_history = []
        self.mse_history = []
        self.n_adaptations = 0
        self.converged = False
        
    def normalize_taps_cursor(self):
        """
        Normalize taps so cursor tap = 1.0 (Method 2).
        
        This maintains the cursor tap as the reference (gain = 1.0)
        and scales all other taps proportionally. This is the standard
        approach for SerDes FFE design.
        
        Only called during adaptation phase, not during inference.
        """
        cursor_value = self.taps[self.cursor_index]
        if abs(cursor_value) > 1e-10:  # Avoid division by very small numbers
            self.taps /= cursor_value
        
    def equalize_sample(self, x: float) -> float:
        """
        Equalize single input sample.
        
        Parameters
        ----------
        x : float
            Input sample
            
        Returns
        -------
        y : float
            Equalized output
        """
        # Shift input buffer
        self.input_buffer = np.roll(self.input_buffer, 1)
        self.input_buffer[0] = x
        
        # Compute output: y = sum(w_k * x[n-k])
        y = np.dot(self.taps, self.input_buffer)
        
        return y
        
    def equalize(self, signal: np.ndarray) -> np.ndarray:
        """
        Equalize entire signal.
        
        Parameters
        ----------
        signal : ndarray
            Input signal samples
            
        Returns
        -------
        output : ndarray
            Equalized output
        """
        output = np.zeros(len(signal))
        
        for n in range(len(signal)):
            output[n] = self.equalize_sample(signal[n])
            
        return output
        
    def adapt_lms(self, error: float, save_history: bool = True):
        """
        Adapt taps using LMS algorithm.
        
        Update: w[n+1] = w[n] + mu * e[n] * x[n]
        
        Parameters
        ----------
        error : float
            Error signal e[n] = desired[n] - output[n]
        save_history : bool
            Whether to save adaptation history
        """
        # LMS update
        self.taps += self.mu * error * self.input_buffer
        
        self.n_adaptations += 1
        
        # Normalize taps periodically during adaptation
        if self.normalize_interval > 0 and self.n_adaptations % self.normalize_interval == 0:
            self.normalize_taps_cursor()
        
        if save_history:
            self.tap_history.append(self.taps.copy())
            self.error_history.append(error)
            
    def adapt_sign_sign_lms(self, error: float, save_history: bool = True):
        """
        Adapt taps using Sign-Sign LMS algorithm.
        
        Update: w[n+1] = w[n] + mu * sign(e[n]) * sign(x[n])
        
        Parameters
        ----------
        error : float
            Error signal
        save_history : bool
            Whether to save adaptation history
        """
        # Sign-Sign LMS update
        self.taps += self.mu * np.sign(error) * np.sign(self.input_buffer)
        
        self.n_adaptations += 1
        
        # Normalize taps periodically during adaptation
        if self.normalize_interval > 0 and self.n_adaptations % self.normalize_interval == 0:
            self.normalize_taps_cursor()
        
        if save_history:
            self.tap_history.append(self.taps.copy())
            self.error_history.append(error)
            
    def adapt_rls(self, error: float, save_history: bool = True):
        """
        Adapt taps using RLS algorithm.
        
        Update:
          k[n] = P[n-1] * x[n] / (lambda + x[n]^T * P[n-1] * x[n])
          w[n+1] = w[n] + k[n] * e[n]
          P[n] = (P[n-1] - k[n] * x[n]^T * P[n-1]) / lambda
        
        Parameters
        ----------
        error : float
            Error signal
        save_history : bool
            Whether to save adaptation history
            
        Note
        ----
        RLS normalization is DISABLED because normalizing the taps invalidates
        the RLS P matrix (inverse correlation matrix), causing instability.
        RLS maintains its own implicit normalization through the P matrix update.
        """
        x = self.input_buffer
        
        # Compute gain vector
        Px = self.P @ x
        denominator = self.lambda_rls + x @ Px
        
        if denominator > 1e-10:  # Avoid division by zero
            k = Px / denominator
            
            # Update taps
            self.taps += k * error
            
            # Update inverse correlation matrix
            self.P = (self.P - np.outer(k, Px)) / self.lambda_rls
        
        self.n_adaptations += 1
        
        # NOTE: Normalization is DISABLED for RLS
        # Normalizing taps breaks the RLS P matrix consistency, causing instability
        # If you really need normalized taps with RLS, normalize AFTER convergence
        
        if save_history:
            self.tap_history.append(self.taps.copy())
            self.error_history.append(error)
            
    def adapt(self, error: float, save_history: bool = True):
        """
        Adapt taps using configured algorithm.
        
        Parameters
        ----------
        error : float
            Error signal
        save_history : bool
            Whether to save adaptation history
        """
        if self.algorithm == 'lms':
            self.adapt_lms(error, save_history)
        elif self.algorithm == 'sign_sign_lms':
            self.adapt_sign_sign_lms(error, save_history)
        elif self.algorithm == 'rls':
            self.adapt_rls(error, save_history)
            
    def process_and_adapt(self, 
                         signal: np.ndarray,
                         desired: Optional[np.ndarray] = None,
                         save_history: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process signal with adaptation (training mode).
        
        Parameters
        ----------
        signal : ndarray
            Input signal
        desired : ndarray, optional
            Desired output (for training). If None, uses decision-directed mode.
        save_history : bool
            Whether to save adaptation history
            
        Returns
        -------
        output : ndarray
            Equalized output
        errors : ndarray
            Error signal for each sample
        """
        n_samples = len(signal)
        output = np.zeros(n_samples)
        errors = np.zeros(n_samples)
        
        for n in range(n_samples):
            # Equalize
            output[n] = self.equalize_sample(signal[n])
            
            # Compute error
            if desired is not None:
                # Training mode: use known desired signal
                errors[n] = desired[n] - output[n]
            else:
                # Decision-directed: use sliced output as desired
                # Uses PAM levels based on n_levels setting (PAM-4, PAM-6, or PAM-8)
                sliced = self.slice_pam(output[n])
                errors[n] = sliced - output[n]
            
            # Adapt
            self.adapt(errors[n], save_history)
            
        # Compute MSE for this block
        if save_history:
            mse = np.mean(errors**2)
            self.mse_history.append(mse)
        
            
        return output, errors

    def align_sequences(self, detected_symbols: np.ndarray, tx_symbols: np.ndarray, max_search: int = 100) -> Tuple[int, float]:
        """
        Find optimal alignment between detected and transmitted symbols using cross-correlation.
        
        Returns:
            offset: Number of samples to shift tx_symbols to align with detected_symbols
            correlation: Peak correlation value
        """
        search_length = min(1000, len(detected_symbols), len(tx_symbols))
        detected_slice = detected_symbols[:search_length]
        
        best_corr = -np.inf
        best_offset = 0
        
        # Search for best alignment
        for offset in range(-max_search, max_search):
            if offset < 0:
                tx_slice = tx_symbols[-offset:min(-offset + search_length, len(tx_symbols))]
                det_slice = detected_slice[:len(tx_slice)]
            else:
                tx_slice = tx_symbols[:min(search_length - offset, len(tx_symbols))]
                det_slice = detected_slice[offset:offset + len(tx_slice)]
            
            if len(tx_slice) < 100 or len(det_slice) < 100:
                continue
            
            # Compute correlation (matching symbols)
            corr = np.sum(tx_slice == det_slice) / len(tx_slice)
            
            if corr > best_corr:
                best_corr = corr
                best_offset = offset
        
        return best_offset, best_corr
        
    def slice_pam(self, value: float, 
                  levels: Optional[np.ndarray] = None) -> float:
        """
        Slice value to nearest PAM level.
        
        Supports PAM-4, PAM-6, and PAM-8 based on n_levels setting.
        
        Parameters
        ----------
        value : float
            Input value
        levels : ndarray, optional
            PAM levels. If None, uses self.pam_levels (set by n_levels and pam_alpha)
            
        Returns
        -------
        sliced : float
            Nearest PAM level
        """
        if levels is None:
            levels = self.pam_levels
        
        # Find nearest level
        idx = np.argmin(np.abs(levels - value))
        return levels[idx]
    
    def slice_pam4(self, value: float, 
                   levels: Optional[np.ndarray] = None) -> float:
        """
        Slice value to nearest PAM4 level.
        
        Deprecated: Use slice_pam() instead.
        Kept for backward compatibility.
        
        Parameters
        ----------
        value : float
            Input value
        levels : ndarray, optional
            PAM4 levels. If None, uses normalized [-1.5, -0.5, 0.5, 1.5] * RMS
            
        Returns
        -------
        sliced : float
            Nearest PAM4 level
        """
        if levels is None:
            # Default normalized PAM4 levels (backward compatible)
            levels = np.array([-1.5, -0.5, 0.5, 1.5]) * self.adc_rms
        
        # Find nearest level
        idx = np.argmin(np.abs(levels - value))
        return levels[idx]
        
    def get_taps(self) -> np.ndarray:
        """Get current tap coefficients."""
        return self.taps.copy()
        
    def set_taps(self, taps: np.ndarray):
        """
        Set tap coefficients.
        
        Parameters
        ----------
        taps : ndarray
            Tap coefficients (length must match n_taps)
        """
        if len(taps) != self.n_taps:
            raise ValueError(f"Taps length must be {self.n_taps}")
        self.taps = np.array(taps)
        
    def get_frequency_response(self, freqs: np.ndarray, 
                              fs: float = 1.0) -> np.ndarray:
        """
        Compute frequency response of FFE.
        
        Parameters
        ----------
        freqs : ndarray
            Frequencies to evaluate [Hz]
        fs : float
            Sampling frequency [Hz]
            
        Returns
        -------
        H : ndarray (complex)
            Frequency response
        """
        # FFE transfer function: H(f) = sum(w_k * exp(-j*2*pi*f*k/fs))
        H = np.zeros(len(freqs), dtype=complex)
        
        for k in range(self.n_taps):
            H += self.taps[k] * np.exp(-1j * 2 * np.pi * freqs * k / fs)
            
        return H
        
    def check_convergence(self, window: int = 100, threshold: float = 1e-6) -> bool:
        """
        Check if adaptation has converged.
        
        Parameters
        ----------
        window : int
            Window size for checking convergence
        threshold : float
            MSE threshold for convergence
            
        Returns
        -------
        converged : bool
            True if converged
        """
        if len(self.mse_history) < window:
            return False
            
        recent_mse = self.mse_history[-window:]
        mse_std = np.std(recent_mse)
        mse_mean = np.mean(recent_mse)
        
        # Converged if MSE is low and stable
        if mse_mean < threshold and mse_std < threshold / 10:
            self.converged = True
            return True
            
        return False
        
    def get_config(self) -> Dict:
        """Get FFE configuration."""
        return {
            'n_taps': self.n_taps,
            'n_precursor': self.n_precursor,
            'n_postcursor': self.n_postcursor,
            'cursor_index': self.cursor_index,
            'algorithm': self.algorithm,
            'mu': self.mu,
            'lambda_rls': self.lambda_rls,
            'delta_rls': self.delta_rls,
            'normalize_interval': self.normalize_interval,
            'n_adaptations': self.n_adaptations,
            'converged': self.converged,
        }
        
    def print_info(self):
        """Print FFE configuration and status."""
        print("\n" + "="*60)
        print("Adaptive FFE Configuration")
        print("="*60)
        print(f"Total Taps:          {self.n_taps}")
        print(f"Pre-cursor Taps:     {self.n_precursor}")
        print(f"Cursor Tap:          Index {self.cursor_index}")
        print(f"Post-cursor Taps:    {self.n_postcursor}")
        print(f"Algorithm:           {self.algorithm.upper()}")
        
        if self.algorithm in ['lms', 'sign_sign_lms']:
            print(f"Step Size (μ):       {self.mu}")
        elif self.algorithm == 'rls':
            print(f"Forgetting Factor:   {self.lambda_rls}")
            print(f"Initialization δ:    {self.delta_rls}")
        
        print(f"Normalize Interval:  {self.normalize_interval if self.normalize_interval > 0 else 'Disabled'}")
        
        print(f"\nAdaptation Status:")
        print(f"Iterations:          {self.n_adaptations}")
        print(f"Converged:           {self.converged}")
        
        if len(self.mse_history) > 0:
            print(f"Current MSE:         {self.mse_history[-1]:.6e}")
            
        print(f"\nTap Coefficients:")
        for i, tap in enumerate(self.taps):
            tap_type = "CURSOR" if i == self.cursor_index else f"Pre-{self.cursor_index-i}" if i < self.cursor_index else f"Post+{i-self.cursor_index}"
            print(f"  w[{i:2d}] ({tap_type:8s}): {tap:+.6f}")
        print("="*60 + "\n")
        
    def plot_convergence(self, figsize: Tuple[int, int] = (14, 10)):
        """Plot adaptation convergence."""
        if len(self.tap_history) < 2:
            print("Not enough adaptation history to plot")
            return None
            
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Convert tap history to array
        tap_history_array = np.array(self.tap_history)
        iterations = np.arange(len(self.tap_history))
        
        # Plot 1: Tap evolution
        ax1 = axes[0, 0]
        for i in range(self.n_taps):
            if i == self.cursor_index:
                ax1.plot(iterations, tap_history_array[:, i], 'r-', 
                        linewidth=2, label=f'Cursor (w[{i}])')
            elif abs(tap_history_array[-1, i]) > 0.01:  # Only plot significant taps
                ax1.plot(iterations, tap_history_array[:, i], alpha=0.7,
                        label=f'w[{i}]')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Tap Coefficient')
        ax1.set_title('FFE Tap Convergence')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8, ncol=2)
        
        # Plot 2: Error evolution
        ax2 = axes[0, 1]
        if len(self.error_history) > 0:
            ax2.plot(np.abs(self.error_history), 'b-', alpha=0.5, linewidth=0.5)
            # Plot running average
            window = min(100, len(self.error_history) // 10)
            if window > 1:
                running_avg = np.convolve(np.abs(self.error_history), 
                                         np.ones(window)/window, mode='valid')
                ax2.plot(range(window//2, len(running_avg) + window//2), 
                        running_avg, 'r-', linewidth=2, label='Running Average')
                ax2.legend()
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('|Error|')
        ax2.set_title('Error Signal Evolution')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: MSE evolution
        ax3 = axes[1, 0]
        if len(self.mse_history) > 0:
            ax3.plot(self.mse_history, 'g-', linewidth=2)
            ax3.axhline(y=self.mse_history[-1], color='red', linestyle='--',
                       alpha=0.5, label=f'Final: {self.mse_history[-1]:.2e}')
            ax3.legend()
        ax3.set_xlabel('Block')
        ax3.set_ylabel('MSE')
        ax3.set_title('Mean Squared Error')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Final tap coefficients
        ax4 = axes[1, 1]
        tap_indices = np.arange(self.n_taps)
        colors = ['red' if i == self.cursor_index else 'blue' for i in range(self.n_taps)]
        ax4.stem(tap_indices, self.taps, basefmt=' ')
        for i, (idx, tap) in enumerate(zip(tap_indices, self.taps)):
            ax4.plot(idx, tap, 'o', color=colors[i], markersize=8)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax4.set_xlabel('Tap Index')
        ax4.set_ylabel('Coefficient Value')
        ax4.set_title(f'Final FFE Taps (Iter: {self.n_adaptations})')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


# Test function
if __name__ == "__main__":
    print("Testing AdaptiveFFE class...\n")
    
    # Create test signal with ISI
    n_symbols = 5000
    pam4_levels = np.array([-1.5, -0.5, 0.5, 1.5]) * 0.5
    
    # Generate random symbols
    tx_symbols = np.random.choice(pam4_levels, size=n_symbols)
    
    # Simple channel with ISI: h = [0.1, 0.3, 1.0, -0.2, -0.1]
    channel = np.array([0.1, 0.3, 1.0, -0.2, -0.1])
    rx_signal = np.convolve(tx_symbols, channel, mode='same')
    
    # Add noise
    rx_signal += np.random.normal(0, 0.05, len(rx_signal))
    
    print(f"Test signal: {n_symbols} symbols")
    print(f"Channel ISI taps: {channel}")
    print(f"RX signal SNR: ~20 dB\n")
    
    # Test each algorithm
    algorithms = ['lms', 'sign_sign_lms', 'rls']
    
    for algo in algorithms:
        print(f"\n{'='*60}")
        print(f"Testing {algo.upper()} Algorithm")
        print('='*60)
        
        # Create FFE
        if algo == 'rls':
            ffe = AdaptiveFFE(n_taps=16, n_precursor=15, n_postcursor=0,
                            algorithm=algo, lambda_rls=0.99, delta_rls=1.0)
        else:
            ffe = AdaptiveFFE(n_taps=16, n_precursor=15, n_postcursor=0,
                            algorithm=algo, mu=0.02)
        
        # Adapt (decision-directed mode)
        output, errors = ffe.process_and_adapt(rx_signal, desired=None, save_history=True)
        
        # Check convergence
        converged = ffe.check_convergence(window=100, threshold=1e-4)
        
        ffe.print_info()
        
        # Plot convergence
        fig = ffe.plot_convergence()
        if fig:
            fig.savefig(f'/mnt/user-data/outputs/ffe_convergence_{algo}.png',
                       dpi=300, bbox_inches='tight')
            print(f"Saved: ffe_convergence_{algo}.png")
    
    print("\n" + "="*60)
    print("AdaptiveFFE testing complete!")
    print("="*60)
