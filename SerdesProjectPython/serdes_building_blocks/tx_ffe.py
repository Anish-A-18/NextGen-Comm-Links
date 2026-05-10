

import random
import math
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple


class TX_FFE:
    """
    Fixed FeedForward Equalizer with 4 taps
    
    Tap structure: [c0, c1, c2, c3]
    Output: y[n] = c0*x[n-3] + c1*x[n-2] + c2*x[n-1] + c3*x[n]
    """
    
    def __init__(self, taps: Optional[List[float]] = None):
        """
        Initialize FFE
        
        Args:
            taps: List of 4 tap coefficients [c0, c1, c2, c3]
                  Default: [0, 1, 0, 0] (pass-through, main tap only)
        """
        if taps is None:
            self.taps = [0,1,0, 0]  # Default: [0, 1, 0, 0]
        else:
            if len(taps) != 4:
                raise ValueError("FFE must have exactly 4 taps")
            self.taps = list(taps)
        
        self.c0, self.c1, self.c2, self.c3 = self.taps
    
    def equalize(self, signal: List[float], samples_per_symbol: int = 1) -> List[float]:
        """
        Apply FFE to signal at symbol level
        
        FFE taps operate on symbols, not samples. For oversampled signals,
        taps look back by symbol periods (samples_per_symbol), not sample periods.
        
        Tap structure: [c0, c1, c2, c3]
        Symbol-level: y[n] = c0*x[n-3] + c1*x[n-2] + c2*x[n-1] + c3*x[n]
        where n is symbol index, not sample index.
        
        Args:
            signal: Input signal samples (oversampled)
            samples_per_symbol: Number of samples per symbol (default: 1 for symbol-rate signal)
            
        Returns:
            Equalized signal
        """
        if len(signal) == 0:
            return signal
        
        if samples_per_symbol < 1:
            samples_per_symbol = 1
        
        equalized = []
        num_symbols = (len(signal) + samples_per_symbol - 1) // samples_per_symbol
        
        # First, compute FFE output at symbol boundaries (sampling instants)
        # FFE taps operate on symbol values at their decision points (symbol start/boundary)
        symbol_equalized_values = []
        
        for sym_idx in range(num_symbols):
            result = 0.0
            
            # Get symbol boundary (start of symbol period) - this is the decision point
            symbol_start = sym_idx * samples_per_symbol
            
            # c0 * x[n-3] (3 symbols ago)
            if sym_idx >= 3:
                lookback_sym_idx = sym_idx - 3
                lookback_start = lookback_sym_idx * samples_per_symbol
                if lookback_start < len(signal):
                    result += self.c0 * signal[lookback_start]
            
            # c1 * x[n-2] (2 symbols ago)
            if sym_idx >= 2:
                lookback_sym_idx = sym_idx - 2
                lookback_start = lookback_sym_idx * samples_per_symbol
                if lookback_start < len(signal):
                    result += self.c1 * signal[lookback_start]
            
            # c2 * x[n-1] (1 symbol ago)
            if sym_idx >= 1:
                lookback_sym_idx = sym_idx - 1
                lookback_start = lookback_sym_idx * samples_per_symbol
                if lookback_start < len(signal):
                    result += self.c2 * signal[lookback_start]
            
            # c3 * x[n] (current symbol, main tap)
            if symbol_start < len(signal):
                result += self.c3 * signal[symbol_start]
            
            symbol_equalized_values.append(result)
        
        # Now interpolate equalized values across all samples in each symbol
        # The equalized value at symbol boundary (pos_in_symbol = 0) represents the decision point
        # We interpolate smoothly across the symbol period
        for i in range(len(signal)):
            current_symbol_idx = i // samples_per_symbol
            pos_in_symbol = (i % samples_per_symbol) / samples_per_symbol
            
            if current_symbol_idx < len(symbol_equalized_values):
                # Get equalized value for current symbol (at symbol boundary)
                eq_at_symbol = symbol_equalized_values[current_symbol_idx]
                
                # At symbol boundary (pos_in_symbol = 0), use current symbol's equalized value
                if pos_in_symbol == 0.0:
                    eq_value = eq_at_symbol
                elif pos_in_symbol > 0 and pos_in_symbol < 0.15 and current_symbol_idx > 0:
                    # Just after symbol start, blend with previous symbol's equalized value
                    prev_eq = symbol_equalized_values[current_symbol_idx - 1]
                    blend = pos_in_symbol / 0.15
                    eq_value = prev_eq * (1.0 - blend) + eq_at_symbol * blend
                elif pos_in_symbol > 0.85 and current_symbol_idx < len(symbol_equalized_values) - 1:
                    # Near end of symbol, blend with next symbol's equalized value
                    next_eq = symbol_equalized_values[current_symbol_idx + 1]
                    blend = (pos_in_symbol - 0.85) / 0.15
                    eq_value = eq_at_symbol * (1.0 - blend) + next_eq * blend
                else:
                    # In the middle of symbol, use the equalized value (represents decision point)
                    eq_value = eq_at_symbol
            else:
                # Use last available equalized value
                eq_value = symbol_equalized_values[-1] if symbol_equalized_values else signal[i]
            
            equalized.append(eq_value)
        
        return equalized
    
    def get_taps(self) -> List[float]:
        """Get tap coefficients"""
        return self.taps.copy()
    
    def get_config(self) -> dict:
        """Get FFE configuration"""
        return {
            'taps': self.taps,
            'c0': self.c0,
            'c1': self.c1,
            'c2': self.c2,
            'c3': self.c3
        }

    def get_frequency_response(self, frequencies: Optional[np.ndarray] = None, 
                              sample_rate: float = 2.048e12) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get frequency response of FFE linear filter.
        
        The FFE is a FIR filter with impulse response h[n] = [c0, c1, c2, c3]
        where taps are delayed by symbol periods.
        
        Frequency response: H(f) = sum(c[k] * exp(-j*2*pi*f*k*T_symbol))
        
        Parameters
        ----------
        frequencies : ndarray, optional
            Frequency points in Hz. If None, uses default 0 to 100 GHz.
        sample_rate : float
            Sample rate in Hz (default: 2.048 THz = 64 GHz * 32 samples/symbol)
            
        Returns
        -------
        freqs : ndarray
            Frequency points in Hz
        H : ndarray
            Complex frequency response
        """
        # Default frequency range: 0 to 100 GHz
        if frequencies is None:
            frequencies = np.linspace(0, 100e9, 2000)
        
        # Symbol rate and period
        symbol_rate = sample_rate / 32  # Assuming 32 samples per symbol
        T_symbol = 1.0 / symbol_rate
        
        # Compute frequency response
        # H(f) = c0*exp(-j*2*pi*f*3*T) + c1*exp(-j*2*pi*f*2*T) + 
        #        c2*exp(-j*2*pi*f*1*T) + c3*exp(-j*2*pi*f*0*T)
        H = np.zeros(len(frequencies), dtype=complex)
        for k, tap in enumerate(self.taps):
            # Tap k corresponds to delay of (3-k) symbols
            # c0 -> 3 symbol delay, c1 -> 2, c2 -> 1, c3 -> 0
            delay_symbols = 3 - k
            H += tap * np.exp(-1j * 2 * np.pi * frequencies * delay_symbols * T_symbol)
        
        return frequencies, H

    def get_impulse_response(self, sample_rate: float = 2.048e12, 
                            duration: float = 5e-9) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get impulse response of FFE in time domain.
        
        The FFE is a linear FIR filter with taps at symbol-spaced intervals.
        Impulse response consists of delta functions at tap positions.
        
        Parameters
        ----------
        sample_rate : float
            Sample rate in Hz (default: 2.048 THz = 64 GHz * 32 samples/symbol)
        duration : float
            Duration of impulse response in seconds (default: 5 ns)
            
        Returns
        -------
        t : ndarray
            Time vector in seconds
        h : ndarray
            Impulse response
        """
        # Symbol rate and period
        symbol_rate = sample_rate / 32  # Assuming 32 samples per symbol
        T_symbol = 1.0 / symbol_rate
        
        # Time vector
        dt = 1.0 / sample_rate
        t = np.arange(0, duration, dt)
        h = np.zeros(len(t))
        
        # Place impulses at tap positions
        # c0 at t=3*T_symbol, c1 at t=2*T_symbol, c2 at t=1*T_symbol, c3 at t=0
        for k, tap in enumerate(self.taps):
            delay_symbols = 3 - k  # c0 at 3 symbols delay, c3 at 0 delay
            delay_time = delay_symbols * T_symbol
            tap_idx = int(delay_time / dt)
            
            if 0 <= tap_idx < len(h):
                h[tap_idx] = tap
        
        return t, h

    def plot_frequency_response(self, frequencies: Optional[np.ndarray] = None,
                               sample_rate: float = 2.048e12):
        """
        Plot frequency response of FFE.
        
        Parameters
        ----------
        frequencies : ndarray, optional
            Frequency points in Hz. If None, uses 0 to 100 GHz.
        sample_rate : float
            Sample rate in Hz (default: 2.048 THz)
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure handle
        """
        freqs, H = self.get_frequency_response(frequencies, sample_rate)
        H_mag_db = 20 * np.log10(np.abs(H) + 1e-12)  # Add small value to avoid log(0)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Magnitude plot
        ax1.plot(freqs / 1e9, H_mag_db, 'b-', linewidth=2)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='0 dB')
        ax1.axvline(x=64, color='g', linestyle='--', alpha=0.5, label='Symbol Rate (64 GHz)')
        ax1.set_xlabel('Frequency [GHz]')
        ax1.set_ylabel('Magnitude [dB]')
        ax1.set_title('TX FFE Frequency Response - Magnitude')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xlim([0, freqs[-1] / 1e9])
        
        # Phase plot
        H_phase_deg = np.angle(H) * 180 / np.pi
        ax2.plot(freqs / 1e9, H_phase_deg, 'r-', linewidth=2)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax2.axvline(x=64, color='g', linestyle='--', alpha=0.5, label='Symbol Rate (64 GHz)')
        ax2.set_xlabel('Frequency [GHz]')
        ax2.set_ylabel('Phase [degrees]')
        ax2.set_title('TX FFE Frequency Response - Phase')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_xlim([0, freqs[-1] / 1e9])
        
        plt.tight_layout()
        return fig

    def plot_impulse_response(self, sample_rate: float = 2.048e12, 
                             duration: float = 5e-9):
        """
        Plot impulse response of FFE.
        
        Parameters
        ----------
        sample_rate : float
            Sample rate in Hz (default: 2.048 THz)
        duration : float
            Duration in seconds (default: 5 ns)
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure handle
        """
        t, h = self.get_impulse_response(sample_rate, duration)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot impulse response
        ax.stem(t * 1e9, h, basefmt=' ', linefmt='b-', markerfmt='bo')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Add tap labels
        symbol_rate = sample_rate / 32
        T_symbol = 1.0 / symbol_rate
        tap_labels = ['c0\n(pre-3)', 'c1\n(pre-2)', 'c2\n(pre-1)', 'c3\n(cursor)']
        for k, (tap, label) in enumerate(zip(self.taps, tap_labels)):
            if tap != 0:
                delay_symbols = 3 - k
                delay_time = delay_symbols * T_symbol * 1e9  # ns
                ax.text(delay_time, tap + 0.05 * np.sign(tap), 
                       f'{label}\n{tap:+.3f}',
                       ha='center', va='bottom' if tap > 0 else 'top')
        
        ax.set_xlabel('Time [ns]')
        ax.set_ylabel('Amplitude')
        ax.set_title('TX FFE Impulse Response (Symbol-Spaced FIR Filter)')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, duration * 1e9])
        
        plt.tight_layout()
        return fig