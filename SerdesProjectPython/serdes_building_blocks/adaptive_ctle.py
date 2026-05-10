"""
Adaptive CTLE for PCIe 7.0 SerDes - Exhaustive Search Algorithm
================================================================

Features:
- 16 hardware-optimized CTLE configurations
- Exhaustive search adaptation during training sequence
- V_P4T threshold calculation for each config
- Optimal sampling point selection (32 sample offsets)
- SNR-based configuration selection
- Symbol error rate (SER) measurement
- Eye diagram analysis (height, width)

Author: Anish Anand
Date: November 6, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sp_signal
from typing import Tuple, Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')


class CTLEConfig:
    """Single CTLE configuration with updated optimized parameters."""
    
    # UPDATED Fixed parameters for ALL configurations (from optimization)
    Z1_FIXED = 0.5e9       # 500 MHz (USER SPECIFIED)
    Z3_FIXED = 30e9        # 30 GHz (OPTIMIZED)
    P1_FIXED = 0.65e9      # 650 MHz (USER SPECIFIED, 1.3 × z1)
    P2_FIXED = 31e9        # 31 GHz (OPTIMIZED)
    P3_FIXED = 44e9        # 44 GHz (OPTIMIZED)
    P4_FIXED = 55e9        # 55 GHz (OPTIMIZED)
    P5_FIXED = 63e9        # 63 GHz (OPTIMIZED)
    P6_FIXED = 65e9        # 65 GHz (OPTIMIZED)
    
    # UPDATED Variable z2 values for each configuration (from optimization)
    Z2_VALUES = {
        0:  9.870e9,   # Config 0:  DC =  0.0 dB
        1:  8.710e9,   # Config 1:  DC = -1.0 dB
        2:  7.700e9,   # Config 2:  DC = -2.0 dB
        3:  6.830e9,   # Config 3:  DC = -3.0 dB
        4:  6.060e9,   # Config 4:  DC = -4.0 dB
        5:  5.380e9,   # Config 5:  DC = -5.0 dB
        6:  4.780e9,   # Config 6:  DC = -6.0 dB
        7:  4.250e9,   # Config 7:  DC = -7.0 dB
        8:  3.790e9,   # Config 8:  DC = -8.0 dB
        9:  3.370e9,   # Config 9:  DC = -9.0 dB
        10: 3.000e9,   # Config 10: DC = -10.0 dB
        11: 2.670e9,   # Config 11: DC = -11.0 dB
        12: 2.380e9,   # Config 12: DC = -12.0 dB
        13: 2.120e9,   # Config 13: DC = -13.0 dB
        14: 1.890e9,   # Config 14: DC = -14.0 dB
        15: 1.680e9,   # Config 15: DC = -15.0 dB
    }
    
    def __init__(self, config_id: int):
        """Initialize CTLE configuration."""
        if config_id not in range(16):
            raise ValueError(f"Config ID must be 0-15, got {config_id}")
        
        self.config_id = config_id
        self.dc_gain_db = -config_id
        self.boost_db = 8.0 - self.dc_gain_db
        
        # Get variable z2 for this config
        self.z2 = self.Z2_VALUES[config_id]
        
    @property
    def z1(self):
        return self.Z1_FIXED
    
    @property
    def z3(self):
        return self.Z3_FIXED
    
    @property
    def p1(self):
        return self.P1_FIXED
    
    @property
    def p2(self):
        return self.P2_FIXED
    
    @property
    def p3(self):
        return self.P3_FIXED
    
    @property
    def p4(self):
        return self.P4_FIXED
    
    @property
    def p5(self):
        return self.P5_FIXED
    
    @property
    def p6(self):
        return self.P6_FIXED
    
    def get_transfer_function(self, frequencies: np.ndarray) -> np.ndarray:
        """
        Calculate CTLE frequency response H(f) in linear scale.
        
        Parameters
        ----------
        frequencies : ndarray
            Frequency points in Hz
            
        Returns
        -------
        H : ndarray (complex)
            CTLE transfer function (complex, linear scale)
        """
        # Angular frequencies
        w = 2 * np.pi * frequencies
        s = 1j * w
        
        wz1 = 2 * np.pi * self.z1
        wz2 = 2 * np.pi * self.z2
        wz3 = 2 * np.pi * self.z3
        wp1 = 2 * np.pi * self.p1
        wp2 = 2 * np.pi * self.p2
        wp3 = 2 * np.pi * self.p3
        wp4 = 2 * np.pi * self.p4
        wp5 = 2 * np.pi * self.p5
        wp6 = 2 * np.pi * self.p6
        
        # Normalization constant
        K = (wp1 * wp2 * wp3 * wp4 * wp5 * wp6) / (wz1 * wz2 * wz3)
        
        # DC gain (linear scale)
        Adc = 10**(self.dc_gain_db / 20)
        
        # Transfer function
        numerator = (s + wz1) * (s + wz2) * (s + wz3)
        denominator = (s + wp1) * (s + wp2) * (s + wp3) * (s + wp4) * (s + wp5) * (s + wp6)
        
        H = K * Adc * (numerator / denominator)
        
        return H
    
    def get_transfer_function_db(self, frequencies: np.ndarray) -> np.ndarray:
        """Get transfer function in dB."""
        H = self.get_transfer_function(frequencies)
        return 20 * np.log10(np.abs(H))
    
    def get_impulse_response(self, frequencies: np.ndarray, t_sample: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute CTLE impulse response using IFFT (same method as SerdesChannel).
        
        Parameters
        ----------
        frequencies : ndarray
            Frequency points in Hz
        t_sample : float
            Time sampling period in seconds
            
        Returns
        -------
        h_impulse : ndarray
            Impulse response
        t_impulse : ndarray
            Time vector
        """
        H = self.get_transfer_function(frequencies)
        
        # Create symmetric spectrum for real-valued time signal (same as SerdesChannel)
        Hd = np.concatenate((H, np.conj(np.flip(H[1:H.size-1]))))
        h_impulse = np.real(np.fft.ifft(Hd))
        
        # Time vector
        t_impulse = np.linspace(0, 1/frequencies[1], h_impulse.size + 1)
        t_impulse = t_impulse[0:-1]
        
        return h_impulse, t_impulse
    
    def get_peak_info(self) -> Tuple[float, float]:
        """Get peak frequency and gain."""
        freqs = np.linspace(20e9, 40e9, 1000)
        response = self.get_transfer_function_db(freqs)
        peak_idx = np.argmax(response)
        return freqs[peak_idx], response[peak_idx]
    
    def __str__(self):
        """String representation."""
        peak_freq, peak_gain = self.get_peak_info()
        return (f"Config {self.config_id}: DC={self.dc_gain_db:+.0f}dB, "
                f"z2={self.z2/1e9:.3f}GHz, Peak={peak_gain:+.2f}dB@{peak_freq/1e9:.2f}GHz")


class AdaptiveCTLE:
    """
    Adaptive CTLE with exhaustive search algorithm for PCIe 7.0 SerDes.
    
    Performs training-based adaptation to select optimal CTLE configuration
    and sampling point.
    """
    
    def __init__(self, 
                 symbol_rate: float = 64e9,
                 samples_per_symbol: int = 32,
                 n_configs: int = 16,
                 f_max: Optional[float] = None,
                 n_levels: int = 6,
                 alpha: float = 0.87):
        """
        Initialize Adaptive CTLE.
        
        Parameters
        ----------
        symbol_rate : float
            Symbol rate in Hz (default: 64 GHz for PCIe 7.0)
        samples_per_symbol : int
            Number of samples per symbol (default: 32)
        n_configs : int
            Number of CTLE configurations (default: 16)
        f_max : float, optional
            Maximum frequency for CTLE modeling (default: symbol_rate * samples_per_symbol / 2)
        n_levels : int
            Number of PAM levels (default: 6 for PAM6)
        alpha : float
            Threshold scaling factor (default: 0.87)
        """
        self.symbol_rate = symbol_rate
        self.samples_per_symbol = samples_per_symbol
        self.n_configs = n_configs
        self.n_levels = n_levels
        self.alpha = alpha
        self.pam_alpha = 0.87  # Threshold scaling factor for slicing
        
        # Frequency vector setup
        if f_max is None:
            self.f_max = symbol_rate * samples_per_symbol / 2
        else:
            self.f_max = f_max
            
        k = 14  # Same as SerdesChannel
        self.frequencies = np.linspace(0, self.f_max, 2**k + 1)
        self.t_sample = 1 / (2 * self.f_max)
        
        # Create all CTLE configurations
        self.configs = [CTLEConfig(i) for i in range(n_configs)]
        
        # Precompute impulse responses for all configs
        self.impulse_responses = {}
        for config in self.configs:
            h_impulse, t_impulse = config.get_impulse_response(self.frequencies, self.t_sample)
            self.impulse_responses[config.config_id] = h_impulse
            
        # Adaptation results (will be populated after adapt())
        self.adaptation_results = {}
        self.best_config_id = None
        self.best_sampling_offset = None
        self.best_vp4t = None
        self.best_snr_db = None
        self.best_expected_levels = None
        
        # Store signals for analysis
        self.rx_signal_raw = None
        self.rx_signal_ctle = None
        self.tx_symbols = None

    def slice_pam(self, values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        """
        Slice values to PAM symbols based on number of levels.
        
        Parameters
        ----------
        values : ndarray
            Signal values to slice
        thresholds : ndarray
            Decision thresholds (n_levels - 1 thresholds)
            
        Returns
        -------
        symbols : ndarray
            Sliced symbols (0 to n_levels-1)
        """
        symbols = np.zeros(len(values), dtype=int)
        thresholds = thresholds * self.pam_alpha
        
        if self.n_levels == 4:
            # PAM4: 3 thresholds
            symbols[values < thresholds[0]] = 0
            symbols[(values >= thresholds[0]) & (values < thresholds[1])] = 1
            symbols[(values >= thresholds[1]) & (values < thresholds[2])] = 2
            symbols[values >= thresholds[2]] = 3
        elif self.n_levels == 6:
            # PAM6: 5 thresholds
            symbols[values < thresholds[0]] = 0
            symbols[(values >= thresholds[0]) & (values < thresholds[1])] = 1
            symbols[(values >= thresholds[1]) & (values < thresholds[2])] = 2
            symbols[(values >= thresholds[2]) & (values < thresholds[3])] = 3
            symbols[(values >= thresholds[3]) & (values < thresholds[4])] = 4
            symbols[values >= thresholds[4]] = 5
        elif self.n_levels == 8:
            # PAM8: 7 thresholds
            symbols[values < thresholds[0]] = 0
            symbols[(values >= thresholds[0]) & (values < thresholds[1])] = 1
            symbols[(values >= thresholds[1]) & (values < thresholds[2])] = 2
            symbols[(values >= thresholds[2]) & (values < thresholds[3])] = 3
            symbols[(values >= thresholds[3]) & (values < thresholds[4])] = 4
            symbols[(values >= thresholds[4]) & (values < thresholds[5])] = 5
            symbols[(values >= thresholds[5]) & (values < thresholds[6])] = 6
            symbols[values >= thresholds[6]] = 7
        else:
            raise ValueError(f"Unsupported n_levels: {self.n_levels}. Supported: 4, 6, 8")
        
        return symbols

    def align_sequences(self,detected_symbols: np.ndarray, tx_symbols: np.ndarray, max_search: int = 500) -> Tuple[int, float]:
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

    def apply_ctle(self, signal: np.ndarray, config_id: int) -> np.ndarray:
        """
        Apply CTLE filtering to signal.
        
        Parameters
        ----------
        signal : ndarray
            Input signal (after channel)
        config_id : int
            CTLE configuration ID (0-15)
            
        Returns
        -------
        signal_ctle : ndarray
            Signal after CTLE equalization (same length as input)
        """
        h_impulse = self.impulse_responses[config_id]
        # Use full convolution
        signal_ctle_full = np.convolve(signal, h_impulse, mode='full')
        # Extract the causal part (corresponding to the original signal length)
        # The first len(signal) samples correspond to the filtered output
        signal_ctle = signal_ctle_full[:len(signal)]
        return signal_ctle
    
    def calculate_vpnt(self, signal_segment: np.ndarray) -> float:
        """
        Calculate V_PNT (PAM-N Threshold) from signal segment.
        
        V_PNT = sqrt(sum(signal^2) / N)
        
        For PAM4: V_P4T
        For PAM6: V_P6T
        
        Parameters
        ----------
        signal_segment : ndarray
            Signal segment (symbols × samples_per_symbol)
            
        Returns
        -------
        vpnt : float
            PAM-N threshold voltage
        """
        squared_sum = np.sum(signal_segment**2)
        vpnt = np.sqrt(squared_sum / len(signal_segment))
        return vpnt
    
    def bin_samples_by_offset(self, signal_segment: np.ndarray, 
                             n_symbols: int = 100) -> np.ndarray:
        """
        Bin samples by offset within symbol period.
        
        Creates a 32 × 100 array where each row contains samples at the same
        offset within the symbol period.
        
        Parameters
        ----------
        signal_segment : ndarray
            Signal segment (n_symbols × samples_per_symbol)
        n_symbols : int
            Number of symbols to bin (default: 100)
            
        Returns
        -------
        binned_samples : ndarray
            Array of shape (samples_per_symbol, n_symbols)
        """
        signal_segment = signal_segment[:n_symbols * self.samples_per_symbol]
        reshaped = signal_segment.reshape(n_symbols, self.samples_per_symbol)
        # Transpose so each row is a specific offset across all symbols
        binned_samples = reshaped.T
        return binned_samples
    
    def symbols_from_signal(self, signal_samples: np.ndarray, vpnt: float) -> np.ndarray:
        """
        Convert signal samples to symbol decisions based on PAM thresholds.
        
        PAM4 Thresholds (vp4t):
        - symbol 0: < -vp4t         (level -3)
        - symbol 1: -vp4t to 0      (level -1)
        - symbol 2: 0 to +vp4t      (level +1)
        - symbol 3: >= +vp4t        (level +3)
        
        PAM6 Thresholds (vp6t/3):
        Thresholds: (-4, -2, 0, 2, 4) * vp6t/3
        - symbol 0: < -4*vp6t/3     (level -5*vp6t/3)
        - symbol 1: -4*vp6t/3 to -2*vp6t/3  (level -3*vp6t/3)
        - symbol 2: -2*vp6t/3 to 0  (level -1*vp6t/3)
        - symbol 3: 0 to +2*vp6t/3  (level +1*vp6t/3)
        - symbol 4: +2*vp6t/3 to +4*vp6t/3  (level +3*vp6t/3)
        - symbol 5: >= +4*vp6t/3    (level +5*vp6t/3)
        
        PAM8 Thresholds (vp8t/4):
        Thresholds: (-6, -4, -2, 0, 2, 4, 6) * vp8t/4
        - symbol 0: < -6*vp8t/4     (level -7*vp8t/4)
        - symbol 1: -6*vp8t/4 to -4*vp8t/4  (level -5*vp8t/4)
        - symbol 2: -4*vp8t/4 to -2*vp8t/4  (level -3*vp8t/4)
        - symbol 3: -2*vp8t/4 to 0  (level -1*vp8t/4)
        - symbol 4: 0 to +2*vp8t/4  (level +1*vp8t/4)
        - symbol 5: +2*vp8t/4 to +4*vp8t/4  (level +3*vp8t/4)
        - symbol 6: +4*vp8t/4 to +6*vp8t/4  (level +5*vp8t/4)
        - symbol 7: >= +6*vp8t/4    (level +7*vp8t/4)
        
        Parameters
        ----------
        signal_samples : ndarray
            Signal samples to decode
        vpnt : float
            PAM threshold voltage (V_P4T, V_P6T, or V_P8T)
            
        Returns
        -------
        symbols : ndarray
            Decoded symbols (0 to n_levels-1)
        """
        symbols = np.zeros_like(signal_samples, dtype=int)
        
        if self.n_levels == 4:
            # PAM4 slicing with pam_alpha scaling
            th1 = -vpnt * self.pam_alpha
            th2 = 0
            th3 = vpnt * self.pam_alpha
            
            symbols[signal_samples < th1] = 0
            symbols[(signal_samples >= th1) & (signal_samples < th2)] = 1
            symbols[(signal_samples >= th2) & (signal_samples < th3)] = 2
            symbols[signal_samples >= th3] = 3
        elif self.n_levels == 6:
            # PAM6 slicing: thresholds at (-4, -2, 0, 2, 4) * vp6t/3 * pam_alpha
            th1 = -4 * vpnt / 3 * self.pam_alpha
            th2 = -2 * vpnt / 3 * self.pam_alpha
            th3 = 0
            th4 = 2 * vpnt / 3 * self.pam_alpha
            th5 = 4 * vpnt / 3 * self.pam_alpha
            
            symbols[signal_samples < th1] = 0
            symbols[(signal_samples >= th1) & (signal_samples < th2)] = 1
            symbols[(signal_samples >= th2) & (signal_samples < th3)] = 2
            symbols[(signal_samples >= th3) & (signal_samples < th4)] = 3
            symbols[(signal_samples >= th4) & (signal_samples < th5)] = 4
            symbols[signal_samples >= th5] = 5
        elif self.n_levels == 8:
            # PAM8 slicing: thresholds at (-6, -4, -2, 0, 2, 4, 6) * vp8t/4 * pam_alpha
            th1 = -6 * vpnt / 4 * self.pam_alpha
            th2 = -4 * vpnt / 4 * self.pam_alpha
            th3 = -2 * vpnt / 4 * self.pam_alpha
            th4 = 0
            th5 = 2 * vpnt / 4 * self.pam_alpha
            th6 = 4 * vpnt / 4 * self.pam_alpha
            th7 = 6 * vpnt / 4 * self.pam_alpha
            
            symbols[signal_samples < th1] = 0
            symbols[(signal_samples >= th1) & (signal_samples < th2)] = 1
            symbols[(signal_samples >= th2) & (signal_samples < th3)] = 2
            symbols[(signal_samples >= th3) & (signal_samples < th4)] = 3
            symbols[(signal_samples >= th4) & (signal_samples < th5)] = 4
            symbols[(signal_samples >= th5) & (signal_samples < th6)] = 5
            symbols[(signal_samples >= th6) & (signal_samples < th7)] = 6
            symbols[signal_samples >= th7] = 7
        else:
            raise ValueError(f"Unsupported n_levels: {self.n_levels}. Supported: 4, 6, 8")
        
        return symbols
    
    def expected_signal_from_symbols(self, symbols: np.ndarray, vpnt: float) -> np.ndarray:
        """
        Convert symbols to expected signal levels.
        
        PAM4 Mapping (vp4t):
        - symbol 0 → -1.5 * vp4t
        - symbol 1 → -0.5 * vp4t
        - symbol 2 → +0.5 * vp4t
        - symbol 3 → +1.5 * vp4t
        
        PAM6 Mapping (vp6t/3):
        Levels: (-5, -3, -1, 1, 3, 5) * vp6t/3
        - symbol 0 → -5 * vp6t/3
        - symbol 1 → -3 * vp6t/3
        - symbol 2 → -1 * vp6t/3
        - symbol 3 → +1 * vp6t/3
        - symbol 4 → +3 * vp6t/3
        - symbol 5 → +5 * vp6t/3
        
        PAM8 Mapping (vp8t/4):
        Levels: (-7, -5, -3, -1, 1, 3, 5, 7) * vp8t/4
        - symbol 0 → -7 * vp8t/4
        - symbol 1 → -5 * vp8t/4
        - symbol 2 → -3 * vp8t/4
        - symbol 3 → -1 * vp8t/4
        - symbol 4 → +1 * vp8t/4
        - symbol 5 → +3 * vp8t/4
        - symbol 6 → +5 * vp8t/4
        - symbol 7 → +7 * vp8t/4
        
        Parameters
        ----------
        symbols : ndarray
            Symbol array (values 0 to n_levels-1)
        vpnt : float
            PAM threshold voltage
            
        Returns
        -------
        expected_signal : ndarray
            Expected signal levels
        """
        if self.n_levels == 4:
            # PAM4 levels: (-3, -1, 1, 3) * vp4t/2 * pam_alpha
            level_map = {
                0: -1.5 * vpnt * self.pam_alpha,
                1: -0.5 * vpnt * self.pam_alpha,
                2: +0.5 * vpnt * self.pam_alpha,
                3: +1.5 * vpnt * self.pam_alpha
            }
        elif self.n_levels == 6:
            # PAM6 levels: (-5, -3, -1, 1, 3, 5) * vp6t/3 * pam_alpha
            level_map = {
                0: -5 * vpnt / 3 * self.pam_alpha,
                1: -3 * vpnt / 3 * self.pam_alpha,
                2: -1 * vpnt / 3 * self.pam_alpha,
                3: +1 * vpnt / 3 * self.pam_alpha,
                4: +3 * vpnt / 3 * self.pam_alpha,
                5: +5 * vpnt / 3 * self.pam_alpha
            }
        elif self.n_levels == 8:
            # PAM8 levels: (-7, -5, -3, -1, 1, 3, 5, 7) * vp8t/4 * pam_alpha
            level_map = {
                0: -7 * vpnt / 4 * self.pam_alpha,
                1: -5 * vpnt / 4 * self.pam_alpha,
                2: -3 * vpnt / 4 * self.pam_alpha,
                3: -1 * vpnt / 4 * self.pam_alpha,
                4: +1 * vpnt / 4 * self.pam_alpha,
                5: +3 * vpnt / 4 * self.pam_alpha,
                6: +5 * vpnt / 4 * self.pam_alpha,
                7: +7 * vpnt / 4 * self.pam_alpha
            }
        else:
            raise ValueError(f"Unsupported n_levels: {self.n_levels}. Supported: 4, 6, 8")
        
        expected_signal = np.array([level_map[s] for s in symbols])
        return expected_signal
    
    def find_optimal_sampling_offset(self, binned_samples: np.ndarray, 
                                    vpnt: float) -> Tuple[int, float, float]:
        """
        Find optimal sampling offset by minimizing mean squared error.
        
        Parameters
        ----------
        binned_samples : ndarray
            Binned samples (samples_per_symbol × n_symbols)
        vpnt : float
            PAM threshold voltage (V_P4T or V_P6T)
            
        Returns
        -------
        best_offset : int
            Optimal sampling offset (0 to samples_per_symbol-1)
        best_mse : float
            Minimum mean squared error at optimal offset
        best_snr_db : float
            SNR in dB at optimal offset
        """
        n_offsets = binned_samples.shape[0]
        mse_per_offset = np.zeros(n_offsets)
        
        for offset in range(n_offsets):
            samples = binned_samples[offset, :]
            
            # Decode symbols
            symbols = self.symbols_from_signal(samples, vpnt)
            
            # Get expected signal
            expected = self.expected_signal_from_symbols(symbols, vpnt)
            
            # Calculate MSE
            error = samples - expected
            mse_per_offset[offset] = np.mean(error**2)
        
        # Find best offset
        best_offset = np.argmin(mse_per_offset)
        best_mse = mse_per_offset[best_offset]
        
        # Calculate SNR: signal_power / noise_power
        signal_power = vpnt**2
        noise_power = best_mse
        
        if noise_power > 0:
            snr_db = 10 * np.log10(signal_power / noise_power)
        else:
            snr_db = np.inf
        
        return best_offset, best_mse, snr_db
    
    def adapt(self, rx_signal: np.ndarray, 
             tx_symbols: Optional[np.ndarray] = None,
             verbose: bool = True) -> Dict:
        """
        Perform adaptive CTLE configuration selection.
        
        Adaptation sequence structure (per config):
        - Symbols 0-49: Transition (skip)
        - Symbols 50-149: Calculate V_P4T
        - Symbols 150-249: Find optimal sampling offset and SNR
        
        Total: 16 configs × 250 symbols = 4000 symbols
        
        Parameters
        ----------
        rx_signal : ndarray
            Received signal after channel (before CTLE)
        tx_symbols : ndarray, optional
            Transmitted symbols for SER calculation (symbol values 0,1,2,3)
        verbose : bool
            Print progress information
            
        Returns
        -------
        results : dict
            Adaptation results with best config, offset, SNR, etc.
        """
        self.rx_signal_raw = rx_signal
        self.tx_symbols = tx_symbols
        
        symbols_per_config = 2000
        transition_symbols = 300
        vp4t_calc_symbols = 500
        test_symbols = 1200
        
        total_symbols_needed = self.n_configs * symbols_per_config
        total_samples_needed = total_symbols_needed * self.samples_per_symbol
        
        if len(rx_signal) < total_samples_needed:
            raise ValueError(f"RX signal too short. Need {total_samples_needed} samples "
                           f"({total_symbols_needed} symbols), got {len(rx_signal)}")
        
        if verbose:
            print("\n" + "="*70)
            print("Adaptive CTLE - Exhaustive Search")
            print("="*70)
            print(f"Symbol rate: {self.symbol_rate/1e9:.1f} Gbaud")
            print(f"Samples per symbol: {self.samples_per_symbol}")
            print(f"Testing {self.n_configs} configurations...")
            print(f"Symbols per config: {symbols_per_config}")
            print(f"  - Transition: {transition_symbols} symbols (skipped)")
            print(f"  - V_P4T calc: {vp4t_calc_symbols} symbols")
            print(f"  - Optimization: {test_symbols} symbols")
            print("="*70 + "\n")
        
        # Buffer to capture the adaptation-phase CTLE output signal
        adaptation_output_signal = np.zeros(total_samples_needed, dtype=rx_signal.dtype)
        
        # Test each configuration
        for config_id in range(self.n_configs):
            if verbose:
                print(f"Testing Config {config_id}...")
            
            # Extract signal segment for this config
            start_symbol = config_id * symbols_per_config
            start_sample = start_symbol * self.samples_per_symbol
            end_sample = start_sample + symbols_per_config * self.samples_per_symbol
            
            rx_segment = rx_signal[start_sample:end_sample]
            
            # Apply CTLE
            rx_segment_ctle = self.apply_ctle(rx_segment, config_id)
            
            # Store the equalized segment in the adaptation buffer
            adaptation_output_signal[start_sample:end_sample] = rx_segment_ctle
            
            # Skip transition symbols
            trans_end_sample = transition_symbols * self.samples_per_symbol
            
            # Calculate V_PNT (V_P4T or V_P6T) from training symbols
            vpnt_start = trans_end_sample
            vpnt_end = vpnt_start + vp4t_calc_symbols * self.samples_per_symbol
            vpnt_segment = rx_segment_ctle[vpnt_start:vpnt_end]
            vpnt = self.calculate_vpnt(vpnt_segment)
            
            # Find optimal sampling offset from test symbols
            test_start = vpnt_end
            test_end = test_start + test_symbols * self.samples_per_symbol
            test_segment = rx_segment_ctle[test_start:test_end]
            
            binned = self.bin_samples_by_offset(test_segment, test_symbols)
            best_offset, best_mse, snr_db = self.find_optimal_sampling_offset(binned, vpnt)

            test_segment_symbols = test_segment[best_offset::self.samples_per_symbol]
            test_segment_symbols_detected = self.symbols_from_signal(test_segment_symbols, vpnt)
            tx_symbols_segment = tx_symbols[start_symbol+transition_symbols+vp4t_calc_symbols:start_symbol +transition_symbols+vp4t_calc_symbols+symbols_per_config]
            # Align sequences
            delay, correlation = self.align_sequences(test_segment_symbols_detected, tx_symbols_segment)

            # Apply alignment
            if delay < 0:
                # Shift TX symbols earlier (detected symbols are delayed)
                tx_symbols_aligned = tx_symbols_segment[-delay:]
                test_segment_symbols_aligned = test_segment_symbols[:len(tx_symbols_aligned)]
                test_segment_symbols_detected_aligned = test_segment_symbols_detected[:len(tx_symbols_aligned)]
            else:
                # Shift TX symbols later (detected symbols are early)
                tx_symbols_aligned = tx_symbols_segment[:len(tx_symbols_segment) - delay]
                test_segment_symbols_aligned = test_segment_symbols[delay:delay + len(tx_symbols_aligned)]
                test_segment_symbols_detected_aligned = test_segment_symbols_detected[delay:delay + len(tx_symbols_aligned)]
           
            min_length = min(len(tx_symbols_aligned), len(test_segment_symbols_detected_aligned))
            tx_symbols_aligned = tx_symbols_aligned[:min_length]
            test_segment_symbols_detected_aligned = test_segment_symbols_detected_aligned[:min_length]

            # Calculate SER
            errors = np.sum(tx_symbols_aligned != test_segment_symbols_detected_aligned)
            ser = errors / len(tx_symbols_aligned) if len(tx_symbols_aligned) > 0 else 0
            
            if verbose:
                print(f"Config {config_id}: Delay: {delay}, Correlation: {correlation} , SER: {ser}")

            # Store results
            self.adaptation_results[config_id] = {
                'config_id': config_id,
                'vpnt': vpnt,  # V_P4T or V_P6T
                'best_offset': best_offset,
                'mse': best_mse,
                'snr_db': snr_db,
                'rx_segment_ctle': rx_segment_ctle,
                'ser': ser,
                'delay': delay,
                'correlation': correlation,
                'tx_symbols_aligned': tx_symbols_aligned,
                'test_segment_symbols_aligned': test_segment_symbols_aligned,
                'test_segment_symbols_detected_aligned': test_segment_symbols_detected_aligned
            }
            
            if verbose:
                vpnt_name = f"V_P{self.n_levels}T"
                print(f"  {vpnt_name} = {vpnt:.4f} V")
                print(f"  Best offset = {best_offset}")
                print(f"  SNR = {snr_db:.2f} dB")
                print(f"  MSE = {best_mse:.6e}\n")
        
        # Select best configuration (highest SNR)
        best_config_id = min(self.adaptation_results.keys(), 
                            key=lambda k: self.adaptation_results[k]['ser'])
        
        self.best_config_id = best_config_id
        self.best_sampling_offset = self.adaptation_results[best_config_id]['best_offset']
        self.best_vpnt = self.adaptation_results[best_config_id]['vpnt']
        self.best_snr_db = self.adaptation_results[best_config_id]['snr_db']
        self.best_ser = self.adaptation_results[best_config_id]['ser']

        # Calculate expected levels for best config
        if self.n_levels == 4:
            self.best_expected_levels = {
                0: -1.5 * self.best_vpnt * self.pam_alpha,
                1: -0.5 * self.best_vpnt * self.pam_alpha,
                2: +0.5 * self.best_vpnt * self.pam_alpha,
                3: +1.5 * self.best_vpnt * self.pam_alpha
            }
        elif self.n_levels == 6:
            # PAM6 levels: (-5, -3, -1, 1, 3, 5) * vp6t/3 * pam_alpha
            self.best_expected_levels = {
                0: -5 * self.best_vpnt / 3 * self.pam_alpha,
                1: -3 * self.best_vpnt / 3 * self.pam_alpha,
                2: -1 * self.best_vpnt / 3 * self.pam_alpha,
                3: +1 * self.best_vpnt / 3 * self.pam_alpha,
                4: +3 * self.best_vpnt / 3 * self.pam_alpha,
                5: +5 * self.best_vpnt / 3 * self.pam_alpha
            }
        elif self.n_levels == 8:
            # PAM8 levels: (-7, -5, -3, -1, 1, 3, 5, 7) * vp8t/4 * pam_alpha
            self.best_expected_levels = {
                0: -7 * self.best_vpnt / 4 * self.pam_alpha,
                1: -5 * self.best_vpnt / 4 * self.pam_alpha,
                2: -3 * self.best_vpnt / 4 * self.pam_alpha,
                3: -1 * self.best_vpnt / 4 * self.pam_alpha,
                4: +1 * self.best_vpnt / 4 * self.pam_alpha,
                5: +3 * self.best_vpnt / 4 * self.pam_alpha,
                6: +5 * self.best_vpnt / 4 * self.pam_alpha,
                7: +7 * self.best_vpnt / 4 * self.pam_alpha
            }
        else:
            raise ValueError(f"Unsupported n_levels: {self.n_levels}. Supported: 4, 6, 8")
        
        # Expose the adaptation-phase CTLE output
        self.adaptation_output_signal = adaptation_output_signal
        
        if verbose:
            vpnt_name = f"V_P{self.n_levels}T"
            print("="*70)
            print(f"BEST CONFIGURATION: Config {self.best_config_id}")
            print(f"  SNR: {self.best_snr_db:.2f} dB")
            print(f"  Sampling Offset: {self.best_sampling_offset}")
            print(f"  {vpnt_name}: {self.best_vpnt:.4f} V")
            print(f"  SER: {self.best_ser:.6e}")
            print(f"  Expected Levels:")
            for sym, level in self.best_expected_levels.items():
                print(f"    Symbol {sym}: {level:+.4f} V")
            print("="*70 + "\n")
        
        return {
            'best_config_id': self.best_config_id,
            'best_offset': self.best_sampling_offset,
            'best_vpnt': self.best_vpnt,
            'best_snr_db': self.best_snr_db,
            'best_ser': self.best_ser,
            'expected_levels': self.best_expected_levels,
            'all_results': self.adaptation_results,
            'adaptation_output_signal': adaptation_output_signal
        }
    def apply_ctle_with_best_config(self, rx_signal: np.ndarray) -> np.ndarray:
        """
        Apply best CTLE configuration to the RX signal.
        
        Parameters
        ----------
        rx_signal : ndarray
            RX signal data to apply CTLE to
        """
        if self.best_config_id is None:
            raise RuntimeError("Must run adapt() before applying best CTLE configuration")
        
        return self.apply_ctle(rx_signal, self.best_config_id)
    
  
    def process_data_sequence(self, rx_signal_data: np.ndarray, 
                             tx_symbols_data: Optional[np.ndarray] = None) -> Dict:
        """
        Process data sequence with best CTLE configuration.
        
        Parameters
        ----------
        rx_signal_data : ndarray
            Data portion of RX signal (after adaptation sequence)
        tx_symbols_data : ndarray, optional
            Transmitted symbols for SER calculation
            
        Returns
        -------
        results : dict
            Processing results with metrics
        """
        if self.best_config_id is None:
            raise RuntimeError("Must run adapt() before processing data sequence")
        
        # Apply best CTLE config
        rx_ctle = self.apply_ctle(rx_signal_data, self.best_config_id)
        self.rx_signal_ctle = rx_ctle
        
        # Sample at best offset
        sampled_signal = rx_ctle[self.best_sampling_offset::self.samples_per_symbol]
        
        # Decode symbols
        rx_symbols = self.symbols_from_signal(sampled_signal, self.best_vpnt)
        
        # Calculate metrics
        results = {
            'rx_symbols': rx_symbols,
            'sampled_signal': sampled_signal,
        }
        
        # If TX symbols provided, calculate SER
        if tx_symbols_data is not None:
            # Align TX and RX symbols (find delay via cross-correlation)
            delay = self._estimate_delay(tx_symbols_data, rx_symbols)
            
            # Align sequences
            if delay >= 0:
                tx_aligned = tx_symbols_data[delay:]
                rx_aligned = rx_symbols[:len(tx_aligned)]
            else:
                rx_aligned = rx_symbols[-delay:]
                tx_aligned = tx_symbols_data[:len(rx_aligned)]
            
            # Calculate SER
            n_compared = min(len(tx_aligned), len(rx_aligned))
            errors = np.sum(tx_aligned[:n_compared] != rx_aligned[:n_compared])
            ser = errors / n_compared
            
            results['ser'] = ser
            results['n_errors'] = errors
            results['n_symbols'] = n_compared
            results['delay_symbols'] = delay
        
        return results
    
    def _estimate_delay(self, tx_symbols: np.ndarray, rx_symbols: np.ndarray) -> int:
        """
        Estimate delay between TX and RX symbol sequences using cross-correlation.
        
        Parameters
        ----------
        tx_symbols : ndarray
            Transmitted symbols
        rx_symbols : ndarray
            Received symbols
            
        Returns
        -------
        delay : int
            Estimated delay in symbols
        """
        # Use shorter sequence for correlation
        max_len = min(len(tx_symbols), len(rx_symbols), 10000)
        tx_short = tx_symbols[:max_len]
        rx_short = rx_symbols[:max_len]
        
        # Cross-correlation
        correlation = np.correlate(tx_short, rx_short, mode='full')
        
        # Find peak
        peak_idx = np.argmax(np.abs(correlation))
        delay = peak_idx - (len(rx_short) - 1)
        
        return delay
    
    def plot_snr_comparison(self, figsize=(12, 6)):
        """Plot SNR comparison across all configurations."""
        if not self.adaptation_results:
            raise RuntimeError("Must run adapt() first")
        
        config_ids = sorted(self.adaptation_results.keys())
        snrs = [self.adaptation_results[cid]['snr_db'] for cid in config_ids]
        
        plt.figure(figsize=figsize)
        bars = plt.bar(config_ids, snrs, color='steelblue', alpha=0.7, edgecolor='black')
        
        # Highlight best config
        best_idx = config_ids.index(self.best_config_id)
        bars[best_idx].set_color('red')
        bars[best_idx].set_alpha(1.0)
        
        plt.xlabel('CTLE Configuration')
        plt.ylabel('SNR [dB]')
        plt.title('SNR Comparison Across CTLE Configurations')
        plt.grid(True, alpha=0.3, axis='y')
        plt.xticks(config_ids)
        
        # Add text annotation for best
        plt.text(self.best_config_id, snrs[best_idx] + 0.5, 
                f'Best: {snrs[best_idx]:.2f} dB',
                ha='center', fontweight='bold', color='red')
        
        plt.tight_layout()
        return plt.gcf()
    
    def plot_eye_diagram(self, signal: np.ndarray, title: str = "Eye Diagram",
                        n_traces: int = 500, figsize=(12, 8)):
        """
        Plot eye diagram of signal.
        
        Parameters
        ----------
        signal : ndarray
            Signal to plot
        title : str
            Plot title
        n_traces : int
            Number of traces to overlay
        figsize : tuple
            Figure size
        """
        samples_per_trace = 3 * self.samples_per_symbol
        offset = 100  # Skip initial transient
        
        signal_trimmed = signal[offset * self.samples_per_symbol:]
        t_trace = np.arange(samples_per_trace) * self.t_sample * 1e12  # in ps
        
        plt.figure(figsize=figsize)
        
        for i in range(min(n_traces, len(signal_trimmed) // self.samples_per_symbol)):
            start_idx = i * self.samples_per_symbol
            end_idx = start_idx + samples_per_trace
            if end_idx > len(signal_trimmed):
                break
            trace = signal_trimmed[start_idx:end_idx]
            plt.plot(t_trace, trace, 'b', alpha=0.05, linewidth=0.5)
        
        # Add vertical lines for best sampling offset (if available)
        if self.best_sampling_offset is not None:
            # Calculate time positions for sampling offset in each symbol period
            symbol_period_ps = (1 / self.symbol_rate) * 1e12
            t_offset = self.best_sampling_offset * self.t_sample * 1e12
            for sym_idx in range(3):  # Show for 3 symbol periods
                t_sample = sym_idx * symbol_period_ps + t_offset
                plt.axvline(x=t_sample, color='red', linestyle='--', linewidth=2, 
                           alpha=0.7, label='Sampling Point' if sym_idx == 0 else '')
        
        plt.xlabel('Time [ps]')
        plt.ylabel('Amplitude [V]')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        if self.best_sampling_offset is not None:
            plt.legend(loc='upper right', fontsize=10)
        plt.tight_layout()
        return plt.gcf()
    
    def plot_comparison_eye_diagrams(self, rx_signal_data: np.ndarray, 
                                    figsize=(16, 6)):
        """
        Plot eye diagrams before and after CTLE.
        
        Parameters
        ----------
        rx_signal_data : ndarray
            Data portion of RX signal
        figsize : tuple
            Figure size
        """
        if self.best_config_id is None:
            raise RuntimeError("Must run adapt() first")
        
        # Apply best CTLE
        rx_ctle = self.apply_ctle(rx_signal_data, self.best_config_id)
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        samples_per_trace = 3 * self.samples_per_symbol
        offset = 100
        n_traces = 500
        
        # Before CTLE
        signal_trimmed = rx_signal_data[offset * self.samples_per_symbol:]
        t_trace = np.arange(samples_per_trace) * self.t_sample * 1e12
        
        for i in range(min(n_traces, len(signal_trimmed) // self.samples_per_symbol)):
            start_idx = i * self.samples_per_symbol
            end_idx = start_idx + samples_per_trace
            if end_idx > len(signal_trimmed):
                break
            trace = signal_trimmed[start_idx:end_idx]
            ax1.plot(t_trace, trace, 'r', alpha=0.05, linewidth=0.5)
        
        # Add vertical lines for best sampling offset
        symbol_period_ps = (1 / self.symbol_rate) * 1e12
        t_offset = self.best_sampling_offset * self.t_sample * 1e12
        for sym_idx in range(3):  # Show for 3 symbol periods
            t_sample = sym_idx * symbol_period_ps + t_offset
            ax1.axvline(x=t_sample, color='yellow', linestyle='--', linewidth=2, 
                       alpha=0.8, label='Sampling Point' if sym_idx == 0 else '')
        
        ax1.set_xlabel('Time [ps]')
        ax1.set_ylabel('Amplitude [V]')
        ax1.set_title('Eye Diagram - Before CTLE (After Channel)')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=10)
        
        # After CTLE
        signal_trimmed = rx_ctle[offset * self.samples_per_symbol:]
        
        for i in range(min(n_traces, len(signal_trimmed) // self.samples_per_symbol)):
            start_idx = i * self.samples_per_symbol
            end_idx = start_idx + samples_per_trace
            if end_idx > len(signal_trimmed):
                break
            trace = signal_trimmed[start_idx:end_idx]
            ax2.plot(t_trace, trace, 'b', alpha=0.05, linewidth=0.5)
        
        # Add vertical lines for best sampling offset
        for sym_idx in range(3):  # Show for 3 symbol periods
            t_sample = sym_idx * symbol_period_ps + t_offset
            ax2.axvline(x=t_sample, color='red', linestyle='--', linewidth=2, 
                       alpha=0.8, label='Sampling Point' if sym_idx == 0 else '')
        
        ax2.set_xlabel('Time [ps]')
        ax2.set_ylabel('Amplitude [V]')
        ax2.set_title(f'Eye Diagram - After CTLE (Config {self.best_config_id})')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def get_eye_metrics(self, signal: np.ndarray, 
                       sampling_offset: Optional[int] = None) -> Dict:
        """
        Calculate eye diagram metrics (height, width).
        
        Parameters
        ----------
        signal : ndarray
            Signal to analyze
        sampling_offset : int, optional
            Sampling offset (default: use best_sampling_offset)
            
        Returns
        -------
        metrics : dict
            Eye height, width, and other metrics
        """
        if sampling_offset is None:
            if self.best_sampling_offset is None:
                raise RuntimeError("Must run adapt() first")
            sampling_offset = self.best_sampling_offset
        
        # Sample signal at offset
        offset = 100  # Skip transient
        signal_trimmed = signal[offset * self.samples_per_symbol:]
        sampled = signal_trimmed[sampling_offset::self.samples_per_symbol]
        
        # Get symbols
        symbols = self.symbols_from_signal(sampled, self.best_vpnt)
        
        # Calculate eye height for each level transition
        # Eye height is minimum vertical distance between adjacent levels
        level_voltages = {}
        for sym in range(self.n_levels):
            mask = symbols == sym
            if np.any(mask):
                level_voltages[sym] = sampled[mask]
        
        # Calculate separation between levels
        separations = []
        for sym in range(self.n_levels - 1):
            if sym in level_voltages and (sym+1) in level_voltages:
                max_lower = np.max(level_voltages[sym])
                min_upper = np.min(level_voltages[sym+1])
                sep = min_upper - max_lower
                separations.append(sep)
        
        eye_height = np.min(separations) if separations else 0.0
        
        # Eye width estimation (time where signal is stable)
        # Use jitter estimate from sampling around optimal point
        time_samples = []
        for offset_test in range(max(0, sampling_offset-5), 
                                 min(self.samples_per_symbol, sampling_offset+6)):
            sampled_test = signal_trimmed[offset_test::self.samples_per_symbol]
            symbols_test = self.symbols_from_signal(sampled_test, self.best_vpnt)
            
            # Trim both arrays to the same length to avoid shape mismatch
            min_len = min(len(symbols_test), len(symbols))
            if min_len == 0:
                continue
            symbols_test_trimmed = symbols_test[:min_len]
            symbols_trimmed = symbols[:min_len]
            
            error_rate = np.mean(symbols_test_trimmed != symbols_trimmed)
            if error_rate < 0.01:  # Less than 1% error
                time_samples.append(offset_test)
        
        eye_width_samples = len(time_samples)
        eye_width_ps = eye_width_samples * self.t_sample * 1e12
        
        return {
            'eye_height': eye_height,
            'eye_width_samples': eye_width_samples,
            'eye_width_ps': eye_width_ps,
            'level_voltages': level_voltages
        }
    
    def plot_adc_alignment(self, rx_signal_data: np.ndarray,
                          tx_symbols_data: np.ndarray,
                          n_symbols: int = 200,
                          start_symbol: int = 100,
                          figsize: tuple = (16, 10)):
        """
        Plot ADC outputs aligned with TX symbols to visualize SER.
        
        Shows:
        1. TX symbols vs time
        2. RX sampled values vs time
        3. RX decoded symbols vs time
        4. TX vs RX overlay with error highlighting
        5. Symbol distribution histogram
        
        Parameters
        ----------
        rx_signal_data : ndarray
            RX signal after CTLE (data portion)
        tx_symbols_data : ndarray
            Transmitted symbols (0, 1, 2, 3)
        n_symbols : int
            Number of symbols to plot (default: 200)
        start_symbol : int
            Starting symbol index (default: 100, skip transients)
        figsize : tuple
            Figure size (default: (16, 10))
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure
        """
        if self.best_config_id is None:
            raise RuntimeError("Must run adapt() first")
        
        # Apply CTLE to data signal
        rx_ctle = self.apply_ctle(rx_signal_data, self.best_config_id)
        
        # Sample at best offset
        sampled_signal = rx_ctle[self.best_sampling_offset::self.samples_per_symbol]
        
        # Decode symbols
        rx_symbols = self.symbols_from_signal(sampled_signal, self.best_vpnt)
        
        # Align TX and RX symbols using cross-correlation
        search_len = min(5000, len(tx_symbols_data), len(rx_symbols))
        correlation = np.correlate(rx_symbols[:search_len], tx_symbols_data[:search_len], mode='full')
        delay = np.argmax(correlation) - (search_len - 1)
        
        # Align sequences
        if delay >= 0:
            tx_aligned = tx_symbols_data[delay:]
            rx_aligned_samples = sampled_signal[:len(tx_aligned)]
            rx_aligned_symbols = rx_symbols[:len(tx_aligned)]
        else:
            rx_aligned_samples = sampled_signal[-delay:]
            rx_aligned_symbols = rx_symbols[-delay:]
            tx_aligned = tx_symbols_data[:len(rx_aligned_symbols)]
        
        # Limit to common length
        n_compare = min(len(tx_aligned), len(rx_aligned_symbols))
        tx_aligned = tx_aligned[:n_compare]
        rx_aligned_samples = rx_aligned_samples[:n_compare]
        rx_aligned_symbols = rx_aligned_symbols[:n_compare]
        
        # Extract plot window
        end_symbol = min(start_symbol + n_symbols, n_compare)
        plot_indices = np.arange(n_symbols)
        
        tx_plot = tx_aligned[start_symbol:end_symbol]
        rx_samples_plot = rx_aligned_samples[start_symbol:end_symbol]
        rx_symbols_plot = rx_aligned_symbols[start_symbol:end_symbol]
        
        # Calculate errors
        errors = (tx_plot != rx_symbols_plot)
        n_errors = np.sum(errors)
        ser = n_errors / len(tx_plot)
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        
        # Plot 1: TX Symbols
        ax1 = plt.subplot(3, 2, 1)
        ax1.step(plot_indices, tx_plot, where='mid', linewidth=2, color='blue', label='TX Symbols')
        ax1.set_ylabel('Symbol Value', fontsize=10)
        ax1.set_title(f'TX Symbols (Symbols {start_symbol}-{end_symbol-1})', fontweight='bold')
        ax1.set_ylim([-0.5, 3.5])
        ax1.set_yticks([0, 1, 2, 3])
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: RX Sampled Values
        ax2 = plt.subplot(3, 2, 2)
        ax2.plot(plot_indices, rx_samples_plot, 'o-', markersize=4, linewidth=1, 
                alpha=0.7, color='green', label='RX Sampled')
        # Add PAM level lines and thresholds
        if self.n_levels == 4:
            pam_levels = np.array([-1.5, -0.5, 0.5, 1.5]) * self.best_vpnt * self.pam_alpha
            thresholds = np.array([-1.0, 0.0, 1.0]) * self.best_vpnt * self.pam_alpha
        elif self.n_levels == 6:
            # PAM6 levels: (-5, -3, -1, 1, 3, 5) * vp6t/3 * pam_alpha
            # PAM6 thresholds: (-4, -2, 0, 2, 4) * vp6t/3 * pam_alpha
            pam_levels = np.array([-5, -3, -1, 1, 3, 5]) * self.best_vpnt / 3 * self.pam_alpha
            thresholds = np.array([-4, -2, 0, 2, 4]) * self.best_vpnt / 3 * self.pam_alpha
        elif self.n_levels == 8:
            # PAM8 levels: (-7, -5, -3, -1, 1, 3, 5, 7) * vp8t/4 * pam_alpha
            # PAM8 thresholds: (-6, -4, -2, 0, 2, 4, 6) * vp8t/4 * pam_alpha
            pam_levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7]) * self.best_vpnt / 4 * self.pam_alpha
            thresholds = np.array([-6, -4, -2, 0, 2, 4, 6]) * self.best_vpnt / 4 * self.pam_alpha
        else:
            pam_levels = []
            thresholds = []
            
        for i, level in enumerate(pam_levels):
            ax2.axhline(y=level, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax2.text(n_symbols * 1.01, level, f'Lvl {i}', va='center', fontsize=8, color='gray')
        # Add thresholds
        for thresh in thresholds:
            ax2.axhline(y=thresh, color='red', linestyle=':', linewidth=1, alpha=0.5)
        ax2.set_ylabel('Amplitude [V]', fontsize=10)
        vpnt_name = f"V_P{self.n_levels}T"
        ax2.set_title(f'RX Sampled Values ({vpnt_name}={self.best_vpnt:.4f}V)', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: RX Decoded Symbols
        ax3 = plt.subplot(3, 2, 3)
        ax3.step(plot_indices, rx_symbols_plot, where='mid', linewidth=2, 
                color='orange', label='RX Decoded')
        ax3.set_ylabel('Symbol Value', fontsize=10)
        ax3.set_title('RX Decoded Symbols', fontweight='bold')
        ax3.set_ylim([-0.5, 3.5])
        ax3.set_yticks([0, 1, 2, 3])
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Plot 4: TX vs RX Overlay with Error Highlighting
        ax4 = plt.subplot(3, 2, 4)
        ax4.step(plot_indices, tx_plot, where='mid', linewidth=2.5, 
                label='TX', alpha=0.8, color='blue')
        ax4.step(plot_indices, rx_symbols_plot, where='mid', linewidth=1.5, 
                label='RX', alpha=0.8, linestyle='--', color='red')
        # Highlight errors
        if np.any(errors):
            error_indices = np.where(errors)[0]
            ax4.scatter(error_indices, tx_plot[errors], color='red', s=150, 
                       marker='x', linewidths=3, label=f'Errors ({n_errors})', zorder=5)
        ax4.set_ylabel('Symbol Value', fontsize=10)
        ax4.set_title(f'TX vs RX Comparison (SER = {ser:.2e}, {n_errors}/{len(tx_plot)} errors)', 
                     fontweight='bold')
        ax4.set_ylim([-0.5, 3.5])
        ax4.set_yticks([0, 1, 2, 3])
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Plot 5: Symbol Distribution
        ax5 = plt.subplot(3, 2, 5)
        symbol_labels = [str(i) for i in range(self.n_levels)]
        tx_counts = [np.sum(tx_aligned[:n_compare] == sym) for sym in range(self.n_levels)]
        rx_counts = [np.sum(rx_aligned_symbols[:n_compare] == sym) for sym in range(self.n_levels)]
        
        x = np.arange(len(symbol_labels))
        width = 0.35
        
        bars1 = ax5.bar(x - width/2, tx_counts, width, label='TX', alpha=0.8, color='blue')
        bars2 = ax5.bar(x + width/2, rx_counts, width, label='RX', alpha=0.8, color='orange')
        
        ax5.set_xlabel('Symbol Value', fontsize=10)
        ax5.set_ylabel('Count', fontsize=10)
        ax5.set_title('Symbol Distribution (Full Sequence)', fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels(symbol_labels)
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
        
        # Add count labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=8)
        
        # Plot 6: Error Statistics
        ax6 = plt.subplot(3, 2, 6)
        ax6.axis('off')
        
        # Calculate statistics
        total_errors = np.sum(tx_aligned != rx_aligned_symbols)
        total_symbols = len(tx_aligned)
        total_ser = total_errors / total_symbols
        
        # Error breakdown by symbol
        error_by_symbol = []
        for sym in range(self.n_levels):
            mask = tx_aligned == sym
            if np.sum(mask) > 0:
                sym_errors = np.sum((tx_aligned[mask] != rx_aligned_symbols[mask]))
                sym_total = np.sum(mask)
                error_by_symbol.append((sym, sym_errors, sym_total, sym_errors/sym_total))
        
        stats_text = "Alignment Statistics:\n"
        stats_text += "=" * 40 + "\n"
        stats_text += f"Alignment delay: {delay} symbols\n"
        stats_text += f"Compared symbols: {total_symbols}\n"
        stats_text += f"Total errors: {total_errors}\n"
        stats_text += f"Overall SER: {total_ser:.6e}\n"
        stats_text += f"BER (est): {total_ser/2:.6e}\n\n"
        
        stats_text += "Error Breakdown by Symbol:\n"
        stats_text += "-" * 40 + "\n"
        for sym, errs, total, ser_sym in error_by_symbol:
            stats_text += f"Symbol {sym}: {errs}/{total} ({ser_sym:.2e})\n"
        
        vpnt_name = f"V_P{self.n_levels}T"
        stats_text += "\n" + "=" * 40 + "\n"
        stats_text += f"Best CTLE Config: {self.best_config_id}\n"
        stats_text += f"Sampling Offset: {self.best_sampling_offset}\n"
        stats_text += f"{vpnt_name}: {self.best_vpnt:.4f} V\n"
        stats_text += f"SNR: {self.best_snr_db:.2f} dB"
        
        ax6.text(0.1, 0.95, stats_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        return fig
    
    def print_summary(self, rx_signal_data: Optional[np.ndarray] = None,
                     tx_symbols_data: Optional[np.ndarray] = None):
        """
        Print comprehensive summary of adaptation and data processing.
        
        Parameters
        ----------
        rx_signal_data : ndarray, optional
            Data portion for metric calculation
        tx_symbols_data : ndarray, optional
            TX symbols for SER calculation
        """
        print("\n" + "="*70)
        print("ADAPTIVE CTLE SUMMARY")
        print("="*70)
        
        vpnt_name = f"V_P{self.n_levels}T"
        print("\nAdaptation Results:")
        print(f"  Best Configuration: {self.best_config_id}")
        print(f"  Best SNR:          {self.best_snr_db:.2f} dB")
        print(f"  Best Offset:       {self.best_sampling_offset} samples")
        print(f"  {vpnt_name}:             {self.best_vpnt:.4f} V")
        
        print("\nExpected Signal Levels:")
        if self.n_levels == 4:
            pam_levels_map = [-3, -1, 1, 3]
        elif self.n_levels == 6:
            pam_levels_map = [-5, -3, -1, 1, 3, 5]
        elif self.n_levels == 8:
            pam_levels_map = [-7, -5, -3, -1, 1, 3, 5, 7]
        else:
            pam_levels_map = list(range(self.n_levels))
            
        for sym, level in self.best_expected_levels.items():
            pam_level = pam_levels_map[sym]
            print(f"  Symbol {sym} (PAM level {pam_level:+d}): {level:+.4f} V")
        
        if rx_signal_data is not None:
            print("\nProcessing data sequence...")
            results = self.process_data_sequence(rx_signal_data, tx_symbols_data)
            
            print("\nData Sequence Metrics:")
            if 'ser' in results:
                print(f"  Symbol Error Rate: {results['ser']:.6e}")
                print(f"  Errors:           {results['n_errors']} / {results['n_symbols']}")
                print(f"  Delay (symbols):  {results['delay_symbols']}")
            
            # Eye metrics
            eye_before = self.get_eye_metrics(rx_signal_data)
            eye_after = self.get_eye_metrics(self.rx_signal_ctle)
            
            print("\nEye Diagram Metrics:")
            print("  Before CTLE:")
            print(f"    Eye Height:  {eye_before['eye_height']:.4f} V")
            print(f"    Eye Width:   {eye_before['eye_width_ps']:.2f} ps")
            print("  After CTLE:")
            print(f"    Eye Height:  {eye_after['eye_height']:.4f} V")
            print(f"    Eye Width:   {eye_after['eye_width_ps']:.2f} ps")
        
        print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    print("Adaptive CTLE module loaded successfully!")
    print("Use AdaptiveCTLE class to perform adaptation.")
