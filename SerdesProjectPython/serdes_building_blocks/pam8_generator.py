"""
PAM-8 Signal Generation Utilities
"""

import numpy as np


class PAM8Generator:
    """
    Generator for PAM-8 (Pulse Amplitude Modulation 8-level) signals.
    
    PAM-8 encodes 3 bits per symbol using 8 voltage levels: -7, -5, -3, -1, +1, +3, +5, +7
    This provides higher spectral efficiency than PAM-4/PAM-6 at the cost of reduced noise margin.
    
    Note: alpha scaling is applied to thresholds and expected levels for RX slicing,
    not to TX signal generation.
    """
    
    def __init__(self, seed: int = 42, alpha: float = 0.87):
        """
        Initialize PAM8 generator.
        
        Parameters
        ----------
        seed : int
            Random seed for reproducibility
        alpha : float
            Scaling factor for RX thresholds and expected levels (default: 0.87)
        """
        self.rng = np.random.default_rng(seed)
        self.levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7])  # PAM-8 levels (unscaled for TX)
        self.n_levels = 8
        self.alpha = alpha  # Scaling factor for RX thresholds and expected levels
        
    def generate_random_symbols(self, n_symbols: int) -> np.ndarray:
        """
        Generate random PAM-8 symbols.
        
        Parameters
        ----------
        n_symbols : int
            Number of symbols to generate
            
        Returns
        -------
        symbols : ndarray
            Array of symbol indices (values: 0, 1, 2, 3, 4, 5, 6, 7)
        """
        # Generate random integers 0-7
        symbols = self.rng.integers(0, 8, size=n_symbols)
        return symbols
    
    def symbols_to_levels(self, symbols: np.ndarray) -> np.ndarray:
        """
        Convert symbol indices to voltage levels.
        
        Parameters
        ----------
        symbols : ndarray
            Array of symbol indices (0-7)
            
        Returns
        -------
        levels : ndarray
            Array of voltage levels (-7, -5, -3, -1, 1, 3, 5, 7)
        """
        return self.levels[symbols]
    
    def levels_to_symbols(self, levels: np.ndarray) -> np.ndarray:
        """
        Convert voltage levels to symbol indices.
        
        Parameters
        ----------
        levels : ndarray
            Array of voltage levels
            
        Returns
        -------
        symbols : ndarray
            Array of symbol indices (0-7)
        """
        symbols = np.zeros(len(levels), dtype=int)
        for i, level in enumerate(self.levels):
            symbols[np.isclose(levels, level, atol=0.1)] = i
        return symbols
    
    def generate_prbs(self, n_symbols: int, pattern: str = 'prbs7') -> np.ndarray:
        """
        Generate pseudo-random binary sequence (PRBS) PAM-8 symbols.
        
        Parameters
        ----------
        n_symbols : int
            Number of symbols to generate
        pattern : str
            PRBS pattern ('prbs7', 'prbs15', 'prbs23', 'prbs31')
            
        Returns
        -------
        symbols : ndarray
            Array of PAM-8 symbol indices (0-7)
        """
        # PAM-8 uses exactly 3 bits per symbol (8 = 2^3)
        n_bits = n_symbols * 3
        bit_sequence = self._generate_prbs_bits(n_bits, pattern)
        
        # Convert 3-bit groups to PAM-8 symbols
        symbols = self._bits_to_pam8(bit_sequence)
        return symbols[:n_symbols]
    
    def _generate_prbs_bits(self, n_bits: int, pattern: str) -> np.ndarray:
        """Generate PRBS bit sequence (simplified)."""
        # Simplified: just use random bits
        # Real implementation would use LFSR
        return self.rng.integers(0, 2, size=n_bits)
    
    def _bits_to_pam8(self, bits: np.ndarray) -> np.ndarray:
        """
        Convert 3-bit groups to PAM-8 symbols with 100% efficiency.
        
        Method: 3 bits -> 8 possible values (0-7)
                Direct mapping to PAM-8 symbols (perfect efficiency)
        
        Parameters
        ----------
        bits : ndarray
            Bit sequence (length must be multiple of 3)
            
        Returns
        -------
        symbols : ndarray
            PAM-8 symbol indices (1 symbol per 3 bits)
        """
        n_symbols = len(bits) // 3
        symbols = np.zeros(n_symbols, dtype=int)
        
        for i in range(n_symbols):
            # Extract 3 bits
            bit_group = bits[3*i : 3*i + 3]
            
            # Convert 3 bits to integer (0-7)
            # MSB first: bit_value = b0*4 + b1*2 + b2*1
            bit_value = bit_group[0] * 4 + bit_group[1] * 2 + bit_group[2]
            
            symbols[i] = bit_value
                
        return symbols
    
    def oversample(self, symbols: np.ndarray, samples_per_symbol: int,
                  pulse_shape: str = 'rect') -> np.ndarray:
        """
        Oversample PAM-8 symbols to create time-domain signal.
        
        Parameters
        ----------
        symbols : ndarray
            PAM-8 symbol sequence (indices 0-7 or levels -7 to 7)
        samples_per_symbol : int
            Number of samples per symbol
        pulse_shape : str
            Pulse shaping ('rect', 'rrc', 'rc')
            
        Returns
        -------
        signal : ndarray
            Oversampled signal
        """
        # Check if symbols are indices or levels
        if np.all((symbols >= 0) & (symbols < 8) & (symbols == symbols.astype(int))):
            # Symbols are indices, convert to levels
            levels = self.symbols_to_levels(symbols.astype(int))
        else:
            # Already levels
            levels = symbols
        
        if pulse_shape == 'rect':
            # Simple rectangular pulse (NRZ)
            signal = np.repeat(levels, samples_per_symbol)
        elif pulse_shape == 'rrc':
            # Root raised cosine (not implemented yet)
            signal = np.repeat(levels, samples_per_symbol)
            print("Warning: RRC not implemented, using rectangular")
        else:
            signal = np.repeat(levels, samples_per_symbol)
            
        return signal
    
    def generate_pattern(self, pattern_type: str, n_symbols: int) -> np.ndarray:
        """
        Generate specific test patterns.
        
        Parameters
        ----------
        pattern_type : str
            Pattern type: 'alternating', 'max_isi', 'min_isi', 'random', 'increasing'
        n_symbols : int
            Number of symbols
            
        Returns
        -------
        symbols : ndarray
            Symbol indices (0-7)
        """
        if pattern_type == 'alternating':
            # Alternating between min and max
            symbols = np.tile([0, 7], n_symbols // 2 + 1)[:n_symbols]
        elif pattern_type == 'max_isi':
            # Maximum inter-symbol interference: all transitions
            symbols = np.tile([0, 1, 2, 3, 4, 5, 6, 7], n_symbols // 8 + 1)[:n_symbols]
        elif pattern_type == 'min_isi':
            # Minimum ISI: constant level
            symbols = np.ones(n_symbols, dtype=int) * 7
        elif pattern_type == 'increasing':
            # Increasing ramp pattern
            symbols = np.tile(np.arange(8), n_symbols // 8 + 1)[:n_symbols]
        elif pattern_type == 'random':
            symbols = self.generate_random_symbols(n_symbols)
        else:
            raise ValueError(f"Unknown pattern type: {pattern_type}")
            
        return symbols
    
    def get_thresholds(self, signal_rms: float = 1.0) -> np.ndarray:
        """
        Get decision thresholds for PAM-8 slicer (scaled by alpha).
        
        Parameters
        ----------
        signal_rms : float
            RMS value of the signal for scaling
            
        Returns
        -------
        thresholds : ndarray
            Array of 7 thresholds between the 8 levels, scaled by alpha
        """
        # Thresholds at midpoints between levels
        # Scaled levels: -7, -5, -3, -1, 1, 3, 5, 7
        # Thresholds: -6, -4, -2, 0, 2, 4, 6
        base_thresholds = np.array([-6, -4, -2, 0, 2, 4, 6])
        scale = signal_rms / np.sqrt(np.mean(self.levels**2))
        return base_thresholds * scale * self.alpha
    
    def get_expected_levels(self, signal_rms: float = 1.0) -> np.ndarray:
        """
        Get expected RX levels scaled by alpha.
        
        Parameters
        ----------
        signal_rms : float
            RMS value of the signal for scaling
            
        Returns
        -------
        expected_levels : ndarray
            Expected levels scaled by alpha: (-7, -5, -3, -1, 1, 3, 5, 7) * scale * alpha
        """
        scale = signal_rms / np.sqrt(np.mean(self.levels**2))
        return self.levels * scale * self.alpha
    
    def slice_pam8(self, signal: np.ndarray, thresholds: np.ndarray = None,
                   signal_rms: float = None) -> np.ndarray:
        """
        Slice received signal to PAM-8 symbol decisions (thresholds already scaled by alpha).
        
        Parameters
        ----------
        signal : ndarray
            Received signal
        thresholds : ndarray, optional
            Decision thresholds. If None, computed from signal_rms (already scaled by alpha)
        signal_rms : float, optional
            Signal RMS for threshold calculation
            
        Returns
        -------
        symbols : ndarray
            Decoded symbol indices (0-7)
        """
        if thresholds is None:
            if signal_rms is None:
                signal_rms = np.sqrt(np.mean(signal**2))
            thresholds = self.get_thresholds(signal_rms)  # Already scaled by alpha
        # Note: thresholds from get_thresholds are already scaled by alpha
        symbols = np.zeros(len(signal), dtype=int)
        symbols[signal < thresholds[0]] = 0
        symbols[(signal >= thresholds[0]) & (signal < thresholds[1])] = 1
        symbols[(signal >= thresholds[1]) & (signal < thresholds[2])] = 2
        symbols[(signal >= thresholds[2]) & (signal < thresholds[3])] = 3
        symbols[(signal >= thresholds[3]) & (signal < thresholds[4])] = 4
        symbols[(signal >= thresholds[4]) & (signal < thresholds[5])] = 5
        symbols[(signal >= thresholds[5]) & (signal < thresholds[6])] = 6
        symbols[signal >= thresholds[6]] = 7
        
        return symbols


def add_noise(signal: np.ndarray, snr_db: float, seed: int = 42) -> np.ndarray:
    """
    Add Gaussian noise to signal.
    
    Parameters
    ----------
    signal : ndarray
        Input signal
    snr_db : float
        Signal-to-noise ratio in dB
    seed : int
        Random seed
        
    Returns
    -------
    noisy_signal : ndarray
        Signal with added noise
    """
    rng = np.random.default_rng(seed)
    
    # Calculate signal power
    signal_power = np.mean(signal**2)
    
    # Calculate noise power from SNR
    snr_linear = 10**(snr_db / 10)
    noise_power = signal_power / snr_linear
    
    # Generate noise
    noise = rng.normal(0, np.sqrt(noise_power), size=signal.shape)
    
    return signal + noise


def normalize_signal(signal: np.ndarray, target_amplitude: float = 1.0) -> np.ndarray:
    """
    Normalize signal amplitude.
    
    Parameters
    ----------
    signal : ndarray
        Input signal
    target_amplitude : float
        Target peak amplitude
        
    Returns
    -------
    normalized_signal : ndarray
        Normalized signal
    """
    current_max = np.max(np.abs(signal))
    if current_max > 0:
        return signal * (target_amplitude / current_max)
    return signal


def calculate_ser_pam8(transmitted: np.ndarray, received: np.ndarray) -> float:
    """
    Calculate Symbol Error Rate for PAM-8.
    
    Parameters
    ----------
    transmitted : ndarray
        Transmitted symbol indices (0-7)
    received : ndarray
        Received symbol indices (0-7)
        
    Returns
    -------
    ser : float
        Symbol error rate
    """
    if len(transmitted) != len(received):
        min_len = min(len(transmitted), len(received))
        transmitted = transmitted[:min_len]
        received = received[:min_len]
    
    errors = np.sum(transmitted != received)
    ser = errors / len(transmitted)
    return ser

