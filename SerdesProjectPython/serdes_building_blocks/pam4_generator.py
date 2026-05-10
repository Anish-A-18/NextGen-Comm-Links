"""
PAM-4 Signal Generation Utilities
"""

import numpy as np


class PAM4Generator:
    """
    Generator for PAM-4 (Pulse Amplitude Modulation 4-level) signals.
    
    PAM-4 encodes 2 bits per symbol using 4 voltage levels: -3, -1, +1, +3
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize PAM4 generator.
        
        Parameters
        ----------
        seed : int
            Random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)
        self.levels = np.array([-3, -1, 1, 3])  # PAM-4 levels
        
    def generate_random_symbols(self, n_symbols: int) -> np.ndarray:
        """
        Generate random PAM-4 symbols.
        
        Parameters
        ----------
        n_symbols : int
            Number of symbols to generate
            
        Returns
        -------
        symbols : ndarray
            Array of PAM-4 symbols (values: -3, -1, 1, 3)
        """
        # Generate random integers 0-3, map to PAM-4 levels
        indices = self.rng.integers(0, 4, size=n_symbols)
        symbols = self.levels[indices]
        return symbols
    
    def generate_prbs(self, n_symbols: int, pattern: str = 'prbs7') -> np.ndarray:
        """
        Generate pseudo-random binary sequence (PRBS) PAM-4 symbols.
        
        Parameters
        ----------
        n_symbols : int
            Number of symbols to generate
        pattern : str
            PRBS pattern ('prbs7', 'prbs15', 'prbs23', 'prbs31')
            
        Returns
        -------
        symbols : ndarray
            Array of PAM-4 symbols
        """
        # Generate PRBS bit sequence (simplified version)
        # In practice, use proper LFSR implementation
        bit_sequence = self._generate_prbs_bits(n_symbols * 2, pattern)
        
        # Convert pairs of bits to PAM-4 symbols
        symbols = self._bits_to_pam4(bit_sequence)
        return symbols[:n_symbols]
    
    def _generate_prbs_bits(self, n_bits: int, pattern: str) -> np.ndarray:
        """Generate PRBS bit sequence (simplified)."""
        # Simplified: just use random bits
        # Real implementation would use LFSR
        return self.rng.integers(0, 2, size=n_bits)
    
    def _bits_to_pam4(self, bits: np.ndarray) -> np.ndarray:
        """
        Convert bit pairs to PAM-4 symbols.
        
        Bit mapping:
        00 -> -3
        01 -> -1
        10 -> +1
        11 -> +3
        """
        n_symbols = len(bits) // 2
        symbols = np.zeros(n_symbols)
        
        for i in range(n_symbols):
            b1 = bits[2*i]
            b0 = bits[2*i + 1]
            
            # Gray coding
            if b1 == 0 and b0 == 0:
                symbols[i] = -3
            elif b1 == 0 and b0 == 1:
                symbols[i] = -1
            elif b1 == 1 and b0 == 0:
                symbols[i] = 1
            else:  # b1 == 1 and b0 == 1
                symbols[i] = 3
                
        return symbols
    
    def oversample(self, symbols: np.ndarray, samples_per_symbol: int,
                  pulse_shape: str = 'rect') -> np.ndarray:
        """
        Oversample PAM-4 symbols to create time-domain signal.
        
        Parameters
        ----------
        symbols : ndarray
            PAM-4 symbol sequence
        samples_per_symbol : int
            Number of samples per symbol
        pulse_shape : str
            Pulse shaping ('rect', 'rrc', 'rc')
            
        Returns
        -------
        signal : ndarray
            Oversampled signal
        """
        if pulse_shape == 'rect':
            # Simple rectangular pulse (NRZ)
            signal = np.repeat(symbols, samples_per_symbol)
        elif pulse_shape == 'rrc':
            # Root raised cosine (not implemented yet)
            signal = np.repeat(symbols, samples_per_symbol)
            print("Warning: RRC not implemented, using rectangular")
        else:
            signal = np.repeat(symbols, samples_per_symbol)
            
        return signal
    
    def generate_pattern(self, pattern_type: str, n_symbols: int) -> np.ndarray:
        """
        Generate specific test patterns.
        
        Parameters
        ----------
        pattern_type : str
            Pattern type: 'alternating', 'max_isi', 'min_isi', 'random'
        n_symbols : int
            Number of symbols
            
        Returns
        -------
        symbols : ndarray
            Symbol sequence
        """
        if pattern_type == 'alternating':
            # Alternating between min and max
            symbols = np.tile([-3, 3], n_symbols // 2 + 1)[:n_symbols]
        elif pattern_type == 'max_isi':
            # Maximum inter-symbol interference: all transitions
            symbols = np.tile([-3, -1, 1, 3], n_symbols // 4 + 1)[:n_symbols]
        elif pattern_type == 'min_isi':
            # Minimum ISI: constant level
            symbols = np.ones(n_symbols) * 3
        elif pattern_type == 'random':
            symbols = self.generate_random_symbols(n_symbols)
        else:
            raise ValueError(f"Unknown pattern type: {pattern_type}")
            
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
