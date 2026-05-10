#!/usr/bin/env python3
"""
SerDes System Simulator for PAM-4, PAM-6, and PAM-8
====================================================

Provides a clean, configurable interface for running complete SerDes simulations
from PAM generation through TX FFE, channel, CTLE, and ADC.

Supported modulation schemes:
- PAM-4: 4 voltage levels (-3, -1, +1, +3), 2 bits/symbol
- PAM-6: 6 voltage levels (-5, -3, -1, +1, +3, +5), ~2.58 bits/symbol
- PAM-8: 8 voltage levels (-7, -5, -3, -1, +1, +3, +5, +7), 3 bits/symbol

Classes:
- ConfigSerdesSystemPAM6: Configuration dataclass for system parameters (supports PAM-4/6/8)
- SerdesSystemPAM6: Main simulation class that runs the complete signal chain

Author: Anish Anand
Date: December 2025
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List
import sys
import os

# Add the current directory first to ensure local imports take precedence
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from serdes_channel import SerdesChannel
from pam6_generator import PAM6Generator, add_noise
from pam8_generator import PAM8Generator
from adaptive_ctle import AdaptiveCTLE
from tx_ffe import TX_FFE
from adc_n_bits import ADC_n_bits, add_nonlinearity_distortion
from adaptive_ffe_2 import AdaptiveFFE


class PAM4Generator:
    """
    Generator for PAM-4 (Pulse Amplitude Modulation 4-level) signals.
    
    PAM-4 encodes 2 bits per symbol using 4 voltage levels: -3, -1, +1, +3
    
    Note: alpha scaling is applied to thresholds and expected levels for RX slicing,
    not to TX signal generation.
    """
    
    def __init__(self, seed: int = 42, alpha: float = 0.87):
        self.rng = np.random.default_rng(seed)
        self.levels = np.array([-3, -1, 1, 3])  # PAM-4 levels (unscaled for TX)
        self.n_levels = 4
        self.alpha = alpha  # Scaling factor for RX thresholds and expected levels
        
    def generate_random_symbols(self, n_symbols: int) -> np.ndarray:
        """Generate random PAM-4 symbol indices (0-3)."""
        return self.rng.integers(0, 4, size=n_symbols)
    
    def symbols_to_levels(self, symbols: np.ndarray) -> np.ndarray:
        """Convert symbol indices to voltage levels (unscaled for TX)."""
        return self.levels[symbols]
    
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
            Expected levels scaled by alpha: (-3, -1, 1, 3) * scale * alpha
        """
        scale = signal_rms / np.sqrt(np.mean(self.levels**2))
        return self.levels * scale * self.alpha
    
    def levels_to_symbols(self, levels: np.ndarray) -> np.ndarray:
        """Convert voltage levels to symbol indices."""
        symbols = np.zeros(len(levels), dtype=int)
        for i, level in enumerate(self.levels):
            symbols[np.isclose(levels, level, atol=0.1)] = i
        return symbols
    
    def oversample(self, symbols: np.ndarray, samples_per_symbol: int,
                  pulse_shape: str = 'rect') -> np.ndarray:
        """Oversample PAM-4 symbols to create time-domain signal (unscaled for TX)."""
        if np.all((symbols >= 0) & (symbols < 4) & (symbols == symbols.astype(int))):
            levels = self.symbols_to_levels(symbols.astype(int))
        else:
            levels = symbols
        
        if pulse_shape == 'rect':
            signal = np.repeat(levels, samples_per_symbol)
        else:
            signal = np.repeat(levels, samples_per_symbol)
        return signal
    
    def get_thresholds(self, signal_rms: float = 1.0) -> np.ndarray:
        """
        Get decision thresholds for PAM-4 slicer (scaled by alpha).
        
        Parameters
        ----------
        signal_rms : float
            RMS value of the signal for scaling
            
        Returns
        -------
        thresholds : ndarray
            Thresholds scaled by alpha: (-2, 0, 2) * scale * alpha
        """
        base_thresholds = np.array([-2, 0, 2])
        scale = signal_rms / np.sqrt(np.mean(self.levels**2))
        return base_thresholds * scale * self.alpha
    
    def slice_pam4(self, signal: np.ndarray, thresholds: np.ndarray = None,
                   signal_rms: float = None) -> np.ndarray:
        """Slice received signal to PAM-4 symbol decisions (thresholds already scaled by alpha)."""
        if thresholds is None:
            if signal_rms is None:
                signal_rms = np.sqrt(np.mean(signal**2))
            thresholds = self.get_thresholds(signal_rms)  # Already scaled by alpha
        # Note: thresholds from get_thresholds are already scaled by alpha
        symbols = np.zeros(len(signal), dtype=int)
        symbols[signal < thresholds[0]] = 0
        symbols[(signal >= thresholds[0]) & (signal < thresholds[1])] = 1
        symbols[(signal >= thresholds[1]) & (signal < thresholds[2])] = 2
        symbols[signal >= thresholds[2]] = 3
        return symbols


@dataclass
class ConfigSerdesSystemPAM6:
    """
    Configuration class for PAM SerDes system parameters.
    
    Supports PAM-4, PAM-6, and PAM-8 modulation schemes.
    All system parameters are defined here and can be easily modified
    to run different configurations.
    """
    
    # PAM modulation type (4, 6, or 8)
    n_levels: int = 6
    
    # Symbol rate and sampling
    symbol_rate: float = 64e9  # Hz (64 Gbaud for PCIe 7.0)
    samples_per_symbol: int = 32
    
    # Channel parameters
    channel_length: float = 0.05  # meters (200 mm)
    channel_z0: float = 50.0  # Ohms
    channel_eps_r: float = 4.9  # Dielectric constant
    channel_cap_source: float = 50e-15  # Farads
    channel_cap_term: float = 50e-15  # Farads
    channel_theta_0: float = 0.015
    channel_k_r: float = 120
    channel_rdc: float = 0.0002
    
    # Signal generation
    total_symbols: int = 32000 + 50000 + 50000 + 50000  # Total symbols to generate
    adaptation_symbols: int = 32000  # Symbols used for CTLE adaptation
    pam6_seed: int = 42  # Random seed for reproducibility (kept for backward compatibility)
    pam_seed: int = None  # Alternative name for seed
    pam_alpha: float = 0.87  # Scaling factor for RX thresholds and expected levels
    
    # TX FFE configuration
    tx_ffe_taps: List[float] = field(default_factory=lambda: [0.083, -0.208, 0.709, 0])

    # Noise configuration
    snr_db: float = 28.0  # Signal-to-noise ratio in dB
    noise_seed: int = 123
    
    # CTLE configuration
    n_ctle_configs: int = 16  # Number of CTLE configurations to test
    
    # ADC configuration
    adc_n_bits: int = 6
    adc_full_scale_voltage: float = 2.0
    adc_target_rms: float = 0.5

    # Nonlinearity configuration
    non_linearity_gain_compression: float = 0.08
    non_linearity_third_order: float = 0.15

    # RX FFE configuration
    rx_ffe_algorithm: str = 'lms'
    rx_ffe_mu: float = 0.01
    rx_ffe_training_symbols: int = 50000
    rx_ffe_n_taps: int = 8
    rx_ffe_n_precursor: int = 3
    rx_ffe_n_postcursor: int = 4  # Will be computed in __post_init__
    rx_ffe_lambda: float = 0.95
    rx_ffe_taps: List[float] = field(default_factory=lambda: [np.zeros(3), 1, np.zeros(4)])

    # Output control
    verbose: bool = True  # Print detailed progress information
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.n_levels not in [4, 6, 8]:
            raise ValueError(f"n_levels must be 4, 6, or 8, got {self.n_levels}")
        if self.adaptation_symbols >= self.total_symbols:
            raise ValueError(f"adaptation_symbols ({self.adaptation_symbols}) must be < total_symbols ({self.total_symbols})")
        if self.samples_per_symbol < 1:
            raise ValueError(f"samples_per_symbol must be >= 1, got {self.samples_per_symbol}")
        if self.adc_n_bits < 1 or self.adc_n_bits > 16:
            raise ValueError(f"adc_n_bits must be between 1 and 16, got {self.adc_n_bits}")
        # Handle pam_seed alias
        if self.pam_seed is not None:
            self.pam6_seed = self.pam_seed
        self.rx_ffe_n_postcursor = self.rx_ffe_n_taps - self.rx_ffe_n_precursor - 1
    
    def update(self, **kwargs):
        """
        Update configuration parameters.
        
        Example:
            config.update(channel_length=0.25, snr_db=30)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
    
    def copy(self) -> 'ConfigSerdesSystemPAM6':
        """Create a deep copy of the configuration."""
        import copy
        return copy.deepcopy(self)
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            'n_levels': self.n_levels,
            'pam_alpha': self.pam_alpha,
            'symbol_rate': self.symbol_rate,
            'samples_per_symbol': self.samples_per_symbol,
            'channel_length': self.channel_length,
            'channel_z0': self.channel_z0,
            'channel_eps_r': self.channel_eps_r,
            'total_symbols': self.total_symbols,
            'adaptation_symbols': self.adaptation_symbols,
            'tx_ffe_taps': self.tx_ffe_taps,
            'snr_db': self.snr_db,
            'n_ctle_configs': self.n_ctle_configs,
            'adc_n_bits': self.adc_n_bits,
            'adc_target_rms': self.adc_target_rms,
        }
    
    def print_summary(self):
        """Print a summary of the configuration."""
        bits_per_symbol = np.log2(self.n_levels)
        data_rate = self.symbol_rate * bits_per_symbol / 1e9
        
        print("="*70)
        print(f"PAM-{self.n_levels} SerDes System Configuration")
        print("="*70)
        print(f"  Modulation: PAM-{self.n_levels} ({self.n_levels} levels)")
        print(f"  Symbol Rate: {self.symbol_rate/1e9:.1f} Gbaud")
        print(f"  Data Rate: {data_rate:.2f} Gb/s")
        print(f"  Samples per Symbol: {self.samples_per_symbol}")
        print(f"  Channel Length: {self.channel_length*1000:.1f} mm")
        print(f"  Total Symbols: {self.total_symbols}")
        print(f"  Adaptation Symbols: {self.adaptation_symbols}")
        print(f"  Data Symbols: {self.total_symbols - self.adaptation_symbols}")
        print(f"  TX FFE Taps: {self.tx_ffe_taps}")
        print(f"  SNR: {self.snr_db:.1f} dB")
        print(f"  CTLE Configs: {self.n_ctle_configs}")
        print(f"  ADC: {self.adc_n_bits} bits, Target RMS: {self.adc_target_rms:.3f} V")
        print("="*70)


class SerdesSystemPAM6:
    """
    Complete PAM SerDes system simulator supporting PAM-4, PAM-6, and PAM-8.
    
    Implements the full signal chain:
    1. PAM symbol generation (4, 6, or 8 levels)
    2. Oversampling
    3. TX FFE (pre-emphasis)
    4. SerDes channel
    5. AWGN noise
    6. Adaptive CTLE
    7. Normalization
    8. ADC conversion
    9. Alignment of TX symbols with ADC output
    
    Usage:
        config = ConfigSerdesSystemPAM6(n_levels=6, channel_length=0.20, snr_db=28)
        system = SerdesSystemPAM6(config)
        results = system.run_ctle_adc()
        
        # Access results
        tx_symbols = results['tx_symbols']
        adc_output = results['adc_output']
        ctle_snr = results['ctle_best_snr_db']
    """
    
    def __init__(self, config: ConfigSerdesSystemPAM6):
        """
        Initialize PAM SerDes system with configuration.
        
        Parameters
        ----------
        config : ConfigSerdesSystemPAM6
            System configuration object
        """
        self.config = config
        self.results = None
        
        # Components (will be initialized during run)
        self.channel = None
        self.pam_gen = None  # Generic PAM generator
        self.pam6_gen = None  # Alias for backward compatibility
        self.tx_ffe = None
        self.adaptive_ctle = None
        self.adc = None
        
        # Initialize the appropriate PAM generator
        self._init_pam_generator()
    
    def _init_pam_generator(self):
        """Initialize the PAM generator based on n_levels with pam_alpha scaling."""
        alpha = self.config.pam_alpha
        if self.config.n_levels == 4:
            self.pam_gen = PAM4Generator(seed=self.config.pam6_seed, alpha=alpha)
        elif self.config.n_levels == 6:
            self.pam_gen = PAM6Generator(seed=self.config.pam6_seed, alpha=alpha)
        elif self.config.n_levels == 8:
            self.pam_gen = PAM8Generator(seed=self.config.pam6_seed, alpha=alpha)
        else:
            raise ValueError(f"Unsupported n_levels: {self.config.n_levels}")
        # Backward compatibility alias
        self.pam6_gen = self.pam_gen
    
    def _print(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.config.verbose:
            print(message)
    
    def _generate_pam_signal(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate PAM symbols and oversample.
        
        Returns
        -------
        tx_symbols : ndarray
            Transmitted symbol indices
        tx_signal : ndarray
            Oversampled transmitted signal
        """
        n_levels = self.config.n_levels
        self._print("\n" + "="*70)
        self._print(f"Step 1: Generating PAM-{n_levels} Signal")
        self._print("="*70)
        
        # Generate PAM symbol indices
        tx_symbols = self.pam_gen.generate_random_symbols(self.config.total_symbols)
        
        # Convert to voltage levels
        tx_pam_levels = self.pam_gen.symbols_to_levels(tx_symbols)
        
        # Oversample
        tx_signal = self.pam_gen.oversample(tx_symbols, self.config.samples_per_symbol)
        
        self._print(f"  Generated {self.config.total_symbols} symbols")
        self._print(f"  Symbol distribution: {np.bincount(tx_symbols)}")
        self._print(f"  TX signal length: {len(tx_signal)} samples")
        self._print(f"  PAM-{n_levels} levels: {self.pam_gen.levels}")
        
        return tx_symbols, tx_signal
    
    # Backward compatibility alias
    def _generate_pam6_signal(self) -> Tuple[np.ndarray, np.ndarray]:
        """Alias for _generate_pam_signal for backward compatibility."""
        return self._generate_pam_signal()
    
    def _apply_tx_ffe(self, tx_signal: np.ndarray, tx_symbol: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """
        Apply transmitter FFE (pre-emphasis).
        
        Parameters
        ----------
        tx_signal : ndarray
            Input signal
        tx_symbol : ndarray
            Transmitted symbols
            
        Returns
        -------
        tx_ffe_output : ndarray
            Equalized signal
        tx_ffe_symbol : ndarray
            Symbols with delay
        tx_ffe_taps : list
            FFE tap values
        """
        self._print("\n" + "="*70)
        self._print("Step 2: Applying TX FFE")
        self._print("="*70)
        
        self.tx_ffe = TX_FFE(taps=self.config.tx_ffe_taps)
        tx_ffe_output = self.tx_ffe.equalize(tx_signal, samples_per_symbol=self.config.samples_per_symbol)
        
        self._print(f"  TX FFE taps: {self.tx_ffe.get_taps()}")
        self._print(f"  Output length: {len(tx_ffe_output)} samples")

        tx_ffe_symbol = np.concatenate([[0], tx_symbol])
        self._print(f"  TX FFE symbol length: {len(tx_ffe_symbol)} symbols")

        return tx_ffe_output, tx_ffe_symbol, self.tx_ffe.get_taps()
    
    def _create_and_apply_channel(self, signal: np.ndarray, tx_symbol: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create channel and apply to signal.
        
        Parameters
        ----------
        signal : ndarray 
            Input signal
        tx_symbol : ndarray
            Transmitted symbols
            
        Returns
        -------
        rx_signal : ndarray
            Signal after channel
        channel_response : ndarray
            Channel frequency response
        channel_impulse : ndarray
            Channel impulse response
        """
        self._print("\n" + "="*70)
        self._print("Step 3: Applying SerDes Channel")
        self._print("="*70)
        
        # Create channel
        self.channel = SerdesChannel(
            symbol_rate=self.config.symbol_rate,
            samples_per_symbol=self.config.samples_per_symbol,
            length=self.config.channel_length,
            z0=self.config.channel_z0,
            eps_r=self.config.channel_eps_r
        )
        
        # Set parasitics
        self.channel.set_parasitics(
            cap_source=self.config.channel_cap_source,
            cap_term=self.config.channel_cap_term
        )
        
        # Set loss parameters
        self.channel.set_loss_parameters(
            theta_0=self.config.channel_theta_0,
            k_r=self.config.channel_k_r,
            RDC=self.config.channel_rdc
        )
        
        # Compute and apply channel
        channel_response, channel_impulse, _ = self.channel.compute_channel()
        delayed_sample, _, _ = self.channel.calculate_delay(method='group_delay')
        rx_signal = self.channel.apply_channel(signal)
        
        self._print(f"  Channel length: {self.config.channel_length*1000:.1f} mm")
        self._print(f"  RX signal length: {len(rx_signal)} samples")
        
        return rx_signal, channel_response, channel_impulse
    
    def _add_noise(self, signal: np.ndarray) -> np.ndarray:
        """
        Add AWGN noise to signal.
        
        Parameters
        ----------
        signal : ndarray
            Input signal
            
        Returns
        -------
        noisy_signal : ndarray
            Signal with added noise
        """
        self._print("\n" + "="*70)
        self._print("Step 4: Adding AWGN Noise")
        self._print("="*70)
        
        noisy_signal = add_noise(signal, snr_db=self.config.snr_db, seed=self.config.noise_seed)
        
        self._print(f"  SNR: {self.config.snr_db:.1f} dB")
        
        return noisy_signal

    def _apply_nonlinearity(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply nonlinearity to signal.
        
        Parameters
        ----------
        signal : ndarray
            Input signal
            
        Returns
        -------
        nonlinear_signal : ndarray
            Signal with nonlinearity
        """
        self._print("\n" + "="*70)
        self._print("Step 6: Applying Nonlinearity")
        self._print("="*70)
        
        nonlinear_signal = add_nonlinearity_distortion(
            signal, 
            gain_compression=self.config.non_linearity_gain_compression,
            third_order=self.config.non_linearity_third_order
        )
        
        self._print(f"  Nonlinearity: {self.config.non_linearity_gain_compression:.2f}, {self.config.non_linearity_third_order:.2f}")
        self._print(f"  Nonlinear signal length: {len(nonlinear_signal)} samples")
        
        return nonlinear_signal
    
    def _apply_adaptive_ctle(self, rx_signal: np.ndarray, tx_symbols: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Apply adaptive CTLE and return results.
        
        Parameters
        ----------
        rx_signal : ndarray
            Received signal
        tx_symbols : ndarray
            Transmitted symbol indices
            
        Returns
        -------
        ctle_output : ndarray
            CTLE output signal (data portion only)
        tx_symbols_data : ndarray
            TX symbols aligned with CTLE output
        ctle_info : dict
            CTLE adaptation information
        """
        self._print("\n" + "="*70)
        self._print("Step 5: Applying Adaptive CTLE")
        self._print("="*70)
        
        # Create adaptive CTLE
        self.adaptive_ctle = AdaptiveCTLE(
            symbol_rate=self.config.symbol_rate,
            samples_per_symbol=self.config.samples_per_symbol,
            n_levels=self.config.n_levels,
            n_configs=self.config.n_ctle_configs
        )
        
        self._print(f"  Created {self.config.n_ctle_configs} CTLE configurations")
        
        # Use adaptation symbols for CTLE adaptation
        adapt_length = self.config.adaptation_symbols * self.config.samples_per_symbol
        adapt_signal = rx_signal[:adapt_length]

        self._print(f"  Adaptation signal length: {len(adapt_signal)} samples")
        self._print(f"  Adaptation symbols: {self.config.adaptation_symbols} symbols")
        
        # Run adaptation
        adapt_results = self.adaptive_ctle.adapt(
            rx_signal=adapt_signal,
            tx_symbols=tx_symbols[:self.config.adaptation_symbols],
            verbose=self.config.verbose
        )
        
        self._print(f"  Best CTLE config: {self.adaptive_ctle.best_config_id}")
        self._print(f"  Best SNR: {self.adaptive_ctle.best_snr_db:.2f} dB")
        self._print(f"  Best sampling offset: {self.adaptive_ctle.best_sampling_offset}")
        
        # Apply best CTLE to data portion
        data_start_sample = adapt_length
        data_start_symbol = self.config.adaptation_symbols
        
        ctle_data = self.adaptive_ctle.apply_ctle_with_best_config(rx_signal[data_start_sample:])
        
        # Calculate channel delay
        channel_delay, channel_delay_symbols, delay_time = self.channel.calculate_delay(method='group_delay')
        self._print(f"  Channel delay (estimated): {channel_delay} samples ({channel_delay_symbols:.2f} symbols, {delay_time*1e12:.2f} ps)")
        
        # Estimate TX symbol alignment
        estimated_delay_symbols = int(np.round(channel_delay_symbols))
        tx_symbols_data = tx_symbols[data_start_symbol - estimated_delay_symbols:]
        
        # Ensure alignment
        min_length = min(len(ctle_data) // self.config.samples_per_symbol, len(tx_symbols_data))
        tx_symbols_data = tx_symbols_data[:min_length]
        ctle_data = ctle_data[:min_length * self.config.samples_per_symbol]
        
        self._print(f"  CTLE data output: {len(ctle_data)} samples")
        self._print(f"  TX symbols (estimated alignment): {len(tx_symbols_data)} symbols")
        
        # Gather CTLE info
        ctle_info = {
            'best_config_id': self.adaptive_ctle.best_config_id,
            'best_snr_db': self.adaptive_ctle.best_snr_db,
            'best_sampling_offset': self.adaptive_ctle.best_sampling_offset,
            'channel_delay_samples': channel_delay,
            'channel_delay_symbols': channel_delay_symbols,
            'channel_delay_time': delay_time,
            'adaptation_output': adapt_results['adaptation_output_signal'],
        }
        
        return ctle_data, tx_symbols_data, ctle_info
    
    def _normalize_and_convert_adc(self, ctle_output: np.ndarray, sampling_offset: int) -> Tuple[np.ndarray, Dict]:
        """
        Normalize signal and convert through ADC.
        
        Parameters
        ----------
        ctle_output : ndarray
            CTLE output signal
        sampling_offset : int
            Best sampling offset from CTLE
            
        Returns
        -------
        adc_output : ndarray
            ADC output (one sample per symbol)
        adc_info : dict
            ADC conversion information
        """
        self._print("\n" + "="*70)
        self._print("Step 6: Normalization and ADC Conversion")
        self._print("="*70)
        
        # Normalize to target RMS
        rms_ctle = np.sqrt(np.mean(ctle_output**2))
        ctle_normalized = ctle_output * (self.config.adc_target_rms / rms_ctle)
        
        self._print(f"  CTLE RMS: {rms_ctle:.4f} V")
        self._print(f"  Normalized RMS: {self.config.adc_target_rms:.4f} V")
        
        # Create and apply ADC
        self.adc = ADC_n_bits(
            n_bits=self.config.adc_n_bits,
            full_scale_voltage=self.config.adc_full_scale_voltage,
            samples_per_symbol=self.config.samples_per_symbol,
            sampling_offset=sampling_offset,
            target_rms=self.config.adc_target_rms
        )
        
        adc_output, adc_info = self.adc.convert(ctle_normalized, auto_normalize=False)
        
        if self.config.verbose:
            self.adc.print_info()
        
        self._print(f"  ADC output: {len(adc_output)} samples")
        
        return adc_output, adc_info
    
    def _slice_pam(self, values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        """
        Slice values to PAM symbols based on n_levels.
        
        Parameters
        ----------
        values : ndarray
            Signal values
        thresholds : ndarray
            Thresholds for PAM slicing (already scaled by alpha from get_thresholds)
            
        Returns
        -------
        symbols : ndarray
            Symbol indices
        """
        # Note: thresholds are already scaled by alpha in get_thresholds()
        n_levels = self.config.n_levels
        symbols = np.zeros(len(values), dtype=int)
        
        if n_levels == 4:
            # 3 thresholds for PAM-4
            symbols[values < thresholds[0]] = 0
            symbols[(values >= thresholds[0]) & (values < thresholds[1])] = 1
            symbols[(values >= thresholds[1]) & (values < thresholds[2])] = 2
            symbols[values >= thresholds[2]] = 3
        elif n_levels == 6:
            # 5 thresholds for PAM-6
            symbols[values < thresholds[0]] = 0
            symbols[(values >= thresholds[0]) & (values < thresholds[1])] = 1
            symbols[(values >= thresholds[1]) & (values < thresholds[2])] = 2
            symbols[(values >= thresholds[2]) & (values < thresholds[3])] = 3
            symbols[(values >= thresholds[3]) & (values < thresholds[4])] = 4
            symbols[values >= thresholds[4]] = 5
        elif n_levels == 8:
            # 7 thresholds for PAM-8
            symbols[values < thresholds[0]] = 0
            symbols[(values >= thresholds[0]) & (values < thresholds[1])] = 1
            symbols[(values >= thresholds[1]) & (values < thresholds[2])] = 2
            symbols[(values >= thresholds[2]) & (values < thresholds[3])] = 3
            symbols[(values >= thresholds[3]) & (values < thresholds[4])] = 4
            symbols[(values >= thresholds[4]) & (values < thresholds[5])] = 5
            symbols[(values >= thresholds[5]) & (values < thresholds[6])] = 6
            symbols[values >= thresholds[6]] = 7
        
        return symbols
    
    # Backward compatibility alias
    def _slice_pam6(self, values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        """Alias for _slice_pam for backward compatibility."""
        return self._slice_pam(values, thresholds)

    def _align_sequences(self, detected_symbols: np.ndarray, tx_symbols: np.ndarray, max_search: int = 100) -> Tuple[int, float]:
        """
        Find optimal alignment between detected and transmitted symbols using cross-correlation.
        
        Parameters
        ----------
        detected_symbols : ndarray
            Detected symbol indices
        tx_symbols : ndarray
            Transmitted symbol indices
        max_search : int
            Maximum search range for alignment
            
        Returns
        -------
        offset : int
            Number of samples to shift tx_symbols to align with detected_symbols
        correlation : float
            Peak correlation value
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

    def _apply_rx_ffe(self, rx_signal: np.ndarray, tx_symbols: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        # Apply RX FFE
        self._print("\n" + "="*70)
        self._print(f"Step 3: Applying RX FFE (PAM-{self.config.n_levels})")
        self._print("="*70)
        
        rx_ffe = AdaptiveFFE(
            n_taps=self.config.rx_ffe_n_taps,
            n_precursor=self.config.rx_ffe_n_precursor,
            n_postcursor=self.config.rx_ffe_n_postcursor,
            algorithm=self.config.rx_ffe_algorithm,
            mu=self.config.rx_ffe_mu,
            adc_rms=np.sqrt(np.mean(rx_signal**2))
        )   

        
        adc_output = rx_signal
        tx_symbols_rx_ffe = tx_symbols

        adc_output_rms = np.sqrt(np.mean(adc_output**2))
        
        # Compute symbol offset and scaling based on n_levels
        n_levels = self.config.n_levels
        alpha = self.config.pam_alpha
        
        if n_levels == 4:
            # PAM-4: symbols 0,1,2,3 -> levels -3,-1,1,3 (normalized: -1.5,-0.5,0.5,1.5)
            symbol_offset = 1.5  # Center offset for PAM-4 symbols (0-3)
            symbol_scale =1.0
            base_levels = np.array([-1.5, -0.5, 0.5, 1.5])
            base_thresholds = np.array([-1.0, 0.0, 1.0])
        elif n_levels == 6:
            # PAM-6: symbols 0,1,2,3,4,5 -> levels -5,-3,-1,1,3,5 (normalized by /3)
            symbol_offset = 2.5  # Center offset for PAM-6 symbols (0-5)
            symbol_scale =2.0/3.0
            base_levels = np.array([-5, -3, -1, 1, 3, 5]) / 3.0
            base_thresholds = np.array([-4, -2, 0, 2, 4]) / 3.0
        elif n_levels == 8:
            # PAM-8: symbols 0,1,2,3,4,5,6,7 -> levels -7,-5,-3,-1,1,3,5,7 (normalized by /4)
            symbol_offset = 3.5  # Center offset for PAM-8 symbols (0-7)
            symbol_scale =1.0/2.0
            base_levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7]) / 4.0
            base_thresholds = np.array([-6, -4, -2, 0, 2, 4, 6]) / 4.0
        else:
            raise ValueError(f"Unsupported n_levels: {n_levels}")
        
        tx_symbol_data_rms = np.sqrt(np.mean((tx_symbols_rx_ffe - symbol_offset)**2))
        tx_symbols_data_rx_ffe = (tx_symbols_rx_ffe - symbol_offset) * symbol_scale * (adc_output_rms / tx_symbol_data_rms)
        tx_symbols_data_rx_ffe = np.concatenate([np.array([0, 0, 0]), tx_symbols_data_rx_ffe])

        
        # Train RX FFE
        rx_ffe_training_output, rx_ffe_training_errors = rx_ffe.process_and_adapt(adc_output[:self.config.rx_ffe_training_symbols], 
            desired=tx_symbols_data_rx_ffe[:self.config.rx_ffe_training_symbols], 
            save_history=True)
        rx_ffe.print_info()
        fig1 = rx_ffe.plot_convergence()
        fig1.savefig('./Serdes_v9_pam6/outputs/rx_ffe_convergence.png')
       
            
        rx_ffe_output = rx_ffe.equalize(rx_signal[self.config.rx_ffe_training_symbols:])
        rx_ffe_output_rms = np.sqrt(np.mean(rx_ffe_output**2))
        self._print(f"  RX FFE output RMS: {rx_ffe_output_rms:.4f} V")

        # PAM levels and thresholds scaled by RMS and alpha
        pam_levels_normalized = base_levels * rx_ffe_output_rms * alpha
        thresholds_normalized = base_thresholds * rx_ffe_output_rms * alpha

        # Step 8: Final alignment
        rx_ffe_symbols = self._slice_pam(rx_ffe_output, thresholds_normalized)
        tx_symbols_data = tx_symbols[self.config.rx_ffe_training_symbols:]
        min_length = min(len(tx_symbols_data), len(rx_ffe_symbols))
        alignment_offset, alignment_corr = self._align_sequences(rx_ffe_symbols[:min_length], tx_symbols_data[:min_length])

        self._print(f"\n  Alignment offset: {alignment_offset} symbols")
        self._print(f"  Alignment correlation: {alignment_corr:.4f}")
        
        # Apply alignment
        if alignment_offset < 0:
            # Shift TX symbols earlier (detected symbols are delayed)
            tx_symbols_aligned = tx_symbols_data[-alignment_offset:]
            rx_ffe_output_aligned = rx_ffe_output[:len(tx_symbols_aligned)]
            rx_ffe_symbols_aligned = rx_ffe_symbols[:len(tx_symbols_aligned)]
        else:
            # Shift TX symbols later (detected symbols are early)
            tx_symbols_aligned = tx_symbols_data[:len(tx_symbols_data) - alignment_offset]
            rx_ffe_output_aligned = rx_ffe_output[alignment_offset:alignment_offset + len(tx_symbols_aligned)]
            rx_ffe_symbols_aligned = rx_ffe_symbols[alignment_offset:alignment_offset + len(tx_symbols_aligned)]
        
        min_length = min(len(tx_symbols_aligned), len(rx_ffe_output_aligned))
        tx_symbols_aligned = tx_symbols_aligned[:min_length]
        rx_ffe_output_aligned = rx_ffe_output_aligned[:min_length]
        rx_ffe_symbols_aligned = rx_ffe_symbols_aligned[:min_length]

        rx_ffe_errors = np.sum(rx_ffe_symbols_aligned != tx_symbols_aligned)
        rx_ffe_ser = rx_ffe_errors / len(rx_ffe_symbols_aligned) if len(rx_ffe_symbols_aligned) > 0 else 0

        self._print(f"  RX FFE SER: {rx_ffe_ser:.4e}, Errors: {rx_ffe_errors}/{len(rx_ffe_symbols_aligned)}")

        rx_ffe_info = {
            'rx_ffe_best_offset': alignment_offset,
            'rx_ffe_best_corr': alignment_corr,
            'rx_ffe_ser': rx_ffe_ser,
            'rx_ffe_errors': rx_ffe_errors,
            'rx_ffe_test_length': len(rx_ffe_symbols_aligned),
            'rx_ffe_symbols': rx_ffe_symbols_aligned,
            'rx_ffe_output': rx_ffe_output_aligned,
            'rx_ffe_tx_symbols': tx_symbols_aligned,
            'rx_ffe_taps': rx_ffe.get_taps()
        }
        
        return rx_ffe_output_aligned, rx_ffe_symbols_aligned, tx_symbols_aligned, rx_ffe_info
        


    def run_ctle_adc(self) -> Dict:
        """
        Run complete PAM SerDes simulation.
        
        Returns
        -------
        results : dict
            Dictionary containing all simulation results:
            - tx_symbols: Transmitted symbol indices (aligned)
            - tx_signal: Oversampled TX signal
            - tx_ffe_output: TX FFE output
            - rx_signal: Channel output with noise
            - ctle_output: CTLE output (data portion)
            - adc_output: ADC output (aligned)
            - ctle_best_config: Best CTLE configuration ID
            - ctle_best_snr_db: Best CTLE SNR
            - ctle_best_offset: Best sampling offset
            - channel_delay_samples: Channel delay in samples
            - channel_delay_symbols: Channel delay in symbols
            - adc_info: ADC conversion information
            - config: Copy of configuration used
        """
        if self.config.verbose:
            self.config.print_summary()
        
        # Step 1: Generate PAM signal
        tx_symbols, tx_signal = self._generate_pam_signal()
        
        # Step 2: Apply TX FFE
        tx_ffe_output, tx_ffe_symbols, tx_ffe_taps = self._apply_tx_ffe(tx_signal, tx_symbols)
        
        # Step 3: Apply channel
        rx_signal, channel_response, channel_impulse = self._create_and_apply_channel(tx_ffe_output, tx_ffe_symbols)
        
        # Step 4: Add noise
        rx_signal_noisy = self._add_noise(rx_signal)
        
        # Step 5: Apply adaptive CTLE
        ctle_output, tx_symbols_data, ctle_info = self._apply_adaptive_ctle(rx_signal_noisy, tx_ffe_symbols)
        
        # Step 6: Normalize and convert through ADC
        adc_output, adc_info = self._normalize_and_convert_adc(ctle_output, ctle_info['best_sampling_offset'])
        
        # Step 7: Add noise and nonlinearity (optional)
        if self.config.non_linearity_gain_compression > 0 or self.config.non_linearity_third_order > 0:
            adc_output_noisy = self._add_noise(adc_output)
            adc_output_final = self._apply_nonlinearity(adc_output_noisy)
        else:
            adc_output_final = adc_output
        
        adc_rms = np.sqrt(np.mean(adc_output_final**2))

        # Get PAM thresholds based on n_levels
        thresholds_normalized = self.pam_gen.get_thresholds(adc_rms)

        # Step 8: Final alignment
        adc_symbols = self._slice_pam(adc_output_final, thresholds_normalized)
        alignment_offset, alignment_corr = self._align_sequences(adc_symbols, tx_symbols_data)
        
        self._print(f"\n  Alignment offset: {alignment_offset} symbols")
        self._print(f"  Alignment correlation: {alignment_corr:.4f}")
        
        # Apply alignment
        if alignment_offset < 0:
            tx_symbols_aligned = tx_symbols_data[-alignment_offset:]
            adc_output_aligned = adc_output_final[:len(tx_symbols_aligned)]
            adc_symbols_aligned = adc_symbols[:len(tx_symbols_aligned)]
        else:
            tx_symbols_aligned = tx_symbols_data[:len(tx_symbols_data) - alignment_offset]
            adc_output_aligned = adc_output_final[alignment_offset:alignment_offset + len(tx_symbols_aligned)]
            adc_symbols_aligned = adc_symbols[alignment_offset:alignment_offset + len(tx_symbols_aligned)]
        
        # Calculate SER
        errors = np.sum(tx_symbols_aligned != adc_symbols_aligned)
        ser = errors / len(tx_symbols_aligned) if len(tx_symbols_aligned) > 0 else 0
        
        print(f"  Quick check - Symbol errors: {errors}/{len(tx_symbols_aligned)} (SER: {ser:.4e})")
        
        # Compile results
        self.results = {
            # Signals
            'tx_symbols': tx_symbols_aligned,
            'tx_signal': tx_signal,
            'tx_ffe_output': tx_ffe_output,
            'tx_ffe_taps': tx_ffe_taps,
            'tx_ffe_symbols': tx_ffe_symbols,
            'channel_response': channel_response,
            'channel_impulse': channel_impulse,
            'rx_signal': rx_signal_noisy,
            'ctle_output': ctle_output,
            'adc_output': adc_output_aligned,
            'adc_symbols': adc_symbols_aligned,
            
            # CTLE information
            'ctle_best_config': ctle_info['best_config_id'],
            'ctle_best_snr_db': ctle_info['best_snr_db'],
            'ctle_best_offset': ctle_info['best_sampling_offset'],
            'channel_delay_samples': ctle_info['channel_delay_samples'],
            'channel_delay_symbols': ctle_info['channel_delay_symbols'],
            'channel_delay_time': ctle_info['channel_delay_time'],
            
            # ADC information
            'adc_info': adc_info,
            'adc_rms': adc_rms,
            
            # Configuration
            'config': self.config.to_dict(),
            'n_levels': self.config.n_levels,
        }
        
        self._print("\n" + "="*70)
        self._print(f"PAM-{self.config.n_levels} SerDes Simulation Complete!")
        self._print("="*70)
        
        return self.results
    
    def run_ctle_adc_rx_ffe(self) -> Dict:
        """
        Run RX FFE simulation for PAM-4, PAM-6, or PAM-8.
        
        Returns
        -------
        results : dict
            Dictionary containing all simulation results:
        """
        if self.config.verbose:
            self.config.print_summary()

        # Step 1: Generate PAM signal (PAM-4, PAM-6, or PAM-8)
        tx_symbols, tx_signal = self._generate_pam_signal()
        
        # Step 2: Apply TX FFE
        tx_ffe_output, tx_ffe_symbols, tx_ffe_taps = self._apply_tx_ffe(tx_signal, tx_symbols) 
        
        # Step 3: Apply channel
        rx_signal, channel_response, channel_impulse = self._create_and_apply_channel(tx_ffe_output, tx_ffe_symbols)
        
        # Step 4: Add noise
        rx_signal_noisy = self._add_noise(rx_signal)
        
        # Step 5: Apply adaptive CTLE
        ctle_output, tx_symbols_data, ctle_info = self._apply_adaptive_ctle(rx_signal_noisy, tx_ffe_symbols)
        
        # Step 6: Normalize and convert through ADC
        adc_output, adc_info = self._normalize_and_convert_adc(ctle_output, ctle_info['best_sampling_offset'])
        
        # Step 7: Add noise and nonlinearity (optional - can be disabled for clean alignment)
        if self.config.non_linearity_gain_compression > 0 or self.config.non_linearity_third_order > 0:
            adc_output_noisy = self._add_noise(adc_output)
            adc_output_final = self._apply_nonlinearity(adc_output_noisy)
        else:
            adc_output_final = adc_output
        
        adc_rms = np.sqrt(np.mean(adc_output_final**2))

        # Get PAM thresholds based on n_levels (already scaled by alpha)
        thresholds_normalized = self.pam_gen.get_thresholds(adc_rms)

        # Step 8: Final alignment
        adc_symbols = self._slice_pam(adc_output_final, thresholds_normalized)
        alignment_offset, alignment_corr = self._align_sequences(adc_symbols, tx_symbols_data)
        
        self._print(f"\n  Alignment offset: {alignment_offset} symbols")
        self._print(f"  Alignment correlation: {alignment_corr:.4f}")
        
        # Apply alignment
        if alignment_offset < 0:
            # Shift TX symbols earlier (detected symbols are delayed)
            tx_symbols_aligned = tx_symbols_data[-alignment_offset:]
            adc_output_aligned = adc_output_final[:len(tx_symbols_aligned)]
            adc_symbols_aligned = adc_symbols[:len(tx_symbols_aligned)]
        else:
            # Shift TX symbols later (detected symbols are early)
            tx_symbols_aligned = tx_symbols_data[:len(tx_symbols_data) - alignment_offset]
            adc_output_aligned = adc_output_final[alignment_offset:alignment_offset + len(tx_symbols_aligned)]
            adc_symbols_aligned = adc_symbols[alignment_offset:alignment_offset + len(tx_symbols_aligned)]
        
        # Calculate simple SER for verification
        errors = np.sum(tx_symbols_aligned != adc_symbols_aligned)
        ser = errors / len(tx_symbols_aligned) if len(tx_symbols_aligned) > 0 else 0
        
        print(f"  Quick check - Symbol errors: {errors}/{len(tx_symbols_aligned)} (SER: {ser:.4e})")
           
        # Step 9: Apply RX FFE
        rx_ffe_output, rx_ffe_symbols, tx_symbols_rx_ffe, rx_ffe_info = self._apply_rx_ffe(adc_output_aligned, tx_symbols_aligned)


        # Compile results
        self.results = {
            # Signals
            'tx_symbols': tx_symbols_aligned,
            'tx_signal': tx_signal,
            'tx_ffe_output': tx_ffe_output,
            'tx_ffe_taps': tx_ffe_taps,
            'tx_ffe_symbols': tx_ffe_symbols,
            'channel_response': channel_response,
            'channel_impulse': channel_impulse,
            'rx_signal': rx_signal_noisy,
            'ctle_output': ctle_output,
            'adc_output': adc_output_aligned,
            'adc_symbols': adc_symbols_aligned,
            
            # CTLE information
            'ctle_best_config': ctle_info['best_config_id'],
            'ctle_best_snr_db': ctle_info['best_snr_db'],
            'ctle_best_offset': ctle_info['best_sampling_offset'],
            'channel_delay_samples': ctle_info['channel_delay_samples'],
            'channel_delay_symbols': ctle_info['channel_delay_symbols'],
            'channel_delay_time': ctle_info['channel_delay_time'],
            
            # ADC information
            'adc_info': adc_info,
            'adc_rms': adc_rms,

            # RX FFE information
            'rx_ffe_signal': rx_ffe_output,
            'rx_ffe_symbols': rx_ffe_symbols,
            'rx_ffe_tx_symbols': tx_symbols_rx_ffe,
            'rx_ffe_ser': rx_ffe_info['rx_ffe_ser'],
            'rx_ffe_errors': rx_ffe_info['rx_ffe_errors'],
            'rx_ffe_test_length': rx_ffe_info['rx_ffe_test_length'],
            'rx_ffe_best_corr': rx_ffe_info['rx_ffe_best_corr'],
            'rx_ffe_best_offset': rx_ffe_info['rx_ffe_best_offset'],
            'rx_ffe_taps': rx_ffe_info['rx_ffe_taps'],

            # Configuration
            'config': self.config.to_dict(),
            'n_levels': self.config.n_levels,
        }
        
        return self.results


    def print_summary(self):
        """Print a summary of the simulation results."""
        if self.results is None:
            print("No simulation results available. Run system.run_ctle_adc() first.")
            return
        
        n_levels = self.config.n_levels
        bits_per_symbol = np.log2(n_levels)
        data_rate = self.config.symbol_rate * bits_per_symbol / 1e9
        
        print("\n" + "="*70)
        print(f"PAM-{n_levels} SerDes Simulation Results Summary")
        print("="*70)
        print(f"  Modulation: PAM-{n_levels} ({n_levels} levels)")
        print(f"  Data Rate: {data_rate:.2f} Gb/s")
        print(f"  Channel Length: {self.config.channel_length*1000:.1f} mm")
        print(f"  SNR: {self.config.snr_db:.1f} dB")
        print(f"\nCTLE Results:")
        print(f"  Best Config: {self.results['ctle_best_config']}")
        print(f"  Best SNR: {self.results['ctle_best_snr_db']:.2f} dB")
        print(f"  Sampling Offset: {self.results['ctle_best_offset']}")
        print(f"  Channel Delay: {self.results['channel_delay_samples']} samples "
              f"({self.results['channel_delay_symbols']:.2f} symbols)")
        print(f"\nData Lengths:")
        print(f"  TX Symbols: {len(self.results['tx_symbols'])}")
        print(f"  ADC Output: {len(self.results['adc_output'])}")
        print("="*70)


# Example usage and testing
if __name__ == "__main__":
    print("PAM SerDes System Simulator - Example Usage\n")
    
    # Example 1: PAM-4 simulation
    print("="*70)
    print("Example 1: PAM-4 SerDes Simulation")
    print("="*70)
    
    config_pam4 = ConfigSerdesSystemPAM6(
        n_levels=4,
        channel_length=0.20,
        snr_db=26,  # PAM-4 needs less SNR
        total_symbols=50000,
        adaptation_symbols=32000
    )
    
    system_pam4 = SerdesSystemPAM6(config_pam4)
    results_pam4 = system_pam4.run_ctle_adc()
    system_pam4.print_summary()
    
    # Example 2: PAM-6 simulation
    print("\n" + "="*70)
    print("Example 2: PAM-6 SerDes Simulation")
    print("="*70)
    
    config_pam6 = ConfigSerdesSystemPAM6(
        n_levels=6,
        channel_length=0.20,
        snr_db=28,  # PAM-6 needs moderate SNR
        total_symbols=50000,
        adaptation_symbols=32000
    )
    
    system_pam6 = SerdesSystemPAM6(config_pam6)
    results_pam6 = system_pam6.run_ctle_adc()
    system_pam6.print_summary()
    
    # Example 3: PAM-8 simulation
    print("\n" + "="*70)
    print("Example 3: PAM-8 SerDes Simulation")
    print("="*70)
    
    config_pam8 = ConfigSerdesSystemPAM6(
        n_levels=8,
        channel_length=0.20,
        snr_db=30,  # PAM-8 needs higher SNR
        total_symbols=50000,
        adaptation_symbols=32000
    )
    
    system_pam8 = SerdesSystemPAM6(config_pam8)
    results_pam8 = system_pam8.run_ctle_adc()
    system_pam8.print_summary()
    
    print("\n" + "="*70)
    print("All Simulations Complete")
    print("="*70)

