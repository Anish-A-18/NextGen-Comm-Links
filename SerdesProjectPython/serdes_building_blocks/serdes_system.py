#!/usr/bin/env python3
"""
SerDes System Simulator
=======================

Provides a clean, configurable interface for running complete SerDes simulations
from PAM4 generation through TX FFE, channel, CTLE, and ADC.

Classes:
- ConfigSerdesSystem: Configuration dataclass for system parameters
- SerdesSystem: Main simulation class that runs the complete signal chain

Author: Anish Anand
Date: November 15, 2025
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List
import sys
from pathlib import Path

# Local sibling imports (serdes_building_blocks/)
_BB = Path(__file__).resolve().parent
if str(_BB) not in sys.path:
    sys.path.insert(0, str(_BB))

from serdes_channel import SerdesChannel
from pam4_generator import PAM4Generator, add_noise
from adaptive_ctle import AdaptiveCTLE  # Using original version (without delay compensation)
from tx_ffe import TX_FFE
from adc_n_bits import ADC_n_bits, add_nonlinearity_distortion
from adaptive_ffe_2 import AdaptiveFFE


@dataclass
class ConfigSerdesSystem:
    """
    Configuration class for SerDes system parameters.
    
    All system parameters are defined here and can be easily modified
    to run different configurations.
    """
    
    # Symbol rate and sampling
    symbol_rate: float = 64e9  # Hz (64 Gbaud for PCIe 7.0)
    samples_per_symbol: int = 32
    
    # Channel parameters
    channel_length: float = 0.20  # meters (200 mm)
    channel_z0: float = 50.0  # Ohms
    channel_eps_r: float = 4.9  # Dielectric constant
    channel_cap_source: float = 100e-15  # Farads
    channel_cap_term: float = 150e-15  # Farads
    channel_theta_0: float = 0.015
    channel_k_r: float = 120
    channel_rdc: float = 0.0002
    
    # Signal generation
    total_symbols: int = 32000 + 50000 + 50000 + 50000 # Total symbols to generate
    adaptation_symbols: int = 32000  # Symbols used for CTLE adaptation
    pam4_seed: int = 42  # Random seed for reproducibility
    
    # TX FFE configuration
    tx_ffe_taps: List[float] = field(default_factory=lambda: [0.083, -0.208, 0.709, 0])
    
    # Noise configuration
    snr_db: float = 25.0  # Signal-to-noise ratio in dB
    noise_seed: int = 123
    
    # CTLE configuration
    n_ctle_configs: int = 16  # Number of CTLE configurations to test
    
    # ADC configuration
    adc_n_bits: int = 6
    adc_full_scale_voltage: float = 2.0
    adc_target_rms: float = 0.5

    # Nonlinearity configuration
    non_linearity_gain_compression: float = 0.1
    non_linearity_third_order: float = 0.05

    # RX FFE configuration
    rx_ffe_algorithm: str = 'lms'
    rx_ffe_mu: float = 0.01
    rx_ffe_training_symbols: int = 50000
    rx_ffe_n_taps: int = 8
    rx_ffe_n_precursor: int = 3
    rx_ffe_n_postcursor: int = rx_ffe_n_taps - rx_ffe_n_precursor - 1
    rx_ffe_lambda: float = 0.95
    rx_ffe_taps: List[float] =  field(default_factory=lambda: [np.zeros(3), 1, np.zeros(4)])

    
    # Output control
    verbose: bool = True  # Print detailed progress information
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.adaptation_symbols >= self.total_symbols:
            raise ValueError(f"adaptation_symbols ({self.adaptation_symbols}) must be < total_symbols ({self.total_symbols})")
        if self.samples_per_symbol < 1:
            raise ValueError(f"samples_per_symbol must be >= 1, got {self.samples_per_symbol}")
        if self.adc_n_bits < 1 or self.adc_n_bits > 16:
            raise ValueError(f"adc_n_bits must be between 1 and 16, got {self.adc_n_bits}")
    
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
    
    def copy(self) -> 'ConfigSerdesSystem':
        """Create a deep copy of the configuration."""
        import copy
        return copy.deepcopy(self)
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
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
        print("="*70)
        print("SerDes System Configuration")
        print("="*70)
        print(f"  Symbol Rate: {self.symbol_rate/1e9:.1f} Gbaud")
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


class SerdesSystem:
    """
    Complete SerDes system simulator.
    
    Implements the full signal chain:
    1. PAM4 symbol generation
    2. Oversampling
    3. TX FFE (pre-emphasis)
    4. SerDes channel
    5. AWGN noise
    6. Adaptive CTLE
    7. Normalization
    8. ADC conversion
    9. Alignment of TX symbols with ADC output
    
    Usage:
        config = ConfigSerdesSystem(channel_length=0.20, snr_db=25)
        system = SerdesSystem(config)
        results = system.run()
        
        # Access results
        tx_symbols = results['tx_symbols']
        adc_output = results['adc_output']
        ctle_snr = results['ctle_best_snr_db']
    """
    
    def __init__(self, config: ConfigSerdesSystem):
        """
        Initialize SerDes system with configuration.
        
        Parameters
        ----------
        config : ConfigSerdesSystem
            System configuration object
        """
        self.config = config
        self.results = None
        
        # Components (will be initialized during run)
        self.channel = None
        self.pam4_gen = None
        self.tx_ffe = None
        self.adaptive_ctle = None
        self.adc = None
    
    def _print(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.config.verbose:
            print(message)
    
    def _generate_pam4_signal(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate PAM4 symbols and oversample.
        
        Returns
        -------
        tx_symbols : ndarray
            Transmitted symbols (0, 1, 2, 3)
        tx_signal : ndarray
            Oversampled transmitted signal
        """
        self._print("\n" + "="*70)
        self._print("Step 1: Generating PAM4 Signal")
        self._print("="*70)
        
        self.pam4_gen = PAM4Generator(seed=self.config.pam4_seed) 
        
        # Generate PAM4 levels: [-3, -1, +1, +3]
        tx_pam4_levels = self.pam4_gen.generate_random_symbols(self.config.total_symbols)
        
        # Map to symbols [0, 1, 2, 3]
        level_to_symbol = {-3: 0, -1: 1, 1: 2, 3: 3}
        tx_symbols = np.array([level_to_symbol[level] for level in tx_pam4_levels])
        
        # Oversample
        tx_signal = self.pam4_gen.oversample(tx_pam4_levels, self.config.samples_per_symbol)
        
        self._print(f"  Generated {self.config.total_symbols} symbols")
        self._print(f"  Symbol distribution: {np.bincount(tx_symbols)}")
        self._print(f"  TX signal length: {len(tx_signal)} samples")
        
        return tx_symbols, tx_signal
    
    def _apply_tx_ffe(self, tx_signal: np.ndarray, tx_symbol: np.ndarray) -> np.ndarray:
        """
        Apply transmitter FFE (pre-emphasis).
        
        Parameters
        ----------
        tx_signal : ndarray
            Input signal
            
        Returns
        -------
        tx_ffe_output : ndarray
            Equalized signal
        """
        self._print("\n" + "="*70)
        self._print("Step 2: Applying TX FFE")
        self._print("="*70)
        
        self.tx_ffe = TX_FFE(taps=self.config.tx_ffe_taps)
        tx_ffe_output = self.tx_ffe.equalize(tx_signal, samples_per_symbol=self.config.samples_per_symbol)
        
        self._print(f"  TX FFE taps: {self.tx_ffe.get_taps()}")
        self._print(f"  Output length: {len(tx_ffe_output)} samples")

        tx_ffe_symbol = np.concatenate([[0],tx_symbol])
        self._print(f"  TX FFE symbol length: {len(tx_ffe_symbol)} symbols")

        return tx_ffe_output, tx_ffe_symbol, self.tx_ffe.get_taps()
    
    def _create_and_apply_channel(self, signal: np.ndarray, tx_symbol: np.ndarray) -> np.ndarray:
        """
        Create channel and apply to signal.
        
        Parameters
        ----------
        signal : ndarray 
            Input signal
            
        Returns
        -------
        rx_signal : ndarray
            Signal after channel
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
        delayed_sample , _, _ = self.channel.calculate_delay(method='group_delay')
        rx_signal = self.channel.apply_channel(signal)

        #tx_symbol_ch = np.concatenate([np.zeros(int(delayed_symbol)),tx_symbol])
        
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
        """
        self._print("\n" + "="*70)
        self._print("Step 6: Applying Nonlinearity")
        self._print("="*70)
        
        nonlinear_signal = add_nonlinearity_distortion(signal, 
            gain_compression=self.config.non_linearity_gain_compression,
            third_order=self.config.non_linearity_third_order)
        
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
            Transmitted symbols
            
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
        
        # Create adaptive CTLE (without channel transfer function for delay compensation)
        self.adaptive_ctle = AdaptiveCTLE(
            symbol_rate=self.config.symbol_rate,
            samples_per_symbol=self.config.samples_per_symbol,
            n_configs=self.config.n_ctle_configs
        )
        
        self._print(f"  Created {self.config.n_ctle_configs} CTLE configurations")
        
        # Use adaptation symbols for CTLE adaptation
        adapt_length = self.config.adaptation_symbols * self.config.samples_per_symbol
        adapt_signal = rx_signal[:adapt_length]

        self._print(f"  Adaptation signal length: {len(adapt_signal)} samples")
        self._print(f"  Adaptation symbols: {self.config.adaptation_symbols} symbols")
        self._print(f"  Samples per symbol: {self.config.samples_per_symbol}")
        self._print(f"  Adaptation length: {adapt_length} samples")
        self._print(f"  RX signal length: {len(rx_signal)} samples")
        self._print(f"  TX symbols length: {len(tx_symbols)} symbols")
        
        # Run adaptation (tx_symbols is optional in original version)
        adapt_results = self.adaptive_ctle.adapt(
            rx_signal=adapt_signal,
            tx_symbols=tx_symbols[:self.config.adaptation_symbols],  # Optional: for improved SNR calculation
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
        
        # Estimate TX symbol alignment with CTLE output
        # Note: This is approximate since we don't have CTLE delay compensation in original version
        estimated_delay_symbols = int(np.round(channel_delay_symbols))
        tx_symbols_data = tx_symbols[data_start_symbol - estimated_delay_symbols:]
        
        # Ensure alignment
        min_length = min(len(ctle_data) // self.config.samples_per_symbol, len(tx_symbols_data))
        tx_symbols_data = tx_symbols_data[:min_length]
        ctle_data = ctle_data[:min_length * self.config.samples_per_symbol]
        
        self._print(f"  CTLE data output: {len(ctle_data)} samples")
        self._print(f"  TX symbols (estimated alignment): {len(tx_symbols_data)} symbols")
        self._print(f"  Note: Using channel delay estimate (CTLE delay not calculated)")
        
        # Gather CTLE info (original version doesn't have delay info)
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
    
    def _align_symbols_with_adc(self, tx_symbols: np.ndarray, adc_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ensure TX symbols and ADC output are aligned.
        
        Parameters
        ----------
        tx_symbols : ndarray
            TX symbols
        adc_output : ndarray
            ADC output
            
        Returns
        -------
        tx_symbols_aligned : ndarray
            TX symbols aligned with ADC
        adc_output_aligned : ndarray
            ADC output aligned with TX symbols
        """
        self._print("\n" + "="*70)
        self._print("Step 7: Final Alignment")
        self._print("="*70)
        
        min_length = min(len(adc_output), len(tx_symbols))
        
        if len(adc_output) != len(tx_symbols):
            self._print(f"  Warning: Length mismatch - ADC ({len(adc_output)}) vs TX symbols ({len(tx_symbols)})")
            self._print(f"  Truncating to {min_length} symbols")
        
        tx_symbols_aligned = tx_symbols[:min_length]
        adc_output_aligned = adc_output[:min_length]
        
        self._print(f"  Final aligned length: {min_length} symbols")
        
        # Calculate simple SER for verification
        errors = np.sum(tx_symbols_aligned != np.round(adc_output_aligned / self.config.adc_target_rms * 1.5 + 1.5).astype(int))
        ser = errors / len(tx_symbols_aligned) if len(tx_symbols_aligned) > 0 else 0
        
        self._print(f"  Quick check - Symbol errors: {errors}/{len(tx_symbols_aligned)} (SER: {ser:.4e})")
        
        return tx_symbols_aligned, adc_output_aligned

    def _slice_pam4(self, values, thresholds):
        """Slice values to PAM4 symbols."""
        thresholds = thresholds * .895
        symbols = np.zeros(len(values), dtype=int)
        symbols[values < thresholds[0]] = 0
        symbols[(values >= thresholds[0]) & (values < thresholds[1])] = 1
        symbols[(values >= thresholds[1]) & (values < thresholds[2])] = 2
        symbols[values >= thresholds[2]] = 3
        return symbols

    def _align_sequences(self, detected_symbols, tx_symbols, max_search=100):
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

    def _apply_rx_ffe(self, rx_signal: np.ndarray, tx_symbols: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        # Apply RX FFE
        self._print("\n" + "="*70)
        self._print("Step 3: Applying RX FFE")
        self._print("="*70)
        
        rx_ffe = AdaptiveFFE(
            n_taps=self.config.rx_ffe_n_taps,
            n_precursor=self.config.rx_ffe_n_precursor,
            n_postcursor=self.config.rx_ffe_n_postcursor,
            algorithm=self.config.rx_ffe_algorithm,
            mu=self.config.rx_ffe_mu,
            adc_rms=np.sqrt(np.mean(rx_signal**2))
        )   

        rx_ffe_training_signal = rx_signal[:self.config.rx_ffe_training_symbols]
        rx_ffe_training_rms = np.sqrt(np.mean(rx_ffe_training_signal**2))
        rx_ffe_training_symbols = tx_symbols[:self.config.rx_ffe_training_symbols]
       
        rx_ffe_training_symbols_rms = np.sqrt(np.mean(rx_ffe_training_symbols-1.5)**2)
        rx_ffe_training_symbols_data = (rx_ffe_training_symbols-1.5)*(rx_ffe_training_rms/rx_ffe_training_symbols_rms)
        rx_ffe_training_symbols_data = np.concatenate([np.array([0,0,0]), rx_ffe_training_symbols_data])
        min_length = min(len(rx_ffe_training_signal), len(rx_ffe_training_symbols_data))
        rx_ffe_training_signal = rx_ffe_training_signal[:min_length]
        rx_ffe_training_symbols_data = rx_ffe_training_symbols_data[:min_length]
        self._print(f"  RX FFE training signal: {len(rx_ffe_training_signal)} samples")
        self._print(f"  RX FFE training symbols: {len(rx_ffe_training_symbols_data)} symbols")

        # Train RX FFE
        rx_ffe_training_output, rx_ffe_training_errors = rx_ffe.process_and_adapt(rx_ffe_training_signal, 
            desired=rx_ffe_training_symbols_data, 
            save_history=True)
        rx_ffe.print_info()
        rx_ffe.plot_convergence()
            
        rx_ffe_output = rx_ffe.equalize(rx_signal[self.config.rx_ffe_training_symbols:])
        rx_ffe_output_rms = np.sqrt(np.mean(rx_ffe_output**2))
        self._print(f"  RX FFE output RMS: {rx_ffe_output_rms:.4f} V")

        # PAM4 levels and thresholds
        pam4_levels_normalized = np.array([-1.5, -0.5, 0.5, 1.5]) * rx_ffe_output_rms
        thresholds_normalized = np.array([-1.0, 0.0, 1.0]) * rx_ffe_output_rms

        # Step 8: Final alignment
        rx_ffe_symbols = self._slice_pam4(rx_ffe_output, thresholds_normalized)
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
            'rx_ffe_tx_symbols': tx_symbols_aligned
        }
        
        return rx_ffe_info
        
    
    def run_tx_ffe(self) -> Dict:
        """
        Run simulation up to TX FFE output (before channel).
        
        Useful for:
        - Analyzing TX signal before channel
        - Testing different TX FFE configurations
        - Pre-channel signal analysis
        
        Returns
        -------
        results : dict
            Dictionary containing:
            - tx_symbols: All transmitted symbols
            - tx_signal: Oversampled TX signal
            - tx_ffe_output: TX FFE output
            - config: Copy of configuration used
        """
        if self.config.verbose:
            self.config.print_summary()
        
        # Step 1: Generate PAM4 signal
        tx_symbols, tx_signal = self._generate_pam4_signal()
        
        # Step 2: Apply TX FFE
        tx_ffe_output, tx_ffe_symbol = self._apply_tx_ffe(tx_signal, tx_symbols)
        
        # Compile results
        self.results = {
            'tx_symbols': tx_symbols,
            'tx_signal': tx_signal,
            'tx_ffe_output': tx_ffe_output,
            'tx_ffe_symbol': tx_ffe_symbol,
            'config': self.config.to_dict(),
        }
        
        self._print("\n" + "="*70)
        self._print("Partial Simulation Complete (up to TX FFE)")
        self._print("="*70)
        
        return self.results
    
    def run_channel(self) -> Dict:
        """
        Run simulation up to RX signal with noise (before CTLE).
        
        Useful for:
        - Channel analysis
        - SNR measurement at RX input
        - Testing different channel/noise configurations
        
        Returns
        -------
        results : dict
            Dictionary containing:
            - tx_symbols: All transmitted symbols
            - tx_signal: Oversampled TX signal
            - tx_ffe_output: TX FFE output
            - rx_signal: Channel output with noise
            - config: Copy of configuration used
        """
        if self.config.verbose:
            self.config.print_summary()
        
        # Step 1: Generate PAM4 signal
        tx_symbols, tx_signal = self._generate_pam4_signal()
        
        # Step 2: Apply TX FFE
        tx_ffe_output, tx_ffe_symbols, tx_ffe_taps = self._apply_tx_ffe(tx_signal, tx_symbols)
        
        # Step 3: Apply channel
        rx_signal, channel_response, channel_impulse = self._create_and_apply_channel(tx_ffe_output, tx_ffe_symbols)
        
        # Step 4: Add noise
        rx_signal_noisy = self._add_noise(rx_signal)
        
        # Compile results
        self.results = {
            'tx_symbols': tx_symbols,
            'tx_signal': tx_signal,
            'tx_ffe_output': tx_ffe_output,
            'tx_ffe_symbols': tx_ffe_symbols,
            'channel_response': channel_response,
            'channel_impulse': channel_impulse,
            'rx_signal': rx_signal_noisy,
            'config': self.config.to_dict(),
        }
        
        self._print("\n" + "="*70)
        self._print("Partial Simulation Complete (up to RX with noise)")
        self._print("="*70)
        
        return self.results


    def run_ctle_adc(self) -> Dict: 
        """
        Run complete SerDes simulation.
        
        Returns
        -------
        results : dict
            Dictionary containing all simulation results:
            - tx_symbols: Transmitted symbols (aligned)
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
        
        # Step 1: Generate PAM4 signal
        tx_symbols, tx_signal = self._generate_pam4_signal()
        
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

        # PAM4 levels and thresholds
        pam4_levels_normalized = np.array([-1.5, -0.5, 0.5, 1.5]) * adc_rms
        thresholds_normalized = np.array([-1.0, 0.0, 1.0]) * adc_rms

        # Step 8: Final alignment
        adc_symbols = self._slice_pam4(adc_output_final, thresholds_normalized)
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
        }
        
        self._print("\n" + "="*70)
        self._print("SerDes Simulation Complete!")
        self._print("="*70)
        
        return self.results

    def run_rx_ffe(self, rx_signal: np.ndarray, tx_symbols: np.ndarray) -> Dict:
        """
        Run RX FFE simulation.
        
        Returns
        -------
        results : dict
            Dictionary containing all simulation results:
        """
        if self.config.verbose:
            self.config.print_summary()
        
            # Step 1: Apply RX FFE
        rx_ffe_info = self._apply_rx_ffe(rx_signal, tx_symbols)
        
        # Compile results
        self.results = {
            'rx_ffe_signal': rx_ffe_info['rx_ffe_output'],
            'rx_ffe_symbols': rx_ffe_info['rx_ffe_symbols'],
            'rx_ffe_tx_symbols': rx_ffe_info['rx_ffe_tx_symbols'],
            'rx_ffe_ser': rx_ffe_info['rx_ffe_ser'],
            'rx_ffe_errors': rx_ffe_info['rx_ffe_errors'],
            'rx_ffe_test_length': rx_ffe_info['rx_ffe_test_length'],
            'rx_ffe_best_corr': rx_ffe_info['rx_ffe_best_corr'],
            'rx_ffe_best_offset': rx_ffe_info['rx_ffe_best_offset'],
            'config': self.config.to_dict(),
        }
        
        return self.results
    
    def print_summary(self):
        """Print a summary of the simulation results."""
        if self.results is None:
            print("No simulation results available. Run system.run() first.")
            return
        
        print("\n" + "="*70)
        print("SerDes Simulation Results Summary")
        print("="*70)
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
    print("SerDes System Simulator - Example Usage\n")
    
    # Example 1: Single simulation with default config
    print("="*70)
    print("Example 1: Default Configuration")
    print("="*70)
    
    config = ConfigSerdesSystem(
        channel_length=0.20,
        snr_db=25,
        total_symbols=5000,
        adaptation_symbols=4000
    )
    
    system = SerdesSystem(config)
    results = system.run()
    system.print_summary()
    
    # Example 2: Multiple runs with different configurations
    print("\n\n" + "="*70)
    print("Example 2: Multiple Configurations")
    print("="*70)
    
    channel_lengths = [0.15, 0.20, 0.25]
    all_results = []
    
    for channel_len in channel_lengths:
        print(f"\n--- Running with channel length: {channel_len*1000:.1f} mm ---")
        
        # Update configuration
        config.update(channel_length=channel_len)
        
        # Create new system and run
        system = SerdesSystem(config)
        results = system.run()
        all_results.append(results)
        
        print(f"  CTLE Best SNR: {results['ctle_best_snr_db']:.2f} dB")
    
    print("\n" + "="*70)
    print("All Simulations Complete")
    print("="*70)

