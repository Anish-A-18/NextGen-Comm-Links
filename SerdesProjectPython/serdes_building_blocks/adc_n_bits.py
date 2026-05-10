"""
N-bit Analog-to-Digital Converter (ADC) for SerDes Receiver
=============================================================

Implements configurable ADC with:
- 1 to 10-bit resolution
- Programmable full-scale voltage
- Optimal sampling offset from CTLE
- Quantization and normalization

Author: Anish Anand
Date: November 8, 2025
"""

import numpy as np
from typing import Optional, Dict, Tuple


class ADC_n_bits:
    """
    N-bit Analog-to-Digital Converter for PAM4 SerDes receiver.
    
    The ADC samples the analog CTLE output at the optimal sampling offset
    and quantizes to n-bit digital representation.
    
    Parameters
    ----------
    n_bits : int
        ADC resolution in bits (1-10)
    full_scale_voltage : float
        Full-scale input range in Volts (±FS/2)
    samples_per_symbol : int
        Oversampling rate of input signal
    sampling_offset : int
        Optimal sampling offset within symbol period (from CTLE)
    target_rms : float
        Target RMS voltage for normalization (default: 0.5V)
        
    Attributes
    ----------
    n_levels : int
        Number of quantization levels (2^n_bits)
    quantization_step : float
        Voltage step between adjacent levels
    """
    
    def __init__(self,
                 n_bits: int = 6,
                 full_scale_voltage: float = 2.0,
                 samples_per_symbol: int = 32,
                 sampling_offset: int = 0,
                 target_rms: float = 0.5):
        """Initialize ADC."""
        
        if not (1 <= n_bits <= 10):
            raise ValueError("ADC resolution must be between 1 and 10 bits")
        
        if full_scale_voltage <= 0:
            raise ValueError("Full-scale voltage must be positive")
        
        self.n_bits = n_bits
        self.full_scale_voltage = full_scale_voltage
        self.samples_per_symbol = samples_per_symbol
        self.sampling_offset = sampling_offset
        self.target_rms = target_rms
        
        # Quantization parameters
        self.n_levels = 2**n_bits
        self.quantization_step = full_scale_voltage / (self.n_levels - 1)
        
        # Voltage range
        self.v_min = -full_scale_voltage / 2
        self.v_max = full_scale_voltage / 2
        
        # Normalization gain (computed after first signal)
        self.normalization_gain = 1.0
        
        # Statistics
        self.n_samples_processed = 0
        self.n_clipped = 0
        self.quantization_noise_power = 0.0
        
    def set_sampling_offset(self, offset: int):
        """
        Update sampling offset.
        
        Parameters
        ----------
        offset : int
            New sampling offset (0 to samples_per_symbol-1)
        """
        if not (0 <= offset < self.samples_per_symbol):
            raise ValueError(f"Offset must be 0 to {self.samples_per_symbol-1}")
        self.sampling_offset = offset
        
    def compute_normalization_gain(self, signal: np.ndarray) -> float:
        """
        Compute gain to normalize signal to target RMS.
        
        Parameters
        ----------
        signal : ndarray
            Input analog signal
            
        Returns
        -------
        gain : float
            Normalization gain factor
        """
        # Sample the signal
        sampled = signal[self.sampling_offset::self.samples_per_symbol]
        
        # Compute RMS
        signal_rms = np.sqrt(np.mean(sampled**2))
        
        if signal_rms > 0:
            gain = self.target_rms / signal_rms
        else:
            gain = 1.0
            
        self.normalization_gain = gain
        return gain
        
    def sample_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Sample analog signal at optimal offset.
        
        Parameters
        ----------
        signal : ndarray
            Analog signal (oversampled)
            
        Returns
        -------
        samples : ndarray
            Sampled values at symbol rate
        """
        samples = signal[self.sampling_offset::self.samples_per_symbol]
        return samples
        
    def quantize(self, samples: np.ndarray) -> np.ndarray:
        """
        Quantize analog samples to n-bit digital values.
        
        Parameters
        ----------
        samples : ndarray
            Analog sample values
            
        Returns
        -------
        quantized : ndarray
            Quantized digital values
        """
        # Normalize to target RMS
        samples_normalized = samples * self.normalization_gain
        
        # Clip to full-scale range
        clipped = np.clip(samples_normalized, self.v_min, self.v_max)
        
        # Count clipping events
        self.n_clipped += np.sum((samples_normalized < self.v_min) | 
                                  (samples_normalized > self.v_max))
        
        # Quantize: map from [v_min, v_max] to [0, n_levels-1]
        normalized = (clipped - self.v_min) / self.full_scale_voltage
        indices = np.round(normalized * (self.n_levels - 1))
        indices = np.clip(indices, 0, self.n_levels - 1).astype(int)
        
        # Convert back to voltage
        quantized = indices / (self.n_levels - 1) * self.full_scale_voltage + self.v_min
        
        # Update statistics
        self.n_samples_processed += len(samples)
        
        # Quantization noise
        quantization_error = quantized - clipped
        self.quantization_noise_power = np.mean(quantization_error**2)
        
        return quantized
        
    def convert(self, signal: np.ndarray, 
                auto_normalize: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Complete ADC conversion: sample and quantize.
        
        Parameters
        ----------
        signal : ndarray
            Input analog signal (oversampled)
        auto_normalize : bool
            If True, automatically compute normalization gain
            
        Returns
        -------
        digital_output : ndarray
            Quantized digital samples at symbol rate
        info : dict
            Conversion statistics and parameters
        """
        # Compute normalization gain if requested
        if auto_normalize:
            self.compute_normalization_gain(signal)
        
        # Sample at optimal offset
        samples = self.sample_signal(signal)
        
        # Quantize
        digital_output = self.quantize(samples)
        
        # Compute SNR
        signal_power = np.mean(samples**2)
        if self.quantization_noise_power > 0:
            snr_db = 10 * np.log10(signal_power / self.quantization_noise_power)
        else:
            snr_db = np.inf
        
        # Theoretical SNR for ideal ADC
        theoretical_snr_db = 6.02 * self.n_bits + 1.76
        
        info = {
            'n_samples': len(digital_output),
            'n_bits': self.n_bits,
            'n_levels': self.n_levels,
            'quantization_step': self.quantization_step,
            'normalization_gain': self.normalization_gain,
            'input_rms': np.sqrt(np.mean(samples**2)),
            'output_rms': np.sqrt(np.mean(digital_output**2)),
            'n_clipped': self.n_clipped,
            'clip_rate': self.n_clipped / self.n_samples_processed if self.n_samples_processed > 0 else 0,
            'quantization_noise_power': self.quantization_noise_power,
            'snr_db': snr_db,
            'theoretical_snr_db': theoretical_snr_db,
            'sampling_offset': self.sampling_offset,
        }
        
        return digital_output, info
        
    def get_quantization_levels(self) -> np.ndarray:
        """
        Get all quantization levels.
        
        Returns
        -------
        levels : ndarray
            Array of all quantization voltage levels
        """
        return np.linspace(self.v_min, self.v_max, self.n_levels)
    
    def add_nonlinear_distortion(self, signal: np.ndarray, 
                                 gain_compression: float = 0.1,
                                 third_order: float = 0.05) -> np.ndarray:
        """
        Add non-linear distortions to signal.
        
        Parameters
        ----------
        signal: Input signal
        gain_compression: Gain compression coefficient
        third_order: Third-order distortion coefficient
        
        Returns
        -------
        distorted: Signal with non-linear distortion
        
        """
        # Normalize signal for distortion calculation
        sig_norm = signal / (np.max(np.abs(signal)) + 1e-10)
        
        # AM-AM distortion (gain compression)
        envelope = np.abs(sig_norm)
        gain = 1.0 / (1.0 + gain_compression * envelope**2)
        
        # Third-order distortion
        distorted = signal * gain + third_order * signal**3
        
        return distorted 


    def get_config(self) -> Dict:
        """
        Get ADC configuration.
        
        Returns
        -------
        config : dict
            ADC parameters
        """
        return {
            'n_bits': self.n_bits,
            'n_levels': self.n_levels,
            'full_scale_voltage': self.full_scale_voltage,
            'quantization_step': self.quantization_step,
            'v_min': self.v_min,
            'v_max': self.v_max,
            'samples_per_symbol': self.samples_per_symbol,
            'sampling_offset': self.sampling_offset,
            'target_rms': self.target_rms,
            'normalization_gain': self.normalization_gain,
        }
        
    def print_info(self):
        """Print ADC configuration and statistics."""
        print("\n" + "="*60)
        print("ADC Configuration")
        print("="*60)
        print(f"Resolution:          {self.n_bits} bits ({self.n_levels} levels)")
        print(f"Full-Scale Range:    ±{self.full_scale_voltage/2:.3f} V")
        print(f"Quantization Step:   {self.quantization_step*1e3:.3f} mV")
        print(f"Sampling Offset:     {self.sampling_offset} samples")
        print(f"Target RMS:          {self.target_rms:.3f} V")
        print(f"Normalization Gain:  {self.normalization_gain:.4f}")
        
        if self.n_samples_processed > 0:
            print(f"\nStatistics:")
            print(f"Samples Processed:   {self.n_samples_processed}")
            print(f"Clipped Samples:     {self.n_clipped} ({100*self.n_clipped/self.n_samples_processed:.2f}%)")
            print(f"Quantization Noise:  {np.sqrt(self.quantization_noise_power)*1e3:.3f} mV RMS")
            print(f"Theoretical SNR:     {6.02*self.n_bits + 1.76:.2f} dB")
        print("="*60 + "\n")
        
    def plot_transfer_characteristic(self):
        """Plot ADC input-output transfer characteristic."""
        import matplotlib.pyplot as plt
        
        # Generate input range
        v_in = np.linspace(self.v_min * 1.2, self.v_max * 1.2, 1000)
        
        # Compute output for each input
        v_out = np.zeros_like(v_in)
        for i, v in enumerate(v_in):
            clipped = np.clip(v, self.v_min, self.v_max)
            normalized = (clipped - self.v_min) / self.full_scale_voltage
            index = int(np.round(normalized * (self.n_levels - 1)))
            index = np.clip(index, 0, self.n_levels - 1)
            v_out[i] = index / (self.n_levels - 1) * self.full_scale_voltage + self.v_min
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Transfer curve
        ax1.plot(v_in, v_out, 'b-', linewidth=2, label=f'{self.n_bits}-bit ADC')
        ax1.plot(v_in, v_in, 'r--', linewidth=1, alpha=0.5, label='Ideal (∞-bit)')
        ax1.axvline(x=self.v_min, color='orange', linestyle='--', alpha=0.5, label='FS Limits')
        ax1.axvline(x=self.v_max, color='orange', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Input Voltage [V]')
        ax1.set_ylabel('Output Voltage [V]')
        ax1.set_title(f'ADC Transfer Characteristic ({self.n_bits}-bit, {self.n_levels} levels)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Quantization error
        error = v_out - v_in
        ax2.plot(v_in, error * 1e3, 'r-', linewidth=1.5)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax2.axhline(y=self.quantization_step/2 * 1e3, color='gray', 
                   linestyle='--', alpha=0.5, label='±LSB/2')
        ax2.axhline(y=-self.quantization_step/2 * 1e3, color='gray',
                   linestyle='--', alpha=0.5)
        ax2.set_xlabel('Input Voltage [V]')
        ax2.set_ylabel('Quantization Error [mV]')
        ax2.set_title('Quantization Error')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        return fig


def add_nonlinearity_distortion(signal: np.ndarray, 
                                gain_compression: float = 0.1,
                                third_order: float = 0.05) -> np.ndarray:
    """
    Add non-linear distortions to signal (standalone function).
    
    Parameters
    ----------
    signal : ndarray
        Input signal
    gain_compression : float
        Gain compression coefficient (default: 0.1)
    third_order : float
        Third-order distortion coefficient (default: 0.05)
    
    Returns
    -------
    distorted : ndarray
        Signal with non-linear distortion
    
    """
    # Normalize signal for distortion calculation
    sig_norm = signal / (np.max(np.abs(signal)) + 1e-10)
    
    # AM-AM distortion (gain compression)
    envelope = np.abs(sig_norm)
    gain = 1.0 / (1.0 + gain_compression * envelope**2)
    
    # Third-order distortion
    distorted = signal * gain + third_order * signal**3
    
    return distorted


# Test function
if __name__ == "__main__":
    print("Testing ADC_n_bits class...\n")
    
    # Create test signal (PAM4-like)
    n_symbols = 1000
    samples_per_symbol = 32
    
    # Generate PAM4 levels: -1.5, -0.5, 0.5, 1.5
    rms_target = 0.5
    pam4_levels = np.array([-1.5, -0.5, 0.5, 1.5]) * rms_target
    symbols = np.random.choice(pam4_levels, size=n_symbols)
    
    # Oversample
    signal = np.repeat(symbols, samples_per_symbol)
    
    # Add some noise
    signal += np.random.normal(0, 0.05, len(signal))
    
    print(f"Test signal: {n_symbols} symbols, {len(signal)} samples")
    print(f"Signal RMS: {np.sqrt(np.mean(signal**2)):.4f} V\n")
    
    # Test different resolutions
    for n_bits in [3, 6, 8]:
        print(f"\n{'='*60}")
        print(f"Testing {n_bits}-bit ADC")
        print('='*60)
        
        adc = ADC_n_bits(
            n_bits=n_bits,
            full_scale_voltage=2.0,
            samples_per_symbol=samples_per_symbol,
            sampling_offset=0,
            target_rms=0.5
        )
        
        # Convert
        digital_out, info = adc.convert(signal, auto_normalize=True)
        
        adc.print_info()
        
        print(f"Conversion Info:")
        for key, value in info.items():
            if isinstance(value, float):
                print(f"  {key:25s}: {value:.4f}")
            else:
                print(f"  {key:25s}: {value}")
    
    # Plot transfer characteristic for 6-bit ADC
    print("\nGenerating ADC transfer characteristic plot...")
    adc_6bit = ADC_n_bits(n_bits=6, full_scale_voltage=2.0)
    fig = adc_6bit.plot_transfer_characteristic()
    fig.savefig('/mnt/user-data/outputs/adc_transfer_characteristic.png',
                dpi=300, bbox_inches='tight')
    print("Saved: adc_transfer_characteristic.png")
