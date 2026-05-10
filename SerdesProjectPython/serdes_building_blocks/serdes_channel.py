"""
SerDes Channel Modeling Class
Implements lossy transmission line modeling with RLGC parameters
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import scipy.signal as signal


class SerdesChannel:
    """
    A class for modeling lossy transmission lines in high-speed SerDes applications.
    
    Attributes
    ----------
    symbol_rate : float
        Symbol rate in Hz (e.g., 32e9 for 32 GHz)
    samples_per_symbol : int
        Number of samples per symbol for oversampling
    f_max : float
        Maximum frequency for channel modeling
    length : float
        Transmission line length in meters
    """
    
    def __init__(self, 
                 symbol_rate: float = 64e9,
                 samples_per_symbol: int = 32,
                 length: float = 0.1,
                 z0: float = 50.0,
                 eps_r: float = 4.9):
        """
        Initialize SerDes channel model.
        
        Parameters
        ----------
        symbol_rate : float
            Symbol rate in Hz (default: 32 GHz)
        samples_per_symbol : int
            Oversampling factor (default: 32)
        length : float
            Transmission line length in meters (default: 0.1 m)
        z0 : float
            Characteristic impedance in ohms (default: 50)
        eps_r : float
            Effective relative dielectric constant (default: 4.9)
        """
        self.symbol_rate = symbol_rate
        self.samples_per_symbol = samples_per_symbol
        self.length = length
        self.z0 = z0
        self.eps_r = eps_r
        
        # Physical constants
        self.c = 2.998e8  # Speed of light [m/s]
        self.eps_0 = 8.85e-12  # Vacuum permittivity [F/m]
        
        # Derived parameters
        self.v0 = np.sqrt(1/eps_r) * self.c  # Propagation velocity
        self.L0 = z0 / self.v0  # Inductance per unit length [H/m]
        self.C0 = 1 / (z0 * self.v0)  # Capacitance per unit length [F/m]
        
        # Loss parameters (defaults - can be modified)
        self.w_0 = 2 * np.pi * 10e9  # Reference frequency [rad/s]
        self.theta_0 = 0.01  # Loss tangent parameter
        self.k_r = 87  # Skin-effect scaling factor [ohms/m @ w_0]
        self.RDC = 0.0001  # DC resistance [ohm/m]
        self.G0 = 1e-12  # Conductance [S/m]
        
        # Set up frequency vector
        self.f_max = symbol_rate * samples_per_symbol / 2
        self._setup_frequency_vector()
        
        # Channel response (computed when needed)
        self.H_channel = None
        self.h_impulse = None
        self.t_impulse = None
        self.h_pulse = None
        
        # Source and termination
        self.r_source = 50.0
        self.r_term = 50.0
        self.cap_source = 0.0  # Parasitic capacitance [F]
        self.cap_term = 0.0
        
    def _setup_frequency_vector(self, k: int = 14):
        """Set up frequency vector for channel modeling."""
        self.f = np.linspace(0, self.f_max, 2**k + 1)
        self.w = self.f * 2 * np.pi
        self.t_sample = 1 / (2 * self.f_max)
        
    def set_loss_parameters(self, 
                           theta_0: Optional[float] = None,
                           k_r: Optional[float] = None,
                           RDC: Optional[float] = None,
                           G0: Optional[float] = None):
        """
        Set loss parameters for the transmission line.
        
        Parameters
        ----------
        theta_0 : float, optional
            Loss tangent parameter
        k_r : float, optional
            Skin-effect scaling factor [ohms/m]
        RDC : float, optional
            DC resistance [ohm/m]
        G0 : float, optional
            Conductance [S/m]
        """
        if theta_0 is not None:
            self.theta_0 = theta_0
        if k_r is not None:
            self.k_r = k_r
        if RDC is not None:
            self.RDC = RDC
        if G0 is not None:
            self.G0 = G0
            
    def set_parasitics(self, cap_source: float = 0.0, cap_term: float = 0.0):
        """
        Set parasitic capacitances at source and termination.
        
        Parameters
        ----------
        cap_source : float
            Source parasitic capacitance in Farads
        cap_term : float
            Termination parasitic capacitance in Farads
        """
        self.cap_source = cap_source
        self.cap_term = cap_term
        
    def _compute_rlgc(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute frequency-dependent RLGC parameters."""
        # Frequency-dependent resistance (skin effect)
        RAC = self.k_r * (1 + 1j) * np.sqrt(self.w / self.w_0)
        R = np.sqrt(self.RDC**2 + RAC**2)
        
        # Constant inductance
        L = self.L0 * np.ones(np.size(self.f))
        
        # Constant conductance
        G = self.G0 * np.ones(np.size(self.f))
        
        # Frequency-dependent capacitance (dielectric loss)
        # Handle DC point separately to avoid division by zero
        C = np.zeros_like(self.w, dtype=complex)
        
        if self.f[0] == 0:
            # Set DC point to second frequency point value (will be computed below)
            mask = self.w != 0
            C[mask] = self.C0 * (1j * self.w[mask] / self.w_0)**(-2 * self.theta_0 / np.pi)
            C[0] = C[1]
        else:
            C = self.C0 * (1j * self.w / self.w_0)**(-2 * self.theta_0 / np.pi)
            
        return R, L, G, C
    
    def _rlgc_abcd(self, r: np.ndarray, l: np.ndarray, 
                   g: np.ndarray, c: np.ndarray) -> np.ndarray:
        """
        Compute ABCD parameters for transmission line from RLGC.
        
        Returns
        -------
        s : ndarray
            Array of shape (n_freq, 2, 2) containing ABCD matrices
        """
        gammad = self.length * np.sqrt(
            (r + 1j * self.w * l) * (g + 1j * self.w * c)
        )
        z0 = np.sqrt((r + 1j * self.w * l) / (g + 1j * self.w * c))
        
        A = np.cosh(gammad)
        B = z0 * np.sinh(gammad)
        C = np.sinh(gammad) / z0
        D = A
        
        s = np.zeros((self.f.size, 2, 2), dtype=np.complex128)
        s[:, 0, 0] = A
        s[:, 0, 1] = B
        s[:, 1, 0] = C
        s[:, 1, 1] = D
        
        return s
    
    def _impedance_abcd(self, z: np.ndarray) -> np.ndarray:
        """ABCD parameters for series impedance."""
        l = z.size
        s = np.zeros((l, 2, 2), dtype=np.complex128)
        s[:, 0, 0] = np.ones(l)
        s[:, 0, 1] = z
        s[:, 1, 0] = np.zeros(l)
        s[:, 1, 1] = np.ones(l)
        return s
    
    def _admittance_abcd(self, y: np.ndarray) -> np.ndarray:
        """ABCD parameters for shunt admittance."""
        l = y.size
        s = np.zeros((l, 2, 2), dtype=np.complex128)
        s[:, 0, 0] = np.ones(l)
        s[:, 0, 1] = np.zeros(l)
        s[:, 1, 0] = y
        s[:, 1, 1] = np.ones(l)
        return s
    
    def _shunt_cap_abcd(self, c: float) -> np.ndarray:
        """ABCD parameters for shunt capacitance."""
        l = self.w.size
        s = np.zeros((l, 2, 2), dtype=np.complex128)
        s[:, 0, 0] = np.ones(l)
        s[:, 0, 1] = np.zeros(l)
        s[:, 1, 0] = c * 1j * self.w
        s[:, 1, 1] = np.ones(l)
        return s
    
    def _series_abcd(self, networks: list) -> np.ndarray:
        """Cascade ABCD matrices."""
        out = np.matmul(networks[0], networks[1])
        for i in range(len(networks) - 2):
            out = np.matmul(out, networks[i + 2])
        return out
    
    def _freq2impulse(self, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert frequency response to impulse response via IFFT."""
        # Create symmetric spectrum for real-valued time signal
        Hd = np.concatenate((H, np.conj(np.flip(H[1:H.size-1]))))
        h = np.real(np.fft.ifft(Hd))
        t = np.linspace(0, 1/self.f[1], h.size + 1)
        t = t[0:-1]
        return h, t
    
    def compute_channel(self):
        """Compute the complete channel frequency and impulse response."""
        # Get RLGC parameters
        R, L, G, C = self._compute_rlgc()
        
        # Build transmission line ABCD
        tline = self._rlgc_abcd(R, L, G, C)
        
        # Build source impedance
        source = self._impedance_abcd(self.r_source * np.ones(np.size(self.f)))
        
        # Build termination admittance
        termination = self._admittance_abcd(np.ones(np.size(self.f)) / self.r_term)
        
        # Build network list
        networks = [source]
        
        # Add source parasitic capacitance if present
        if self.cap_source > 0:
            cap_network_source = self._shunt_cap_abcd(self.cap_source)
            networks.append(cap_network_source)
        
        # Add transmission line
        networks.append(tline)
        
        # Add termination parasitic capacitance if present
        if self.cap_term > 0:
            cap_network_term = self._shunt_cap_abcd(self.cap_term)
            networks.append(cap_network_term)
        
        # Add termination
        networks.append(termination)
        
        # Cascade all networks
        channel = self._series_abcd(networks)
        
        # Get frequency response
        self.H_channel = 1 / channel[:, 0, 0]
        
        # Get impulse response
        self.h_impulse, self.t_impulse = self._freq2impulse(self.H_channel)
        
        # Compute pulse response (response to one symbol period)
        samples_in_symbol = int(self.samples_per_symbol)
        self.h_pulse = signal.convolve(
            self.h_impulse, 
            np.ones(samples_in_symbol)
        )[:np.size(self.h_impulse)]
        
        return self.H_channel, self.h_impulse, self.t_impulse
    
    def plot_frequency_response(self, figsize=(10, 6), dpi=100):
        """Plot the channel frequency response magnitude."""
        if self.H_channel is None:
            self.compute_channel()
        
        nyquist_freq = self.symbol_rate / 2  # For PAM4 (2 bits per symbol)
        
        plt.figure(figsize=figsize, dpi=dpi)
        plt.semilogx(self.f * 1e-9, 20 * np.log10(np.abs(self.H_channel)))
        plt.xlim([0.1, self.f_max * 1e-9])
        plt.ylim([-40, 2])
        plt.xlabel('Frequency [GHz]')
        plt.ylabel('Magnitude Response [dB]')
        plt.title('Channel Frequency Response')
        plt.grid(True, which='both', alpha=0.3)
        plt.axvline(x=nyquist_freq * 1e-9, color='red', 
                   linestyle='--', label=f'Nyquist Freq ({nyquist_freq*1e-9:.1f} GHz)')
        plt.legend()
        plt.tight_layout()
        return plt.gcf()
    
    def plot_impulse_response(self, t_max_ns=5.0, figsize=(10, 6), dpi=100):
        """Plot the channel impulse response."""
        if self.h_impulse is None:
            self.compute_channel()
        
        plt.figure(figsize=figsize, dpi=dpi)
        plt.plot(self.t_impulse * 1e9, self.h_impulse)
        plt.xlim([0, t_max_ns])
        plt.xlabel('Time [ns]')
        plt.ylabel('Impulse Response')
        plt.title('Channel Impulse Response')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()
    
    def plot_pulse_response(self, t_max_ns=5.0, figsize=(10, 6), dpi=100):
        """Plot the channel pulse response (response to one symbol)."""
        if self.h_pulse is None:
            self.compute_channel()
        
        plt.figure(figsize=figsize, dpi=dpi)
        plt.plot(self.t_impulse * 1e9, self.h_pulse)
        plt.xlim([0, t_max_ns])
        plt.xlabel('Time [ns]')
        plt.ylabel('Pulse Response')
        plt.title('Channel Pulse Response (One Symbol)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()
    
    def plot_step_response(self, t_max_ns=5.0, figsize=(10, 6), dpi=100):
        """Plot the channel step response."""
        if self.h_impulse is None:
            self.compute_channel()
        
        h_step = signal.convolve(
            self.h_impulse, 
            np.ones(np.shape(self.h_impulse))
        )[:np.size(self.h_impulse)]
        
        plt.figure(figsize=figsize, dpi=dpi)
        plt.plot(self.t_impulse * 1e9, h_step)
        plt.xlim([0, t_max_ns])
        plt.xlabel('Time [ns]')
        plt.ylabel('Step Response [V]')
        plt.title('Channel Step Response')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()
    
    def apply_channel(self, signal_in: np.ndarray) -> np.ndarray:
        """
        Apply channel response to input signal.
        
        Parameters
        ----------
        signal_in : ndarray
            Input signal (oversampled at samples_per_symbol rate)
            
        Returns
        -------
        signal_out : ndarray
            Output signal after passing through channel
        """
        if self.h_impulse is None:
            self.compute_channel()
        
        signal_out = signal.convolve(self.h_impulse, signal_in)
        return signal_out
    
    def calculate_delay(self, method: str = 'group_delay') -> Tuple[int, float, float]:
        """
        Calculate channel delay from the transfer function.
        
        Computes the true delay based on the channel's phase response and group delay.
        This is more accurate than cross-correlation as it uses the actual channel
        transfer function H(f).
        
        Parameters
        ----------
        method : str, optional
            Method to calculate delay (default: 'group_delay')
            - 'group_delay': Uses derivative of phase (most accurate)
            - 'phase_slope': Uses linear fit to unwrapped phase
            - 'impulse_peak': Finds peak of impulse response
            
        Returns
        -------
        delay_samples : int
            Channel delay in samples
        delay_symbols : float
            Channel delay in symbols
        delay_time : float
            Channel delay in seconds
        
        Notes
        -----
        Group delay τ_g(ω) = -dφ(ω)/dω where φ(ω) is the phase response.
        This represents the true propagation delay through the channel.
        
        For a transmission line:
        - Physical delay = length / velocity = L / (c/√ε_r)
        - Additional delay from dispersion and loss
        """
        # Ensure channel has been computed
        if self.H_channel is None:
            self.compute_channel()
        
        if method == 'impulse_peak':
            # Method 1: Find peak of impulse response
            peak_idx = np.argmax(np.abs(self.h_impulse))
            delay_samples = peak_idx
            
        elif method == 'phase_slope':
            # Method 2: Linear fit to unwrapped phase in passband
            # Use frequencies up to Nyquist (symbol rate / 2)
            f_nyquist = self.symbol_rate / 2
            nyquist_idx = np.argmin(np.abs(self.f - f_nyquist))
            
            # Use first half of frequencies (avoid noise at high frequencies)
            freq_range = slice(1, nyquist_idx)  # Skip DC
            f_fit = self.f[freq_range]
            phase = np.unwrap(np.angle(self.H_channel[freq_range]))
            
            # Linear fit: phase = -2πf·τ + φ0
            # Slope = -2π·delay_time
            coeffs = np.polyfit(f_fit, phase, 1)
            delay_time = -coeffs[0] / (2 * np.pi)
            delay_samples = int(np.round(delay_time * 2 * self.f_max))
            
        else:  # 'group_delay' (default and most accurate)
            # Method 3: Group delay from derivative of phase
            # τ_g(ω) = -dφ/dω
            
            # Unwrap phase to avoid 2π discontinuities
            phase = np.unwrap(np.angle(self.H_channel))
            omega = 2 * np.pi * self.f
            
            # Compute group delay (negative derivative of phase)
            # Use gradient for numerical derivative
            group_delay = -np.gradient(phase, omega)
            
            # Average group delay in passband (up to Nyquist frequency)
            f_nyquist = self.symbol_rate / 2
            nyquist_idx = np.argmin(np.abs(self.f - f_nyquist))
            
            # Use middle of passband for most stable estimate (avoid DC and high freq)
            start_idx = max(1, nyquist_idx // 10)  # Skip DC region
            end_idx = int(nyquist_idx * 0.8)  # Avoid high frequency rolloff
            
            avg_group_delay = np.mean(group_delay[start_idx:end_idx])
            delay_time = avg_group_delay
            delay_samples = int(np.round(delay_time * 2 * self.f_max))
        
        # Calculate delay in symbols and seconds
        delay_symbols = delay_samples / self.samples_per_symbol
        delay_time = delay_samples * self.t_sample
        
        return delay_samples, delay_symbols, delay_time
    
    def plot_eye_diagram(self, signal: np.ndarray, n_traces: int = 500,
                        offset: int = 100, figsize=(10, 8), dpi=100):
        """
        Plot eye diagram of signal.
        
        Parameters
        ----------
        signal : ndarray
            Signal to plot
        n_traces : int
            Number of traces to overlay
        offset : int
            Number of symbols to skip at start
        """
        samples_per_trace = 3 * self.samples_per_symbol  # 3 symbols per trace
        
        # Skip initial transient
        signal_trimmed = signal[offset * self.samples_per_symbol:]
        
        # Create time vector for one trace
        t_trace = np.arange(samples_per_trace) * self.t_sample * 1e12  # in ps
        
        plt.figure(figsize=figsize, dpi=dpi)
        
        # Overlay traces
        for i in range(n_traces):
            start_idx = i * self.samples_per_symbol
            end_idx = start_idx + samples_per_trace
            
            if end_idx > len(signal_trimmed):
                break
                
            trace = signal_trimmed[start_idx:end_idx]
            plt.plot(t_trace, trace, 'b', alpha=0.05, linewidth=0.5)
        
        plt.xlabel('Time [ps]')
        plt.ylabel('Amplitude [V]')
        symbol_period_ps = (1 / self.symbol_rate) * 1e12
        plt.title(f'Eye Diagram - {self.symbol_rate/1e9:.0f} Gbaud PAM-4 Signal\n'
                 f'Symbol Period: {symbol_period_ps:.1f} ps')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()
    
    def get_channel_info(self) -> dict:
        """Return dictionary with channel parameters."""
        return {
            'symbol_rate_GHz': self.symbol_rate / 1e9,
            'samples_per_symbol': self.samples_per_symbol,
            'length_m': self.length,
            'z0_ohm': self.z0,
            'eps_r': self.eps_r,
            'v0_m_per_s': self.v0,
            'L0_H_per_m': self.L0,
            'C0_F_per_m': self.C0,
            'theta_0': self.theta_0,
            'k_r': self.k_r,
            'RDC_ohm_per_m': self.RDC,
            'G0_S_per_m': self.G0,
            'f_max_GHz': self.f_max / 1e9,
            't_sample_ps': self.t_sample * 1e12,
            'cap_source_fF': self.cap_source * 1e15,
            'cap_term_fF': self.cap_term * 1e15
        }
    
    def print_info(self):
        """Print channel parameters."""
        info = self.get_channel_info()
        print("=" * 60)
        print("SerDes Channel Parameters")
        print("=" * 60)
        print(f"Symbol Rate:           {info['symbol_rate_GHz']:.1f} Gbaud")
        print(f"Samples per Symbol:    {info['samples_per_symbol']}")
        print(f"Sample Rate:           {info['symbol_rate_GHz'] * info['samples_per_symbol']:.1f} GS/s")
        print(f"Time per Sample:       {info['t_sample_ps']:.3f} ps")
        print(f"Max Frequency:         {info['f_max_GHz']:.1f} GHz")
        print(f"Nyquist Frequency:     {info['symbol_rate_GHz']/2:.1f} GHz")
        print()
        print(f"Line Length:           {info['length_m']*1000:.1f} mm")
        print(f"Char. Impedance:       {info['z0_ohm']:.1f} Ω")
        print(f"Rel. Permittivity:     {info['eps_r']:.2f}")
        print(f"Propagation Velocity:  {info['v0_m_per_s']/self.c:.3f}c")
        print()
        print(f"Inductance L0:         {info['L0_H_per_m']*1e9:.2f} nH/m")
        print(f"Capacitance C0:        {info['C0_F_per_m']*1e12:.2f} pF/m")
        print(f"Loss Tangent θ₀:       {info['theta_0']:.4f}")
        print(f"Skin Effect k_r:       {info['k_r']:.1f} Ω/m")
        print(f"DC Resistance:         {info['RDC_ohm_per_m']*1000:.2f} mΩ/m")
        print(f"Conductance G0:        {info['G0_S_per_m']:.2e} S/m")
        print()
        print(f"Source Capacitance:    {info['cap_source_fF']:.1f} fF")
        print(f"Term. Capacitance:     {info['cap_term_fF']:.1f} fF")
        print("=" * 60)
    
    def calculate_vp4t(self, signal_segment: np.ndarray, n_symbols: int = 100) -> float:
        """
        Calculate V_P4T threshold from signal segment.
        
        V_P4T = sqrt(sum(signal^2) / N) where N = n_symbols × samples_per_symbol
        
        Parameters
        ----------
        signal_segment : ndarray
            Signal segment to analyze
        n_symbols : int
            Number of symbols to use (default: 100)
            
        Returns
        -------
        vp4t : float
            PAM-4 threshold voltage
        """
        n_samples = n_symbols * self.samples_per_symbol
        segment = signal_segment[:n_samples]
        squared_sum = np.sum(segment**2)
        vp4t = np.sqrt(squared_sum / len(segment))
        return vp4t
    
    def get_pam4_levels(self, vp4t: float) -> dict:
        """
        Get PAM-4 threshold and expected levels based on V_P4T.
        
        Parameters
        ----------
        vp4t : float
            PAM-4 threshold voltage
            
        Returns
        -------
        levels : dict
            Dictionary with 'thresholds' and 'expected' levels
        """
        return {
            'thresholds': {
                'lower': -vp4t,
                'zero': 0.0,
                'upper': vp4t
            },
            'expected': {
                'symbol_0': -1.5 * vp4t,  # bits 00, PAM level -3
                'symbol_1': -0.5 * vp4t,  # bits 01, PAM level -1
                'symbol_2': +0.5 * vp4t,  # bits 11, PAM level +1
                'symbol_3': +1.5 * vp4t   # bits 10, PAM level +3
            }
        }
