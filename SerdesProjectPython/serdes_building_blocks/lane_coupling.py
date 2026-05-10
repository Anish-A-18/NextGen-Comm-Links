#!/usr/bin/env python3
"""
Multi-Lane Coupling and Crosstalk Simulation for PCIe 7.0
==========================================================

Simulates realistic multi-lane SerDes channel effects:
- Near-End Crosstalk (NEXT)
- Far-End Crosstalk (FEXT)
- Pattern-dependent crosstalk
- Non-linear coupling effects
- Simultaneous Switching Noise (SSN)

Supports 1, 2, 4, 8, or 16 lane configurations typical in PCIe

Author: Anish Anand
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
from scipy.signal import lfilter, butter


@dataclass
class LaneCouplingConfig:
    """Configuration for lane coupling simulation."""
    n_lanes: int = 4                    # Number of lanes (1, 2, 4, 8, 16)
    
    # Crosstalk coupling coefficients (realistic PCIe 7.0 values)
    next_coupling: float = 0.03         # Near-end crosstalk (3%)
    fext_coupling: float = 0.02         # Far-end crosstalk (2%)
    
    # Adjacent vs non-adjacent lane coupling
    adjacent_coupling: float = 1.0      # Full coupling for adjacent lanes
    next_adjacent_coupling: float = 0.4  # Next-to-adjacent (40%)
    far_coupling: float = 0.1           # Non-adjacent lanes (10%)
    
    # Pattern-dependent effects
    pattern_dependent: bool = True      # Enable pattern-dependent crosstalk
    pattern_strength: float = 0.5       # Pattern dependence strength
    
    # Nonlinear coupling effects
    nonlinear_coupling: bool = True     # Enable nonlinear crosstalk
    nonlinear_strength: float = 0.08    # Nonlinear coupling strength
    
    # Frequency-dependent crosstalk (realistic for PCB traces)
    frequency_dependent: bool = True    # Enable frequency dependence
    xt_corner_freq: float = 10e9       # Crosstalk corner frequency (10 GHz)
    
    # Simultaneous Switching Noise
    ssn_enabled: bool = True           # Enable SSN effects
    ssn_strength: float = 0.05         # SSN amplitude (5% of signal)
    
    # Differential vs single-ended
    differential_mode: bool = True     # True for differential signaling
    
    def __post_init__(self):
        """Validate configuration."""
        valid_lanes = [1, 2, 4, 8, 16]
        if self.n_lanes not in valid_lanes:
            raise ValueError(f"n_lanes must be one of {valid_lanes}, got {self.n_lanes}")


class LaneCoupling:
    """
    Multi-lane coupling and crosstalk simulator.
    
    Models realistic PCIe 7.0 multi-lane effects including:
    - Capacitive and inductive coupling between lanes
    - Pattern-dependent crosstalk
    - Nonlinear aggressor-victim interactions
    - Frequency-dependent coupling
    """
    
    def __init__(self, config: LaneCouplingConfig):
        """
        Initialize lane coupling simulator.
        
        Args:
            config: Lane coupling configuration
        """
        self.config = config
        self.n_lanes = config.n_lanes
        
        # Build coupling matrix
        self.coupling_matrix = self._build_coupling_matrix()
        
        # Frequency-dependent filter (if enabled)
        if config.frequency_dependent:
            self._setup_frequency_filters()
        
        print(f"\n{'='*70}")
        print(f"Lane Coupling Simulator - {self.n_lanes} Lanes")
        print(f"{'='*70}")
        print(f"Configuration:")
        print(f"  NEXT coupling:     {config.next_coupling*100:.1f}%")
        print(f"  FEXT coupling:     {config.fext_coupling*100:.1f}%")
        print(f"  Pattern-dependent: {config.pattern_dependent}")
        print(f"  Nonlinear coupling: {config.nonlinear_coupling}")
        print(f"  SSN enabled:       {config.ssn_enabled}")
        print(f"  Differential mode: {config.differential_mode}")
        print(f"{'='*70}\n")
    
    def _build_coupling_matrix(self) -> np.ndarray:
        """
        Build lane-to-lane coupling matrix.
        
        Returns:
            coupling_matrix: (n_lanes, n_lanes) coupling coefficients
                           coupling_matrix[i,j] = coupling from lane j to lane i
        """
        n = self.n_lanes
        coupling = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    coupling[i, j] = 1.0  # Self (victim lane)
                else:
                    # Distance-based coupling
                    distance = abs(i - j)
                    
                    if distance == 1:
                        # Adjacent lanes - strongest coupling
                        coupling[i, j] = self.config.adjacent_coupling
                    elif distance == 2:
                        # Next-to-adjacent
                        coupling[i, j] = self.config.next_adjacent_coupling
                    else:
                        # Far lanes - weak coupling
                        coupling[i, j] = self.config.far_coupling
        
        # Normalize off-diagonal elements by coupling strength
        for i in range(n):
            for j in range(n):
                if i != j:
                    coupling[i, j] *= self.config.next_coupling
        
        return coupling
    
    def _setup_frequency_filters(self):
        """Setup frequency-dependent crosstalk filters."""
        # First-order low-pass for crosstalk frequency dependence
        # Crosstalk increases with frequency up to corner, then levels off
        nyquist_estimate = 32e9  # Assume ~32 GHz Nyquist for 64 GBaud
        normalized_freq = self.config.xt_corner_freq / nyquist_estimate
        normalized_freq = np.clip(normalized_freq, 0.01, 0.99)
        
        # Low-pass Butterworth filter
        self.xt_b, self.xt_a = butter(1, normalized_freq, btype='low')
    
    def add_lane_coupling(self, 
                         signals: np.ndarray,
                         symbols: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """
        Apply multi-lane coupling to signals.
        
        Args:
            signals: Input signals (n_lanes, n_samples)
            symbols: Symbol sequences for pattern-dependent effects (n_lanes, n_symbols)
                    Required if pattern_dependent=True
        
        Returns:
            coupled_signals: Signals with crosstalk (n_lanes, n_samples)
            info: Dictionary with coupling information
        """
        if signals.shape[0] != self.n_lanes:
            raise ValueError(f"Expected {self.n_lanes} lanes, got {signals.shape[0]}")
        
        n_samples = signals.shape[1]
        coupled_signals = np.zeros_like(signals)
        
        # Track components for analysis
        next_component = np.zeros_like(signals)
        fext_component = np.zeros_like(signals)
        nonlinear_component = np.zeros_like(signals)
        ssn_component = np.zeros_like(signals)
        
        # Apply coupling lane by lane
        for victim_lane in range(self.n_lanes):
            # Start with victim signal
            victim_signal = signals[victim_lane, :]
            
            # Accumulate crosstalk from all aggressor lanes
            next_xt = np.zeros(n_samples)
            fext_xt = np.zeros(n_samples)
            nonlinear_xt = np.zeros(n_samples)
            
            for aggressor_lane in range(self.n_lanes):
                if aggressor_lane == victim_lane:
                    continue
                
                aggressor_signal = signals[aggressor_lane, :]
                coupling_coeff = self.coupling_matrix[victim_lane, aggressor_lane]
                
                # Near-End Crosstalk (NEXT) - derivative coupling
                next_xt_lane = self._compute_next(aggressor_signal, coupling_coeff)
                next_xt += next_xt_lane
                
                # Far-End Crosstalk (FEXT) - direct coupling
                fext_xt_lane = self._compute_fext(aggressor_signal, coupling_coeff)
                fext_xt += fext_xt_lane
                
                # Pattern-dependent crosstalk
                if self.config.pattern_dependent and symbols is not None:
                    pattern_xt = self._compute_pattern_dependent_xt(
                        aggressor_signal, 
                        symbols[aggressor_lane, :],
                        symbols[victim_lane, :],
                        coupling_coeff
                    )
                    fext_xt += pattern_xt
                
                # Nonlinear coupling
                if self.config.nonlinear_coupling:
                    nonlinear_xt_lane = self._compute_nonlinear_coupling(
                        victim_signal,
                        aggressor_signal,
                        coupling_coeff
                    )
                    nonlinear_xt += nonlinear_xt_lane
            
            # Simultaneous Switching Noise (affects all lanes similarly)
            if self.config.ssn_enabled:
                ssn = self._compute_ssn(signals, victim_lane)
                ssn_component[victim_lane, :] = ssn
            else:
                ssn = 0
            
            # Combine all effects
            coupled_signals[victim_lane, :] = (
                victim_signal + 
                next_xt + 
                fext_xt + 
                nonlinear_xt + 
                ssn
            )
            
            # Store components for analysis
            next_component[victim_lane, :] = next_xt
            fext_component[victim_lane, :] = fext_xt
            nonlinear_component[victim_lane, :] = nonlinear_xt
        
        # Compute statistics
        info = {
            'next_rms': np.sqrt(np.mean(next_component**2, axis=1)),
            'fext_rms': np.sqrt(np.mean(fext_component**2, axis=1)),
            'nonlinear_rms': np.sqrt(np.mean(nonlinear_component**2, axis=1)),
            'ssn_rms': np.sqrt(np.mean(ssn_component**2, axis=1)),
            'total_xt_rms': np.sqrt(np.mean(
                (next_component + fext_component + nonlinear_component + ssn_component)**2, 
                axis=1
            )),
            'next_component': next_component,
            'fext_component': fext_component,
            'nonlinear_component': nonlinear_component,
            'ssn_component': ssn_component
        }
        
        return coupled_signals, info
    
    def _compute_next(self, aggressor: np.ndarray, coupling: float) -> np.ndarray:
        """
        Compute Near-End Crosstalk (derivative coupling).
        
        NEXT is proportional to dI/dt (capacitive/inductive coupling)
        """
        # Derivative approximation
        next_xt = np.diff(aggressor, prepend=aggressor[0])
        next_xt *= coupling * self.config.next_coupling
        
        # Apply frequency-dependent filter if enabled
        if self.config.frequency_dependent:
            next_xt = lfilter(self.xt_b, self.xt_a, next_xt)
        
        return next_xt
    
    def _compute_fext(self, aggressor: np.ndarray, coupling: float) -> np.ndarray:
        """
        Compute Far-End Crosstalk (direct coupling).
        
        FEXT is proportional to aggressor signal amplitude
        """
        fext_xt = aggressor * coupling * self.config.fext_coupling
        
        # FEXT has some delay (propagation through coupled region)
        # Approximate with small shift
        delay_samples = 2
        fext_xt = np.roll(fext_xt, delay_samples)
        fext_xt[:delay_samples] = 0
        
        return fext_xt
    
    def _compute_pattern_dependent_xt(self,
                                     aggressor: np.ndarray,
                                     aggressor_symbols: np.ndarray,
                                     victim_symbols: np.ndarray,
                                     coupling: float) -> np.ndarray:
        """
        Compute pattern-dependent crosstalk.
        
        Crosstalk strength depends on symbol patterns in both lanes.
        Stronger when lanes have opposing transitions.
        """
        n_samples = len(aggressor)
        samples_per_symbol = n_samples // len(aggressor_symbols)
        
        pattern_xt = np.zeros(n_samples)
        
        for i in range(min(len(aggressor_symbols), len(victim_symbols))):
            start = i * samples_per_symbol
            end = start + samples_per_symbol
            
            if end > n_samples:
                break
            
            # Pattern-dependent coupling factor
            # Stronger for opposite transitions
            if i > 0:
                agg_transition = aggressor_symbols[i] - aggressor_symbols[i-1]
                vic_transition = victim_symbols[i] - victim_symbols[i-1]
                
                # Opposing transitions increase crosstalk
                pattern_factor = 1.0 + self.config.pattern_strength * abs(
                    np.sign(agg_transition) - np.sign(vic_transition)
                ) / 2.0
            else:
                pattern_factor = 1.0
            
            # Apply pattern-dependent scaling to this symbol period
            pattern_xt[start:end] = (
                aggressor[start:end] * 
                coupling * 
                self.config.fext_coupling * 
                pattern_factor * 
                0.5  # Scale down since this is additional effect (.5)
            )
        
        return pattern_xt
    
    def _compute_nonlinear_coupling(self,
                                   victim: np.ndarray,
                                   aggressor: np.ndarray,
                                   coupling: float) -> np.ndarray:
        """
        Compute nonlinear crosstalk.
        
        Models saturation and intermodulation effects when multiple
        high-amplitude signals are present.
        """
        # Nonlinear coupling: proportional to victim * aggressor (intermodulation)
        nonlinear_xt = (
            victim * aggressor * 
            coupling * 
            self.config.nonlinear_strength
        )
        
        # Also add third-order term (victim^2 * aggressor)
        victim_norm = victim / (np.max(np.abs(victim)) + 1e-10)
        nonlinear_xt += (
            victim_norm**2 * aggressor * 
            coupling * 
            self.config.nonlinear_strength * 
            0.3  # Weaker third-order effect (.3)
        )
        
        return nonlinear_xt
    
    def _compute_ssn(self, all_signals: np.ndarray, victim_lane: int) -> np.ndarray:
        """
        Compute Simultaneous Switching Noise.
        
        SSN occurs when multiple lanes switch simultaneously,
        causing ground bounce and power supply noise.
        """
        n_samples = all_signals.shape[1]
        
        # Count total switching activity across all lanes
        total_activity = np.zeros(n_samples)
        
        for lane in range(self.n_lanes):
            if lane == victim_lane:
                continue
            # Switching activity is proportional to derivative
            activity = np.abs(np.diff(all_signals[lane, :], prepend=all_signals[lane, 0]))
            total_activity += activity
        
        # SSN is proportional to total switching activity
        ssn = total_activity * self.config.ssn_strength / self.n_lanes
        
        # Add some low-frequency component (power supply droop)
        if len(ssn) > 100:
            # Simple moving average for low-freq component
            window = 50
            kernel = np.ones(window) / window
            ssn_lf = np.convolve(ssn, kernel, mode='same')
            ssn = ssn + 0.3 * ssn_lf #(.3)
        
        return ssn
    
    def visualize_coupling(self, 
                          signals: np.ndarray,
                          coupled_signals: np.ndarray,
                          info: Dict,
                          time_range: Tuple[int, int] = None,
                          lane_to_show: int = 0) -> plt.Figure:
        """
        Visualize lane coupling effects.
        
        Args:
            signals: Original signals (n_lanes, n_samples)
            coupled_signals: Signals with coupling (n_lanes, n_samples)
            info: Coupling information from add_lane_coupling()
            time_range: (start, end) sample indices to plot
            lane_to_show: Which victim lane to analyze
        
        Returns:
            fig: Matplotlib figure
        """
        if time_range is None:
            time_range = (0, min(1000, signals.shape[1]))
        
        start, end = time_range
        time_axis = np.arange(start, end)
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
        
        # 1. All lane signals (original)
        ax1 = fig.add_subplot(gs[0, 0])
        for lane in range(self.n_lanes):
            offset = lane * 3  # Vertical offset for visibility
            ax1.plot(time_axis, signals[lane, start:end] + offset, 
                    label=f'Lane {lane}', alpha=0.7, linewidth=1)
        ax1.set_xlabel('Sample')
        ax1.set_ylabel('Amplitude (offset per lane)')
        ax1.set_title(f'Original Signals - All {self.n_lanes} Lanes', 
                     fontweight='bold')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. Victim lane comparison
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(time_axis, signals[lane_to_show, start:end], 
                label='Original', linewidth=2, alpha=0.7)
        ax2.plot(time_axis, coupled_signals[lane_to_show, start:end], 
                label='With Crosstalk', linewidth=2, alpha=0.7)
        ax2.set_xlabel('Sample')
        ax2.set_ylabel('Amplitude')
        ax2.set_title(f'Lane {lane_to_show} - Before/After Coupling', 
                     fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Crosstalk components
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(time_axis, info['next_component'][lane_to_show, start:end], 
                label='NEXT', alpha=0.7, linewidth=1.5)
        ax3.plot(time_axis, info['fext_component'][lane_to_show, start:end], 
                label='FEXT', alpha=0.7, linewidth=1.5)
        ax3.plot(time_axis, info['nonlinear_component'][lane_to_show, start:end], 
                label='Nonlinear', alpha=0.7, linewidth=1.5)
        ax3.plot(time_axis, info['ssn_component'][lane_to_show, start:end], 
                label='SSN', alpha=0.7, linewidth=1.5)
        ax3.set_xlabel('Sample')
        ax3.set_ylabel('Amplitude')
        ax3.set_title(f'Crosstalk Components - Lane {lane_to_show}', 
                     fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Total crosstalk vs signal
        ax4 = fig.add_subplot(gs[1, 1])
        total_xt = (info['next_component'][lane_to_show, start:end] + 
                   info['fext_component'][lane_to_show, start:end] + 
                   info['nonlinear_component'][lane_to_show, start:end] +
                   info['ssn_component'][lane_to_show, start:end])
        ax4.plot(time_axis, signals[lane_to_show, start:end], 
                label='Signal', linewidth=2, alpha=0.7)
        ax4.plot(time_axis, total_xt, 
                label='Total Crosstalk', linewidth=2, alpha=0.7)
        ax4.set_xlabel('Sample')
        ax4.set_ylabel('Amplitude')
        ax4.set_title(f'Signal vs Total Crosstalk - Lane {lane_to_show}', 
                     fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Coupling matrix heatmap
        ax5 = fig.add_subplot(gs[2, 0])
        im = ax5.imshow(self.coupling_matrix, cmap='hot', aspect='auto')
        ax5.set_xlabel('Aggressor Lane')
        ax5.set_ylabel('Victim Lane')
        ax5.set_title('Lane Coupling Matrix', fontweight='bold')
        plt.colorbar(im, ax=ax5, label='Coupling Coefficient')
        
        # Add values on heatmap
        for i in range(self.n_lanes):
            for j in range(self.n_lanes):
                text = ax5.text(j, i, f'{self.coupling_matrix[i, j]:.3f}',
                              ha="center", va="center", color="w", fontsize=8)
        
        # 6. RMS crosstalk per lane
        ax6 = fig.add_subplot(gs[2, 1])
        lanes = np.arange(self.n_lanes)
        width = 0.15
        ax6.bar(lanes - 1.5*width, info['next_rms'], width, 
               label='NEXT', alpha=0.8)
        ax6.bar(lanes - 0.5*width, info['fext_rms'], width, 
               label='FEXT', alpha=0.8)
        ax6.bar(lanes + 0.5*width, info['nonlinear_rms'], width, 
               label='Nonlinear', alpha=0.8)
        ax6.bar(lanes + 1.5*width, info['ssn_rms'], width, 
               label='SSN', alpha=0.8)
        ax6.set_xlabel('Lane')
        ax6.set_ylabel('RMS Amplitude')
        ax6.set_title('Crosstalk RMS by Component', fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')
        ax6.set_xticks(lanes)
        
        # 7. Signal-to-Crosstalk Ratio
        ax7 = fig.add_subplot(gs[3, 0])
        signal_rms = np.sqrt(np.mean(signals**2, axis=1))
        scr_db = 20 * np.log10(signal_rms / (info['total_xt_rms'] + 1e-10))
        ax7.bar(lanes, scr_db, alpha=0.8, color='steelblue')
        ax7.axhline(y=20, color='r', linestyle='--', 
                   label='20 dB threshold', linewidth=2)
        ax7.set_xlabel('Lane')
        ax7.set_ylabel('SCR (dB)')
        ax7.set_title('Signal-to-Crosstalk Ratio', fontweight='bold')
        ax7.legend()
        ax7.grid(True, alpha=0.3, axis='y')
        ax7.set_xticks(lanes)
        
        # 8. Crosstalk statistics summary
        ax8 = fig.add_subplot(gs[3, 1])
        ax8.axis('off')
        
        stats_text = f"""
        Lane Coupling Statistics
        {'='*40}
        Number of Lanes: {self.n_lanes}
        
        Victim Lane {lane_to_show} Analysis:
        ────────────────────────────────────
        NEXT RMS:      {info['next_rms'][lane_to_show]:.4f} V
        FEXT RMS:      {info['fext_rms'][lane_to_show]:.4f} V
        Nonlinear RMS: {info['nonlinear_rms'][lane_to_show]:.4f} V
        SSN RMS:       {info['ssn_rms'][lane_to_show]:.4f} V
        Total XT RMS:  {info['total_xt_rms'][lane_to_show]:.4f} V
        
        Signal RMS:    {signal_rms[lane_to_show]:.4f} V
        SCR:          {scr_db[lane_to_show]:.1f} dB
        
        Configuration:
        ────────────────────────────────────
        Pattern-dependent: {self.config.pattern_dependent}
        Nonlinear coupling: {self.config.nonlinear_coupling}
        Frequency-dependent: {self.config.frequency_dependent}
        SSN enabled: {self.config.ssn_enabled}
        """
        
        ax8.text(0.1, 0.5, stats_text, transform=ax8.transAxes,
                fontsize=10, verticalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        fig.suptitle(f'{self.n_lanes}-Lane Coupling Analysis', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        return fig
    
    def print_summary(self, info: Dict):
        """Print summary of coupling effects."""
        print(f"\n{'='*70}")
        print(f"Lane Coupling Summary - {self.n_lanes} Lanes")
        print(f"{'='*70}")
        print("\nCrosstalk RMS per Lane (V):")
        print(f"{'Lane':<6} {'NEXT':<10} {'FEXT':<10} {'Nonlinear':<12} {'SSN':<10} {'Total':<10}")
        print("-" * 70)
        
        for lane in range(self.n_lanes):
            print(f"{lane:<6} "
                  f"{info['next_rms'][lane]:<10.4f} "
                  f"{info['fext_rms'][lane]:<10.4f} "
                  f"{info['nonlinear_rms'][lane]:<12.4f} "
                  f"{info['ssn_rms'][lane]:<10.4f} "
                  f"{info['total_xt_rms'][lane]:<10.4f}")
        
        print(f"\nAverage Total Crosstalk: {np.mean(info['total_xt_rms']):.4f} V")
        print(f"Max Total Crosstalk:     {np.max(info['total_xt_rms']):.4f} V")
        print(f"{'='*70}\n")


def demo_lane_coupling():
    """Demonstrate lane coupling with example signals."""
    
    print("\n" + "="*70)
    print("Multi-Lane Coupling Demonstration")
    print("="*70)
    
    # Configuration for 4-lane PCIe
    config = LaneCouplingConfig(
        n_lanes=4,
        next_coupling=0.05,
        fext_coupling=0.03,
        pattern_dependent=True,
        nonlinear_coupling=True,
        ssn_enabled=True
    )
    
    # Create lane coupling simulator
    lane_sim = LaneCoupling(config)
    
    # Generate example PAM4 signals for each lane
    n_symbols = 1000
    samples_per_symbol = 32
    n_samples = n_symbols * samples_per_symbol
    
    signals = np.zeros((config.n_lanes, n_samples))
    symbols = np.zeros((config.n_lanes, n_symbols), dtype=int)
    
    # PAM4 levels
    pam4_levels = np.array([-3, -1, 1, 3])
    
    for lane in range(config.n_lanes):
        # Random PAM4 symbols
        symbols[lane, :] = np.random.randint(0, 4, n_symbols)
        
        # Convert to waveform with pulse shaping
        for i, sym in enumerate(symbols[lane, :]):
            start = i * samples_per_symbol
            end = start + samples_per_symbol
            
            # Simple raised cosine pulse
            t = np.linspace(0, 1, samples_per_symbol)
            pulse = pam4_levels[sym] * (1 - np.cos(2*np.pi*t)) / 2
            signals[lane, start:end] = pulse
        
        # Add some noise
        signals[lane, :] += np.random.randn(n_samples) * 0.1
    
    # Apply lane coupling
    coupled_signals, info = lane_sim.add_lane_coupling(signals, symbols)
    
    # Print summary
    lane_sim.print_summary(info)
    
    # Visualize
    fig = lane_sim.visualize_coupling(
        signals, 
        coupled_signals, 
        info,
        time_range=(0, 500),
        lane_to_show=1
    )
    
    plt.savefig('./Serdes_v5_ml/outputs/lane_coupling_demo.png', 
                dpi=150, bbox_inches='tight')
    print("Visualization saved to lane_coupling_demo.png")
    
    plt.show()
    
    return coupled_signals, info


if __name__ == "__main__":
    demo_lane_coupling()
