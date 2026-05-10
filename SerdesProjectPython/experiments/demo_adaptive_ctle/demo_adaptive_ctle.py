#!/usr/bin/env python3
"""
Adaptive CTLE Demonstration for PCIe 7.0 SerDes
================================================

Demonstrates:
1. Generating PAM-4 signal at 64 GHz symbol rate
2. Passing through lossy channel
3. Adaptive CTLE configuration selection
4. Data sequence processing with best CTLE config
5. Performance metrics (SNR, SER, eye diagrams)

Experiment: SerdesProjectPython/experiments/demo_adaptive_ctle/
Outputs: SerdesProjectPython/outputs/demo_adaptive_ctle/

Imports use ``serdes_building_blocks`` (aligned with Serdes_Final_Vault1). The demo sets
``AdaptiveCTLE(..., n_levels=4)`` to match the PAM-4 generator; the module default is PAM6.
Adaptation length and ``tx_symbols`` usage follow ``adapt()`` requirements in this repo.

Author: Anish Anand
Date: November 6, 2025
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

# SerdesProjectPython/serdes_building_blocks (same modules as Serdes_Final_Vault1)
_SERDES_PROJECT = Path(__file__).resolve().parent.parent.parent
_BB = _SERDES_PROJECT / "serdes_building_blocks"
if str(_BB) not in sys.path:
    sys.path.insert(0, str(_BB))

from serdes_channel import SerdesChannel
from pam4_generator import PAM4Generator, add_noise
from adaptive_ctle import AdaptiveCTLE
from tx_ffe import TX_FFE


def main():
    """Main demonstration of Adaptive CTLE."""
    
    # Figures: SerdesProjectPython/outputs/demo_adaptive_ctle/
    output_dir = str(_SERDES_PROJECT / "outputs" / "demo_adaptive_ctle")
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("PCIe 7.0 Adaptive CTLE Demonstration")
    print("="*70 + "\n")
    print(f"Output directory: {output_dir}\n")
    
    # ========================================================================
    # 1. Setup Parameters
    # ========================================================================
    print("Step 1: Setting up PCIe 7.0 parameters...")
    
    SYMBOL_RATE = 64e9      # 64 GHz (64 Gbaud) for PCIe 7.0
    SAMPLES_PER_SYMBOL = 32
    CHANNEL_LENGTH = 0.15   # 15 cm PCB trace
    
    # Adaptation: must cover AdaptiveCTLE.adapt() — building_blocks uses
    #   n_configs * symbols_per_config = 16 * 2000 = 32000 symbols.
    ADAPT_SYMBOLS = 32000
    
    # Data sequence for testing
    DATA_SYMBOLS = 10000
    
    TOTAL_SYMBOLS = ADAPT_SYMBOLS + DATA_SYMBOLS
    
    print(f"  Symbol Rate:        {SYMBOL_RATE/1e9:.1f} Gbaud")
    print(f"  Data Rate:          {SYMBOL_RATE*2/1e9:.1f} Gbps (PAM-4)")
    print(f"  Samples/Symbol:     {SAMPLES_PER_SYMBOL}")
    print(f"  Channel Length:     {CHANNEL_LENGTH*100:.1f} cm")
    print(f"  Adapt Symbols:      {ADAPT_SYMBOLS}")
    print(f"  Data Symbols:       {DATA_SYMBOLS}")
    print(f"  Total Symbols:      {TOTAL_SYMBOLS}")
    
    # ========================================================================
    # 2. Create Channel Model
    # ========================================================================
    print("\nStep 2: Creating SerDes channel model...")
    
    channel = SerdesChannel(
        symbol_rate=SYMBOL_RATE,
        samples_per_symbol=SAMPLES_PER_SYMBOL,
        length=CHANNEL_LENGTH,
        z0=50.0,
        eps_r=4.9
    )
    
    # Add realistic parasitics for PCIe
    channel.set_parasitics(
        cap_source=100e-15,   # 100 fF
        cap_term=150e-15      # 150 fF
    )
    
    # Adjust loss for PCIe 7.0 speeds
    channel.set_loss_parameters(
        theta_0=0.015,  # Higher loss at high frequencies
        k_r=120,        # Increased skin effect
        RDC=0.0002
    )
    
    # Compute channel response
    channel.compute_channel()
    channel.print_info()
    
    # ========================================================================
    # 3. Generate PAM-4 Signal
    # ========================================================================
    print("\nStep 3: Generating PAM-4 signal...")
    
    pam4_gen = PAM4Generator(seed=42)
    
    # Generate PAM-4 levels: [-3, -1, +1, +3]
    tx_pam4_levels = pam4_gen.generate_random_symbols(TOTAL_SYMBOLS)
    
    # Map PAM-4 levels to symbols [0, 1, 2, 3]
    level_to_symbol = {-3: 0, -1: 1, 1: 2, 3: 3}
    tx_symbols = np.array([level_to_symbol[level] for level in tx_pam4_levels])
    
    # Oversample
    tx_signal = pam4_gen.oversample(tx_pam4_levels, SAMPLES_PER_SYMBOL)
    
    print(f"  Generated {TOTAL_SYMBOLS} symbols")
    print(f"  TX signal length: {len(tx_signal)} samples")
    print(f"  Symbol distribution: {np.bincount(tx_symbols)}")
    # ========================================================================
    # 4. Apply FFE Equalizer
    # ========================================================================
    print("\nStep 4: Applying FFE equalizer...")
    # Step 2: Create FFE equalizer (taps: [0, 1, 0, 0])
 
    ffe = TX_FFE(taps=[0.083, -0.208, 0.709, 0])
    print(f"  FFE taps: {ffe.get_taps()}")
    
    # Step 3: Apply equalizer

    eq_signal = ffe.equalize(tx_signal, samples_per_symbol=SAMPLES_PER_SYMBOL)
    print(f"  Equalized {len(eq_signal)} samples (taps applied at symbol level)")
    
    # ========================================================================
    # 5. Pass Through Channel
    # ========================================================================
    print("\nStep 5: Passing signal through channel...")
    
    rx_signal = channel.apply_channel(eq_signal)
    
    # Add realistic noise (SNR ~ 20 dB at input)
    rx_signal = add_noise(rx_signal, snr_db=22, seed=123)
    
    print(f"  RX signal length: {len(rx_signal)} samples")
    print(f"  RX signal RMS: {np.sqrt(np.mean(rx_signal**2)):.4f} V")
    
    # ========================================================================
    # 6. Create Adaptive CTLE
    # ========================================================================
    print("\nStep 6: Creating Adaptive CTLE...")
    
    adaptive_ctle = AdaptiveCTLE(
        symbol_rate=SYMBOL_RATE,
        samples_per_symbol=SAMPLES_PER_SYMBOL,
        n_configs=16,
        n_levels=4,  # matches PAM-4 generator (building_blocks default is PAM6)
    )
    
    print("  Created 16 CTLE configurations")
    print("  Precomputed impulse responses")
    
    # ========================================================================
    # 7. Run Adaptation
    # ========================================================================
    print("\nStep 7: Running adaptation sequence...")
    
    # Extract adaptation portion (first 4000 symbols)
    adapt_samples = ADAPT_SYMBOLS * SAMPLES_PER_SYMBOL
    rx_adapt = rx_signal[:adapt_samples]
    
    # Run adaptation
    adapt_results = adaptive_ctle.adapt(
        rx_signal=rx_adapt,
        tx_symbols=tx_symbols[:ADAPT_SYMBOLS],
        verbose=True,
    )
    print(f"  RX CTLE Adaptation data length: {len(adapt_results['adaptation_output_signal'])} samples")
    # ========================================================================
    # 8. Pass Through CTLE with Best Config and Normalize to 0.5 V RMS
    # ========================================================================
    
    # Extract data portion (symbols after adaptation)
    data_start_sample = adapt_samples
    rx_data = rx_signal[data_start_sample:]
    tx_symbols_data = tx_symbols[ADAPT_SYMBOLS:]

    rx_ctle_data = adaptive_ctle.apply_ctle_with_best_config(rx_data)
    print(f"  RX CTLE data length: {len(rx_ctle_data)} samples")

    rx_ctle = np.concatenate([adapt_results['adaptation_output_signal'], rx_ctle_data]) #concatenate the adaptation output signal and the CTLE data
    print(f"  RX CTLE total length: {len(rx_ctle)} samples")

    # Determine root-mean-square (RMS) of the RX CTLE Data signal
    rms_rx_ctle_data = np.sqrt(np.mean(rx_ctle_data**2))
    print(f"  RX CTLE Data RMS: {rms_rx_ctle_data:.4f} V")

    # Normalize the RX CTLE signal to RMS of 0.5 V (simple VGA)
    rx_ctle = rx_ctle * (0.5 / rms_rx_ctle_data)
    print(f"  RX CTLE normalized RMS: {np.sqrt(np.mean(rx_ctle**2)):.4f} V")


    # ========================================================================
    # 9. Process CTLE Data Sequence
    # ========================================================================
    print("\nStep 9: Processing data sequence with best CTLE config...")
    
    
    
    # Process with best config
    data_results = adaptive_ctle.process_data_sequence(
        rx_signal_data=rx_data,
        tx_symbols_data=tx_symbols_data
    )
    
    print(f"  Processed {len(data_results['rx_symbols'])} symbols")
    if 'ser' in data_results:
        print(f"  Symbol Error Rate: {data_results['ser']:.2e}")
        print(f"  Bit Error Rate (est): {data_results['ser']/2:.2e}")
    
    # ========================================================================
    # 10. Print Summary
    # ========================================================================
    print("\nStep 10: Generating summary...")
    
    adaptive_ctle.print_summary(
        rx_signal_data=rx_data,
        tx_symbols_data=tx_symbols_data
    )
    
    # ========================================================================
    # 11. Visualizations
    # ========================================================================
    print("\nStep 11: Creating visualizations...")
    
    # Plot 1: Channel frequency response
    print("  - Channel Frequency Response")
    channel.plot_frequency_response(figsize=(12, 6))
    plt.savefig(f'{output_dir}/channel_frequency_response.png', dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved: {output_dir}/channel_frequency_response.png")
    
    # Plot 2: SNR comparison across configs
    print("  - SNR Comparison")
    adaptive_ctle.plot_snr_comparison(figsize=(14, 6))
    plt.savefig(f'{output_dir}/ctle_snr_comparison.png', dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved: {output_dir}/ctle_snr_comparison.png")
    
    # Plot 3: CTLE transfer functions
    print("  - CTLE Transfer Functions")
    plot_ctle_responses(adaptive_ctle, output_dir)
    
    # Plot 4: Eye diagrams before/after CTLE
    print("  - Eye Diagrams (Before/After CTLE)")
    adaptive_ctle.plot_comparison_eye_diagrams(rx_data, figsize=(16, 7))
    plt.savefig(f'{output_dir}/eye_diagrams_comparison.png', dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved: {output_dir}/eye_diagrams_comparison.png")
    
    # Plot 5: Detailed eye diagram for best config
    print("  - Detailed Eye Diagram (Best Config)")
    if adaptive_ctle.rx_signal_ctle is not None:
        adaptive_ctle.plot_eye_diagram(
            signal=adaptive_ctle.rx_signal_ctle,
            title=f"Eye Diagram - After CTLE (Config {adaptive_ctle.best_config_id})",
            n_traces=500,
            figsize=(12, 8)
        )
        plt.savefig(f'{output_dir}/eye_diagram_best_config.png', dpi=300, bbox_inches='tight')
        print(f"    ✓ Saved: {output_dir}/eye_diagram_best_config.png")
    
    # Plot 6: Signal comparison (time domain)
    print("  - Signal Time Domain Comparison")
    plot_signal_comparison(rx_data, adaptive_ctle, channel, output_dir)
    
    # Plot 7: Constellation diagram
    print("  - Constellation Diagram")
    plot_constellation(adaptive_ctle, data_results, output_dir)
    
    # Plot 8: Histogram comparison across pipeline stages
    print("  - Histogram Comparison (Pipeline Stages)")
    plot_histogram_comparison(
        tx_signal,
        eq_signal,
        rx_signal,
        rx_ctle,
        SAMPLES_PER_SYMBOL,
        output_dir,
        adaptive_ctle=adaptive_ctle,
    )
    
    print("\n" + "="*70)
    print("Demonstration Complete!")
    if matplotlib.get_backend().lower() != "agg":
        print("Close plot windows to exit.")
    print("="*70 + "\n")
    
    if matplotlib.get_backend().lower() != "agg":
        plt.show()


def plot_ctle_responses(adaptive_ctle, output_dir):
    """Plot CTLE transfer functions for all configurations."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    freqs = np.logspace(8, 11, 1000)  # 100 MHz to 100 GHz
    
    # Plot all configs
    for config in adaptive_ctle.configs:
        H_db = config.get_transfer_function_db(freqs)
        
        if config.config_id == adaptive_ctle.best_config_id:
            ax1.semilogx(freqs/1e9, H_db, 'r-', linewidth=2.5, 
                        label=f'Config {config.config_id} (Best)', zorder=10)
        else:
            ax1.semilogx(freqs/1e9, H_db, 'b-', alpha=0.3, linewidth=1)
    
    ax1.set_xlabel('Frequency [GHz]')
    ax1.set_ylabel('Magnitude [dB]')
    ax1.set_title('CTLE Transfer Functions - All Configurations')
    ax1.grid(True, which='both', alpha=0.3)
    ax1.legend()
    ax1.set_xlim([0.1, 100])
    ax1.axvline(x=32, color='green', linestyle='--', alpha=0.5, label='32 GHz')
    
    # Plot SNR vs Config
    if adaptive_ctle.adaptation_results:
        config_ids = sorted(adaptive_ctle.adaptation_results.keys())
        snrs = [adaptive_ctle.adaptation_results[cid]['snr_db'] for cid in config_ids]
        dc_gains = [-cid for cid in config_ids]
        
        ax2.plot(dc_gains, snrs, 'o-', markersize=8, linewidth=2, color='steelblue')
        
        # Highlight best
        best_idx = config_ids.index(adaptive_ctle.best_config_id)
        ax2.plot(dc_gains[best_idx], snrs[best_idx], 'ro', markersize=12, 
                label=f'Best (Config {adaptive_ctle.best_config_id})')
        
        ax2.set_xlabel('DC Gain [dB]')
        ax2.set_ylabel('SNR [dB]')
        ax2.set_title('SNR vs CTLE DC Gain')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/ctle_transfer_functions.png', dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved: {output_dir}/ctle_transfer_functions.png")


def plot_signal_comparison(rx_data, adaptive_ctle, channel, output_dir):
    """Plot signal before and after CTLE in time domain.

    ``rx_data`` and ``adaptive_ctle.rx_signal_ctle`` are both the **data** segment
    (same time origin). Red markers use ``best_sampling_offset`` from ``adapt()`` —
    the phase that minimized MSE vs ideal levels on the training segment — not UI
    phase 0 and not necessarily the local voltage peak within each UI.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # Show first 100 symbols
    n_show = 100
    sps = channel.samples_per_symbol
    samples_show = n_show * sps
    
    t = np.arange(samples_show) * channel.t_sample * 1e9  # in ns
    
    # Before CTLE
    ax1.plot(t, rx_data[:samples_show], 'r-', linewidth=0.8, alpha=0.7)
    ax1.set_ylabel('Amplitude [V]')
    ax1.set_title('Signal Before CTLE (After Channel)')
    ax1.grid(True, alpha=0.3)
    
    # After CTLE (process_data_sequence stores CTLE(rx_data), aligned with rx_data above)
    if adaptive_ctle.rx_signal_ctle is not None:
        rx_ctle = adaptive_ctle.rx_signal_ctle
        n_avail = min(samples_show, len(rx_ctle))
        ax2.plot(t[:n_avail], rx_ctle[:n_avail], 'b-', linewidth=0.8, alpha=0.7)

        off = int(adaptive_ctle.best_sampling_offset)
        sample_idx = np.arange(n_show, dtype=int) * sps + off
        sample_idx = sample_idx[sample_idx < n_avail]
        sampling_times = sample_idx * channel.t_sample * 1e9
        sampling_values = rx_ctle[sample_idx]
        ax2.plot(
            sampling_times,
            sampling_values,
            "ro",
            markersize=4,
            alpha=0.6,
            label=f"Adapt sampling phase (offset={off} sp)",
        )
        # Symbol (UI) boundaries — samples are offset sp samples into each UI, not at boundary
        for k in range(min(5, n_show)):
            t_ui = k * sps * channel.t_sample * 1e9
            ax2.axvline(
                t_ui,
                color="lightgray",
                linestyle=":",
                linewidth=0.9,
                alpha=0.6,
                zorder=0,
            )
        ax2.legend()
    
    ax2.set_xlabel('Time [ns]')
    ax2.set_ylabel('Amplitude [V]')
    ax2.set_title(
        f'Signal After CTLE (Config {adaptive_ctle.best_config_id}) — '
        f'samples at MSE-optimal phase from adapt(), not UI phase 0'
    )
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/signal_time_domain_comparison.png', dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved: {output_dir}/signal_time_domain_comparison.png")


def _pam_decision_thresholds_x(adaptive_ctle):
    """Decision thresholds on the voltage axis (same convention as adaptive_ctle plotting)."""
    vpnt = getattr(adaptive_ctle, "best_vpnt", None)
    if vpnt is None:
        return np.array([])
    a = float(getattr(adaptive_ctle, "pam_alpha", 1.0))
    nl = int(getattr(adaptive_ctle, "n_levels", 4))
    if nl == 4:
        return np.array([-1.0, 0.0, 1.0]) * vpnt * a
    if nl == 6:
        return np.array([-4, -2, 0, 2, 4]) * vpnt / 3 * a
    if nl == 8:
        return np.array([-6, -4, -2, 0, 2, 4, 6]) * vpnt / 4 * a
    return np.array([])


def plot_constellation(adaptive_ctle, data_results, output_dir):
    """Plot constellation diagram of received symbols."""
    if 'sampled_signal' not in data_results:
        return
    
    sampled = data_results['sampled_signal']
    symbols = data_results['rx_symbols']
    
    # Limit to first 5000 symbols for clarity
    n_plot = min(5000, len(sampled))
    sampled = sampled[:n_plot]
    symbols = symbols[:n_plot]
    
    n_levels = int(getattr(adaptive_ctle, "n_levels", 4))
    if adaptive_ctle.best_expected_levels is not None:
        n_levels = max(n_levels, len(adaptive_ctle.best_expected_levels))
    cmap = plt.cm.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(n_levels)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Histogram
    ax1.hist(sampled, bins=100, alpha=0.7, color='steelblue', edgecolor='black')
    
    # Mark expected levels
    for sym, level in adaptive_ctle.best_expected_levels.items():
        ax1.axvline(x=level, color='red', linestyle='--', linewidth=2,
                   label=f'Symbol {sym}' if sym == 0 else None)
    
    # Mark decision thresholds (best_vp4t when set; else best_vpnt-scaled per n_levels)
    vp4t = getattr(adaptive_ctle, "best_vp4t", None)
    if vp4t is not None and int(getattr(adaptive_ctle, "n_levels", 4)) == 4:
        ax1.axvline(
            x=-vp4t, color="orange", linestyle=":", linewidth=1.5, alpha=0.7, label="Thresholds"
        )
        ax1.axvline(x=0, color="orange", linestyle=":", linewidth=1.5, alpha=0.7)
        ax1.axvline(x=vp4t, color="orange", linestyle=":", linewidth=1.5, alpha=0.7)
    else:
        for i, tx in enumerate(_pam_decision_thresholds_x(adaptive_ctle)):
            ax1.axvline(
                x=tx,
                color="orange",
                linestyle=":",
                linewidth=1.5,
                alpha=0.7,
                label="Thresholds" if i == 0 else None,
            )
    
    ax1.set_xlabel('Voltage [V]')
    ax1.set_ylabel('Count')
    ax1.set_title('Voltage Distribution (Histogram)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Scatter plot colored by symbol
    for sym in range(n_levels):
        mask = symbols == sym
        if np.any(mask):
            ax2.scatter(np.arange(n_plot)[mask], sampled[mask], 
                       c=[colors[sym]], alpha=0.3, s=1, 
                       label=f'Symbol {sym}')
    
    # Mark expected levels
    for sym, level in adaptive_ctle.best_expected_levels.items():
        ax2.axhline(y=level, color=colors[sym], linestyle='--', 
                   linewidth=2, alpha=0.7)
    
    ax2.set_xlabel('Symbol Index')
    ax2.set_ylabel('Voltage [V]')
    ax2.set_title('Constellation Diagram')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, min(1000, n_plot)])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/constellation_diagram.png', dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved: {output_dir}/constellation_diagram.png")


def plot_histogram_comparison(
    tx_signal,
    tx_ffe_output,
    rx_signal,
    rx_ctle_output,
    samples_per_symbol,
    output_dir,
    adaptive_ctle=None,
):
    """
    Plot histogram comparison showing signal evolution through SerDes pipeline.

    TX / FFE / channel panels use **oversampled** waveforms (all samples).

    When ``adaptive_ctle`` is provided, the CTLE panel uses **symbol-rate**
    samples at ``best_sampling_offset`` from the **data-phase** CTLE output
    (same as ``process_data_sequence`` / slicer). Without it, the CTLE panel
    falls back to the raw ``rx_ctle_output`` slice (oversampled; may mix phases
    and adaptation vs data if ``rx_ctle_output`` is a long concat buffer).

    Parameters
    ----------
    tx_signal : ndarray or list
        Original PAM-4 signal (oversampled)
    tx_ffe_output : ndarray or list
        Signal after TX FFE
    rx_signal : ndarray or list
        Signal after channel (with noise)
    rx_ctle_output : ndarray or list
        Signal after CTLE (oversampled); used only if ``adaptive_ctle`` is None
    samples_per_symbol : int
        Number of samples per symbol
    output_dir : str
        Directory to save the plot
    adaptive_ctle : AdaptiveCTLE, optional
        If set, CTLE histogram uses ``rx_signal_ctle[best_sampling_offset::sps]``.
    """
    # Convert all inputs to numpy arrays to ensure mathematical operations work
    tx_signal = np.array(tx_signal)
    tx_ffe_output = np.array(tx_ffe_output)
    rx_signal = np.array(rx_signal)
    rx_ctle_output = np.array(rx_ctle_output)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # Use a subset of samples for histogram (e.g., first 10000 symbols)
    n_symbols_hist = 10000
    n_samples_hist = n_symbols_hist * samples_per_symbol

    ctle_title = "CTLE Output\n(After Equalization)"
    ctle_hist = rx_ctle_output[:n_samples_hist]
    if adaptive_ctle is not None and getattr(adaptive_ctle, "rx_signal_ctle", None) is not None:
        off = int(adaptive_ctle.best_sampling_offset)
        rxd = adaptive_ctle.rx_signal_ctle
        ctle_hist = rxd[off::samples_per_symbol][:n_symbols_hist]
        ctle_title = (
            f"CTLE Output\n(symbol-spaced @ adapt offset={off}, data phase)"
        )

    # Prepare data for each stage
    signals = [
        (tx_signal[:n_samples_hist], "PAM-4 TX Output\n(Before FFE)", "blue"),
        (tx_ffe_output[:n_samples_hist], "TX FFE Output\n(After Pre-emphasis)", "green"),
        (rx_signal[:n_samples_hist], "Channel Output\n(With Noise)", "red"),
        (ctle_hist, ctle_title, "purple"),
    ]
    
    # Calculate global min/max for consistent x-axis
    all_data = np.concatenate([sig[0] for sig in signals])
    global_min = np.percentile(all_data, 0.5)  # Ignore extreme outliers
    global_max = np.percentile(all_data, 99.5)
    
    # Plot each stage
    for idx, (signal, title, color) in enumerate(signals):
        ax = axes[idx]
        
        # Calculate statistics
        signal_mean = np.mean(signal)
        signal_std = np.std(signal)
        signal_rms = np.sqrt(np.mean(signal**2))
        signal_min = np.min(signal)
        signal_max = np.max(signal)
        
        # Create histogram
        n_bins = 200
        counts, bins, patches = ax.hist(signal, bins=n_bins, 
                                        range=(global_min, global_max),
                                        alpha=0.7, color=color, 
                                        edgecolor='black', linewidth=0.5,
                                        density=True)  # Normalize for comparison
        
        # For PAM-4 TX: peaks from histogram. For CTLE: expected levels from adapt()
        # when available (matches slicer); else peak-pick from histogram.
        if idx == 3 and adaptive_ctle is not None and adaptive_ctle.best_expected_levels:
            levels_map = adaptive_ctle.best_expected_levels
            sorted_items = sorted(levels_map.items(), key=lambda kv: kv[1])
            cmap = plt.cm.get_cmap("tab10")
            for i, (sym, v) in enumerate(sorted_items):
                ax.axvline(
                    x=v,
                    color=cmap(i % 10),
                    linestyle="--",
                    linewidth=2,
                    alpha=0.8,
                    label=f"Level {sym}",
                )
            vals = [v for _, v in sorted_items]
            for i in range(len(vals) - 1):
                thresh = (vals[i] + vals[i + 1]) / 2
                ax.axvline(
                    x=thresh,
                    color="orange",
                    linestyle=":",
                    linewidth=1.5,
                    alpha=0.6,
                    label="Thresholds" if i == 0 else None,
                )
        elif idx == 0 or idx == 3:  # PAM-4 TX or CTLE (fallback): peaks from histogram
            # Find peaks in histogram (expected 4 levels for PAM-4)
            from scipy.signal import find_peaks

            peaks, properties = find_peaks(counts, height=np.max(counts) * 0.1, distance=10)

            if len(peaks) >= 4:
                # Sort peaks by location and take the 4 most prominent
                peak_heights = counts[peaks]
                sorted_indices = np.argsort(peak_heights)[-4:]
                peak_locations = bins[peaks[sorted_indices]]
                peak_locations = np.sort(peak_locations)

                # Mark the 4 PAM-4 levels
                level_colors = ["darkblue", "darkgreen", "darkorange", "darkred"]
                for i, (peak_loc, level_color) in enumerate(zip(peak_locations, level_colors)):
                    ax.axvline(
                        x=peak_loc,
                        color=level_color,
                        linestyle="--",
                        linewidth=2,
                        alpha=0.8,
                        label=f"Level {i}",
                    )

                # Calculate and mark thresholds (midpoints between levels)
                if len(peak_locations) == 4:
                    thresh_01 = (peak_locations[0] + peak_locations[1]) / 2
                    thresh_12 = (peak_locations[1] + peak_locations[2]) / 2
                    thresh_23 = (peak_locations[2] + peak_locations[3]) / 2

                    for thresh in [thresh_01, thresh_12, thresh_23]:
                        ax.axvline(
                            x=thresh,
                            color="orange",
                            linestyle=":",
                            linewidth=1.5,
                            alpha=0.6,
                        )

                    # Add threshold label only once
                    ax.axvline(
                        x=thresh_01,
                        color="orange",
                        linestyle=":",
                        linewidth=1.5,
                        alpha=0.6,
                        label="Thresholds",
                    )
        
        # Add zero line
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
        
        # Title and labels
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('Voltage [V]', fontsize=11, fontweight='bold')
        ax.set_ylabel('Probability Density', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add statistics text box
        stats_text = f'Statistics:\n'
        stats_text += f'  Mean:    {signal_mean:>8.4f} V\n'
        stats_text += f'  Std Dev: {signal_std:>8.4f} V\n'
        stats_text += f'  RMS:     {signal_rms:>8.4f} V\n'
        stats_text += f'  Min:     {signal_min:>8.4f} V\n'
        stats_text += f'  Max:     {signal_max:>8.4f} V\n'
        stats_text += f'  Samples: {len(signal):>8d}'
        
        ax.text(0.02, 0.98, stats_text,
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
               fontsize=9, fontfamily='monospace')
        
        # Add legend for PAM-4 and CTLE stages
        if idx == 0 or idx == 3:
            ax.legend(loc='upper right', fontsize=8)
    
    # Add overall title
    fig.suptitle('Signal Evolution Through SerDes Pipeline - Histogram Comparison', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save figure
    save_path = f'{output_dir}/histogram_pipeline_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved: {save_path}")
    
    # Print detailed comparison
    print("\n" + "="*70)
    print("Signal Evolution Statistics")
    print("="*70)
    for signal, title, _ in signals:
        print(f"\n{title.replace(chr(10), ' ')}:")
        print(f"  RMS:     {np.sqrt(np.mean(signal**2)):.6f} V")
        print(f"  Std Dev: {np.std(signal):.6f} V")
        print(f"  Peak:    {np.max(np.abs(signal)):.6f} V")
        print(f"  Range:   [{np.min(signal):.6f}, {np.max(signal):.6f}] V")
    print("="*70)


if __name__ == "__main__":
    main()
