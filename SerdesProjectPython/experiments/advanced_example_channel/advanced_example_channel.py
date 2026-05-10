#!/usr/bin/env python3
"""
Advanced Example: Channel Customization and Comparison

Demonstrates:
1. Comparing different transmission line lengths
2. Effect of parasitic capacitances
3. Impact of loss parameters
4. Multiple channel configurations side-by-side

Experiment layout: SerdesProjectPython/experiments/advanced_example_channel/
Figures: SerdesProjectPython/outputs/advanced_example_channel/

Imports use ``serdes_building_blocks`` (same modules as Serdes_Final_Vault1).
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# SerdesProjectPython/ (parent of experiments/)
_SERDES_PROJECT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = _SERDES_PROJECT / "outputs" / "advanced_example_channel"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_BB = _SERDES_PROJECT / "serdes_building_blocks"
if str(_BB) not in sys.path:
    sys.path.insert(0, str(_BB))

from serdes_channel import SerdesChannel
from pam4_generator import PAM4Generator
from tx_ffe import TX_FFE   # For FFE


def compare_line_lengths():
    """Compare channels with different transmission line lengths."""
    print("\n=== Comparing Transmission Line Lengths ===\n")
    
    lengths = [0.05, 0.1, 0.2, 0.3]  # 5cm, 10cm, 20cm, 30cm
    channels = []
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, length in enumerate(lengths):
        print(f"Computing {length*100:.0f} cm line...")
        channel = SerdesChannel(symbol_rate=32e9, samples_per_symbol=32, length=length)
        channel.compute_channel()
        channels.append(channel)
        
        # Plot frequency response
        ax = axes[idx // 2, idx % 2]
        ax.semilogx(channel.f * 1e-9, 20 * np.log10(np.abs(channel.H_channel)))
        ax.set_xlim([0.1, 100])
        ax.set_ylim([-40, 2])
        ax.set_xlabel('Frequency [GHz]')
        ax.set_ylabel('Magnitude [dB]')
        ax.set_title(f'Length = {length*100:.0f} cm')
        ax.grid(True, alpha=0.3)
        ax.axvline(x=16, color='red', linestyle='--', alpha=0.5, label='Nyquist (16 GHz)')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / 'comparison_lengths.png'), dpi=150, bbox_inches='tight')
    print("Saved: comparison_lengths.png\n")
    return channels


def compare_parasitics():
    """Compare effect of parasitic capacitances."""
    print("\n=== Comparing Parasitic Effects ===\n")
    
    configs = [
        ("No Parasitics", 0, 0),
        ("100 fF", 100e-15, 100e-15),
        ("200 fF", 200e-15, 200e-15),
        ("500 fF", 500e-15, 500e-15)
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, (name, cap_s, cap_t) in enumerate(configs):
        print(f"Computing {name}...")
        channel = SerdesChannel(symbol_rate=32e9, samples_per_symbol=32, length=0.1)
        channel.set_parasitics(cap_source=cap_s, cap_term=cap_t)
        channel.compute_channel()
        
        # Plot impulse response
        ax = axes[idx // 2, idx % 2]
        ax.plot(channel.t_impulse * 1e9, channel.h_impulse)
        ax.set_xlim([0, 3])
        ax.set_xlabel('Time [ns]')
        ax.set_ylabel('Impulse Response')
        ax.set_title(f'{name}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / 'comparison_parasitics.png'), dpi=150, bbox_inches='tight')
    print("Saved: comparison_parasitics.png\n")

def compare_parasitics2():
    """Compare effect of parasitic capacitances."""
    print("\n=== Comparing Parasitic Effects ===\n")
    
    configs = [
        ("No Parasitics", 0, 0),
        ("100 fF", 100e-15, 100e-15),
        ("200 fF", 200e-15, 200e-15),
        ("500 fF", 500e-15, 500e-15)
    ]
    
    channels = []
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for name, cap_s, cap_t in configs:
        print(f"Computing {name}...")
        channel = SerdesChannel(symbol_rate=64e9, samples_per_symbol=32, length=.1)
        channel.set_parasitics(cap_source=cap_s, cap_term=cap_t)
        channel.compute_channel()
        channels.append(channel)
        
        # Frequency response
        ax1.semilogx(channel.f * 1e-9, 20 * np.log10(np.abs(channel.H_channel)), 
                     label=name, linewidth=2)
        
        # Impulse response
        ax2.plot(channel.t_impulse * 1e9, channel.h_impulse, label=name, linewidth=2)
    
    ax1.set_xlim([0.1, 100])
    ax1.set_ylim([-40, 2])
    ax1.set_xlabel('Frequency [GHz]')
    ax1.set_ylabel('Magnitude [dB]')
    ax1.set_title('Frequency Response Comparison')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=32, color='red', linestyle='--', alpha=0.3)
    ax1.legend()
    
    ax2.set_xlim([0, 3])
    ax2.set_xlabel('Time [ns]')
    ax2.set_ylabel('Impulse Response')
    ax2.set_title('Impulse Response Comparison')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / 'comparison_parasitics2.png'), dpi=150, bbox_inches='tight')
    print("Saved: comparison_parasitics2.png\n")

def compare_impedance2():
    """Compare channels with different transmission line lengths."""
    print("\n=== Comparing Impedance ===\n")
    
    z0s = [25, 50, 75, 100]  # 25, 50, 75, 100 ohms
    channels = []
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, z0 in enumerate(z0s):
        print(f"Computing {z0} ohm impedance...")
        channel = SerdesChannel(symbol_rate=64e9, samples_per_symbol=32, length=.1, z0=z0s[idx])

        channel.compute_channel()
        channels.append(channel)
        
        # Frequency response
        ax1.semilogx(channel.f * 1e-9, 20 * np.log10(np.abs(channel.H_channel)), 
                     label=f'{z0} ohm', linewidth=2)

        # Impulse response
        ax2.plot(channel.t_impulse * 1e9, channel.h_impulse, label=f'{z0} ohm', linewidth=2)
    
    ax1.set_xlim([0.1, 100])
    ax1.set_ylim([-40, 2])
    ax1.set_xlabel('Frequency [GHz]')
    ax1.set_ylabel('Magnitude [dB]')
    ax1.set_title('Frequency Response Comparison')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=32, color='red', linestyle='--', alpha=0.3)
    ax1.legend()
    
    ax2.set_xlim([0, 3])
    ax2.set_xlabel('Time [ns]')
    ax2.set_ylabel('Impulse Response')
    ax2.set_title('Impulse Response Comparison')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / 'comparison_impedance2.png'), dpi=150, bbox_inches='tight')
    print("Saved: comparison_impedance2.png\n")

def compare_line_lengths2():
    """Compare channels with different transmission line lengths."""
    print("\n=== Comparing Transmission Line Lengths ===\n")
    
    lengths = [0.05, 0.1, 0.2, 0.3]  # 5cm, 10cm, 20cm, 30cm
    channels = []
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, length in enumerate(lengths):
        print(f"Computing {length*100:.0f} cm line...")
        channel = SerdesChannel(symbol_rate=64e9, samples_per_symbol=32, length=lengths[idx])

        channel.compute_channel()
        channels.append(channel)
        
        # Frequency response
        ax1.semilogx(channel.f * 1e-9, 20 * np.log10(np.abs(channel.H_channel)), 
                     label=f'{length*100:.0f} cm', linewidth=2)
        
        # Impulse response
        ax2.plot(channel.t_impulse * 1e9, channel.h_impulse, label=f'{length*100:.0f} cm', linewidth=2)
    
    ax1.set_xlim([0.1, 100])
    ax1.set_ylim([-40, 2])
    ax1.set_xlabel('Frequency [GHz]')
    ax1.set_ylabel('Magnitude [dB]')
    ax1.set_title('Frequency Response Comparison')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=32, color='red', linestyle='--', alpha=0.3)
    ax1.legend()
    
    ax2.set_xlim([0, 3])
    ax2.set_xlabel('Time [ns]')
    ax2.set_ylabel('Impulse Response')
    ax2.set_title('Impulse Response Comparison')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / 'comparison_lengths2.png'), dpi=150, bbox_inches='tight')
    print("Saved: comparison_lengths2.png\n")


def compare_loss_parameters():
    """Compare channels with different loss parameters."""
    print("\n=== Comparing Loss Parameters ===\n")
    
    configs = [
        ("Low Loss", 0.005, 50, 0.00005),
        ("Nominal", 0.01, 87, 0.0001),
        ("High Loss", 0.02, 150, 0.0002),
        ("Very High Loss", 0.03, 200, 0.0003)
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for name, theta, kr, rdc in configs:
        print(f"Computing {name}...")
        channel = SerdesChannel(symbol_rate=32e9, samples_per_symbol=32, length=0.1)
        channel.set_loss_parameters(theta_0=theta, k_r=kr, RDC=rdc)
        channel.compute_channel()
        
        # Frequency response
        ax1.semilogx(channel.f * 1e-9, 20 * np.log10(np.abs(channel.H_channel)), 
                     label=name, linewidth=2)
        
        # Impulse response
        ax2.plot(channel.t_impulse * 1e9, channel.h_impulse, label=name, linewidth=2)
    
    ax1.set_xlim([0.1, 100])
    ax1.set_ylim([-40, 2])
    ax1.set_xlabel('Frequency [GHz]')
    ax1.set_ylabel('Magnitude [dB]')
    ax1.set_title('Frequency Response Comparison')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=32, color='red', linestyle='--', alpha=0.3)
    ax1.legend()
    
    ax2.set_xlim([0, 3])
    ax2.set_xlabel('Time [ns]')
    ax2.set_ylabel('Impulse Response')
    ax2.set_title('Impulse Response Comparison')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / 'comparison_loss.png'), dpi=150, bbox_inches='tight')
    print("Saved: comparison_loss.png\n")

def eye_diagram_lengths_comparison():
    """Generate eye diagrams for different channel conditions."""
    print("\n=== Eye Diagram Lengths Comparison ===\n")
    
    # Generate test signal
    pam4 = PAM4Generator(seed=42)
    symbols = pam4.generate_random_symbols(5000)
    
    # Configurations to compare
    configs = [
        ("Short Line (5cm)", {'length': 0.05}),
        ("Nominal Line (10cm)", {'length': 0.1}),
        ("Medium Line (20cm)", {'length': 0.2}),
        ("Long Line (30cm)", {'length': 0.3})
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, (name, config) in enumerate(configs):
        print(f"Generating eye for: {name}")
        
        # Create channel
        channel = SerdesChannel(
            symbol_rate=64e9,
            samples_per_symbol=32,
            length=config.get('length', 0.1)
        )
        ffe = TX_FFE(taps=[0.083, -0.208, 0.709, 0])
        #ffe = TX_FFE(taps= [0, 0, 0, 1])

        # Apply parasitics if specified
        #if 'cap_source' in config:
        #    channel.set_parasitics(
        #        cap_source=config['cap_source'],
        #        cap_term=config['cap_term']
        #    )
        
        # Apply loss parameters if specified
        #if 'loss_params' in config:
        #    channel.set_loss_parameters(**config['loss_params'])
        
        # Compute and apply
        channel.compute_channel()
        signal_in = pam4.oversample(symbols, channel.samples_per_symbol)
        eq_signal = ffe.equalize(signal_in, samples_per_symbol=channel.samples_per_symbol)
        signal_out = channel.apply_channel(eq_signal)
        
        # Plot eye diagram
        samples_per_trace = 3 * channel.samples_per_symbol
        offset = 500
        signal_trim = signal_out[offset * channel.samples_per_symbol:]
        t_trace = np.arange(samples_per_trace) * channel.t_sample * 1e12  # ps
        
        for i in range(300):
            start = i * channel.samples_per_symbol
            end = start + samples_per_trace
            if end > len(signal_trim):
                break
            trace = signal_trim[start:end]
            axes[idx].plot(t_trace, trace, 'b', alpha=0.05, linewidth=0.5)
        
        axes[idx].set_xlabel('Time [ps]')
        axes[idx].set_ylabel('Amplitude [V]')
        axes[idx].set_title(name)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / 'comparison_eyes_with_ffe_lengths.png'), dpi=150, bbox_inches='tight')
    print("Saved: comparison_eyes_lengths.png\n")

def eye_diagram_impedance_comparison():
    """Generate eye diagrams for different channel conditions."""
    print("\n=== Eye Diagram Impedance Comparison ===\n")
    
    # Generate test signal
    pam4 = PAM4Generator(seed=42)
    symbols = pam4.generate_random_symbols(5000)
    
    # Configurations to compare
    configs = [
        ("With 25ohm Impedance", {'length': 0.1, 'z0': 25}),
        ("With 50ohm Impedance", {'length': 0.1, 'z0': 50}),
        ("With 75ohm Impedance", {'length': 0.1, 'z0': 75}),
        ("With 100ohm Impedance", {'length': 0.1, 'z0': 100})
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, (name, config) in enumerate(configs):
        print(f"Generating eye for: {name}")
        
        # Create channel
        channel = SerdesChannel(
            symbol_rate=64e9,
            samples_per_symbol=32,
            length=config.get('length', 0.1),
            z0=config.get('z0', 50)
        )
        ffe = TX_FFE(taps=[0.083, -0.208, 0.709, 0])
        #ffe = TX_FFE(taps= [0, 0, 0, 1])

        # Apply parasitics if specified
        #if 'cap_source' in config:
        #    channel.set_parasitics(
        #        cap_source=config['cap_source'],
        #        cap_term=config['cap_term']
        #    )
        
        # Apply loss parameters if specified
        #if 'loss_params' in config:
        #    channel.set_loss_parameters(**config['loss_params'])
        
        # Compute and apply
        channel.compute_channel()
        signal_in = pam4.oversample(symbols, channel.samples_per_symbol)
        eq_signal = ffe.equalize(signal_in, samples_per_symbol=channel.samples_per_symbol)
        signal_out = channel.apply_channel(eq_signal)
        
        # Plot eye diagram
        samples_per_trace = 3 * channel.samples_per_symbol
        offset = 500
        signal_trim = signal_out[offset * channel.samples_per_symbol:]
        t_trace = np.arange(samples_per_trace) * channel.t_sample * 1e12  # ps
        
        for i in range(300):
            start = i * channel.samples_per_symbol
            end = start + samples_per_trace
            if end > len(signal_trim):
                break
            trace = signal_trim[start:end]
            axes[idx].plot(t_trace, trace, 'b', alpha=0.05, linewidth=0.5)
        
        axes[idx].set_xlabel('Time [ps]')
        axes[idx].set_ylabel('Amplitude [V]')
        axes[idx].set_title(name)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / 'comparison_eyes_with_ffe_impedance.png'), dpi=150, bbox_inches='tight')
    print("Saved: comparison_eyes_impedance.png\n")

def eye_diagram_parasitics_comparison():
    """Generate eye diagrams for different channel conditions."""
    print("\n=== Eye Diagram Parasitics Comparison ===\n")
    
    # Generate test signal
    pam4 = PAM4Generator(seed=42)
    symbols = pam4.generate_random_symbols(5000)
    
    # Configurations to compare
    configs = [
        ("With No Parasitic Caps", {'length': 0.1, 'cap_source': 0e-15, 'cap_term': 0e-15}),
        ("With 100fF Caps", {'length': 0.1, 'cap_source': 100e-15, 'cap_term': 100e-15}),
        ("With 200fF Caps", {'length': 0.1, 'cap_source': 200e-15, 'cap_term': 200e-15}),
        ("With 500fF Caps", {'length': 0.1, 'cap_source': 500e-15, 'cap_term': 500e-15})
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, (name, config) in enumerate(configs):
        print(f"Generating eye for: {name}")
        
        # Create channel
        channel = SerdesChannel(
            symbol_rate=64e9,
            samples_per_symbol=32,
            length=config.get('length', 0.05)
        )
        ffe = TX_FFE(taps=[0.083, -0.208, 0.709, 0])
        #ffe = TX_FFE(taps= [0, 0, 0, 1])

        # Apply parasitics if specified
        if 'cap_source' in config:
            channel.set_parasitics(
                cap_source=config['cap_source'],
                cap_term=config['cap_term']
            )
        
        # Apply loss parameters if specified
        #if 'loss_params' in config:
        #    channel.set_loss_parameters(**config['loss_params'])
        
        # Compute and apply
        channel.compute_channel()
        signal_in = pam4.oversample(symbols, channel.samples_per_symbol)
        eq_signal = ffe.equalize(signal_in, samples_per_symbol=channel.samples_per_symbol)
        signal_out = channel.apply_channel(eq_signal)
        
        # Plot eye diagram
        samples_per_trace = 3 * channel.samples_per_symbol
        offset = 100
        signal_trim = signal_out[offset * channel.samples_per_symbol:]
        t_trace = np.arange(samples_per_trace) * channel.t_sample * 1e12  # ps
        
        for i in range(300):
            start = i * channel.samples_per_symbol
            end = start + samples_per_trace
            if end > len(signal_trim):
                break
            trace = signal_trim[start:end]
            axes[idx].plot(t_trace, trace, 'b', alpha=0.05, linewidth=0.5)
        
        axes[idx].set_xlabel('Time [ps]')
        axes[idx].set_ylabel('Amplitude [V]')
        axes[idx].set_title(name)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / 'comparison_eyes_with_ffe_parasitics.png'), dpi=150, bbox_inches='tight')
    print("Saved: comparison_eyes_parasitics.png\n")

def eye_diagram_loss_comparison():
    """Generate eye diagrams for different channel conditions."""
    print("\n=== Eye Diagram Comparison ===\n")
    
    # Generate test signal
    pam4 = PAM4Generator(seed=42)
    symbols = pam4.generate_random_symbols(5000)
    
    # Configurations to compare
    configs = [
        ("Low Loss", {'length': 0.1, 'loss_params': {'theta_0': 0.005, 'k_r': 50, 'RDC': 0.00005}}),
        ("Nominal", {'length': 0.1, 'loss_params': {'theta_0': 0.01, 'k_r': 87, 'RDC': 0.0001}}),
        ("High Loss", {'length': 0.1, 'loss_params': {'theta_0': 0.02, 'k_r': 150, 'RDC': 0.0002}}),
        ("Very High Loss", {'length': 0.1, 'loss_params': {'theta_0': 0.03, 'k_r': 200, 'RDC': 0.0003}})
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, (name, config) in enumerate(configs):
        print(f"Generating eye for: {name}")
        
        # Create channel
        channel = SerdesChannel(
            symbol_rate=64e9,
            samples_per_symbol=32,
            length=config.get('length', 0.1)
        )
        ffe = TX_FFE(taps=[0.083, -0.208, 0.709, 0])
        #ffe = TX_FFE(taps= [0, 0, 0, 1])

        # Apply parasitics if specified
        if 'cap_source' in config:
            channel.set_parasitics(
                cap_source=config['cap_source'],
                cap_term=config['cap_term']
            )
        
        # Apply loss parameters if specified
        if 'loss_params' in config:
            channel.set_loss_parameters(**config['loss_params'])
        
        # Compute and apply
        channel.compute_channel()
        signal_in = pam4.oversample(symbols, channel.samples_per_symbol)
        eq_signal = ffe.equalize(signal_in, samples_per_symbol=channel.samples_per_symbol)
        signal_out = channel.apply_channel(eq_signal)
        
        # Plot eye diagram
        samples_per_trace = 3 * channel.samples_per_symbol
        offset = 100
        signal_trim = signal_out[offset * channel.samples_per_symbol:]
        t_trace = np.arange(samples_per_trace) * channel.t_sample * 1e12  # ps
        
        for i in range(300):
            start = i * channel.samples_per_symbol
            end = start + samples_per_trace
            if end > len(signal_trim):
                break
            trace = signal_trim[start:end]
            axes[idx].plot(t_trace, trace, 'b', alpha=0.05, linewidth=0.5)
        
        axes[idx].set_xlabel('Time [ps]')
        axes[idx].set_ylabel('Amplitude [V]')
        axes[idx].set_title(name)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / 'comparison_eyes_with_ffe_loss.png'), dpi=150, bbox_inches='tight')
    print("Saved: comparison_eyes_loss.png\n")


def eye_diagram_comparison():
    """Generate eye diagrams for different channel conditions."""
    print("\n=== Eye Diagram Comparison ===\n")
    
    # Generate test signal
    pam4 = PAM4Generator(seed=42)
    symbols = pam4.generate_random_symbols(5000)
    
    # Configurations to compare
    configs = [
        ("Short Line (5cm)", {'length': 0.05}),
        ("Long Line (30cm)", {'length': 0.3}),
        ("With 200fF Caps", {'length': 0.1, 'cap_source': 200e-15, 'cap_term': 200e-15}),
        ("High Loss", {'length': 0.1, 'loss_params': {'theta_0': 0.03, 'k_r': 200}})
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, (name, config) in enumerate(configs):
        print(f"Generating eye for: {name}")
        
        # Create channel
        channel = SerdesChannel(
            symbol_rate=32e9,
            samples_per_symbol=32,
            length=config.get('length', 0.05)
        )
        
        # Apply parasitics if specified
        if 'cap_source' in config:
            channel.set_parasitics(
                cap_source=config['cap_source'],
                cap_term=config['cap_term']
            )
        
        # Apply loss parameters if specified
        if 'loss_params' in config:
            channel.set_loss_parameters(**config['loss_params'])
        
        # Compute and apply
        channel.compute_channel()
        signal_in = pam4.oversample(symbols, channel.samples_per_symbol)
        signal_out = channel.apply_channel(signal_in)
        
        # Plot eye diagram
        samples_per_trace = 3 * channel.samples_per_symbol
        offset = 100
        signal_trim = signal_out[offset * channel.samples_per_symbol:]
        t_trace = np.arange(samples_per_trace) * channel.t_sample * 1e12  # ps
        
        for i in range(300):
            start = i * channel.samples_per_symbol
            end = start + samples_per_trace
            if end > len(signal_trim):
                break
            trace = signal_trim[start:end]
            axes[idx].plot(t_trace, trace, 'b', alpha=0.05, linewidth=0.5)
        
        axes[idx].set_xlabel('Time [ps]')
        axes[idx].set_ylabel('Amplitude [V]')
        axes[idx].set_title(name)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / 'comparison_eyes.png'), dpi=150, bbox_inches='tight')
    print("Saved: comparison_eyes.png\n")


def parameter_sweep():
    """Sweep a parameter and show insertion loss at Nyquist frequency."""
    print("\n=== Parameter Sweep: Line Length vs Insertion Loss ===\n")
    
    lengths = np.linspace(0.01, 0.5, 20)  # 1cm to 50cm
    insertion_loss = []
    
    for length in lengths:
        channel = SerdesChannel(symbol_rate=64e9, samples_per_symbol=32, length=length)
        channel.compute_channel()
        
        # Find insertion loss at Nyquist frequency (16 GHz)
        nyquist_idx = np.argmin(np.abs(channel.f - 16e9))
        loss_db = 20 * np.log10(np.abs(channel.H_channel[nyquist_idx]))
        insertion_loss.append(loss_db)
    
    plt.figure(figsize=(10, 6))
    plt.plot(lengths * 100, insertion_loss, 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Transmission Line Length [cm]')
    plt.ylabel('Insertion Loss at Nyquist (16 GHz) [dB]')
    plt.title('Insertion Loss vs Line Length')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / 'sweep_length_loss.png'), dpi=150, bbox_inches='tight')
    print("Saved: sweep_length_loss.png")
    print(f"Loss range: {min(insertion_loss):.2f} to {max(insertion_loss):.2f} dB\n")


def main():
    """Run all comparisons."""
    print("\n" + "="*70)
    print("Advanced SerDes Channel Customization Examples")
    print("="*70)
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    # Run comparisons
    compare_loss_parameters()
    compare_line_lengths2()
    compare_parasitics2()
    compare_impedance2()
    eye_diagram_parasitics_comparison()
    eye_diagram_lengths_comparison()
    eye_diagram_loss_comparison()
    eye_diagram_impedance_comparison()
    #eye_diagram_comparison()
    parameter_sweep()
    
    print("\n" + "="*70)
    print("All comparisons complete!")
    print(f"Figures saved under: {OUTPUT_DIR}")
    print("="*70 + "\n")
    
    # Show one plot as example
    plt.show()


if __name__ == "__main__":

    main()