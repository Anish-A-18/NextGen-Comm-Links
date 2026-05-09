#!/usr/bin/env python3
"""
Train RX FFE taps for the lane-coupled Verilog pipeline.

Models the exact Verilog pipeline:
  Victim:    PRBS7(seed=0x7F) -> PAM4 -> Composite FIR -> + coupling -> signal
  Aggressor: PRBS7(seed=0x55) -> PAM4 -> Composite FIR -> * COUPLING_COEFF -> coupling

The coupling is FEXT: a scaled version of the aggressor's composite FIR output
is added to the victim's composite FIR output.

This script:
1. Simulates the 3-mode Verilog pipeline cycle-by-cycle
2. Finds optimal SER delay for Mode 2 (coupled, no FFE)
3. Trains FFE taps on the coupled signal (Mode 3)
4. Outputs all parameters for Verilog
"""

import numpy as np

H_COMP = [288, 84, -30, -27, -24, -17, -12, -7, -4, -2, -3, -2]
N_COMP = len(H_COMP)
PAM4_MAP = {0: -3, 1: -1, 2: 1, 3: 3}

N_FFE = 8
N_PRE = 3
Q_FFE = 14
SYMBOL_DIV = 4
NUM_SYMBOLS = 200000

# FEXT coupling: aggressor FIR output * COUPLING_COEFF added to victim
# Tuned so Mode 2 (coupled, no FFE) ~ 19%, Mode 3 (coupled + FFE) ~ 0.08%
COUPLING_COEFF_NUM = 17    # numerator   (coupling = NUM / DEN)
COUPLING_COEFF_DEN = 64    # denominator (17/64 = 0.265625)


def to_signed_16(val):
    val = val & 0xFFFF
    return val - 0x10000 if val >= 0x8000 else val


def slicer(val, th_low, th_mid, th_high):
    if val < th_low:
        return 0
    elif val < th_mid:
        return 1
    elif val < th_high:
        return 2
    else:
        return 3


def simulate_pipeline():
    """Simulate the full 3-mode Verilog pipeline."""
    print("=" * 70)
    print("LANE COUPLING SIMULATION + FFE TRAINING")
    print(f"  Coupling = {COUPLING_COEFF_NUM}/{COUPLING_COEFF_DEN} "
          f"= {COUPLING_COEFF_NUM/COUPLING_COEFF_DEN:.4f}")
    print("=" * 70)

    # Victim PRBS-7 (seed 0x7F, same as current design)
    prbs_v = 0x7F
    comp_sr_v = [0] * N_COMP
    comp_out_v = 0

    # Aggressor PRBS-7 (different seed 0x55)
    prbs_a = 0x55
    comp_sr_a = [0] * N_COMP
    comp_out_a = 0

    # FFE shift register (operates on coupled signal)
    ffe_sr = [0] * N_FFE

    tx_symbols = []
    ffe_buffers = []  # exact FFE input buffers for training
    coupled_values = []  # coupled signal for Mode 2 slicer verification

    # Mode 1 values (no coupling)
    mode1_comp_acc_list = []

    # coupled_out is registered (NB) in Verilog — FFE reads the OLD value.
    # So we model: coupled_out_reg starts at 0, gets updated to current
    # coupled_val at end of cycle (NB), and FFE reads the OLD reg value.
    coupled_out_reg = 0

    for clk in range((NUM_SYMBOLS + 300) * SYMBOL_DIV):
        symbol_tick = (clk % SYMBOL_DIV == SYMBOL_DIV - 1)
        if not symbol_tick:
            continue

        # Victim PRBS-7
        new_bit_v = ((prbs_v >> 6) ^ (prbs_v >> 5)) & 1
        prbs_v = ((prbs_v << 1) | new_bit_v) & 0x7F
        tx_sym_v = prbs_v & 0x3
        pam4_v = PAM4_MAP[tx_sym_v]

        # Aggressor PRBS-7 (different seed/sequence)
        new_bit_a = ((prbs_a >> 6) ^ (prbs_a >> 5)) & 1
        prbs_a = ((prbs_a << 1) | new_bit_a) & 0x7F
        tx_sym_a = prbs_a & 0x3
        pam4_a = PAM4_MAP[tx_sym_a]

        # Victim composite FIR (blocking acc, reads OLD comp_sr)
        comp_acc_v = sum(comp_sr_v[i] * H_COMP[i] for i in range(N_COMP))
        comp_acc_v_16 = to_signed_16(comp_acc_v)

        # Aggressor composite FIR
        comp_acc_a = sum(comp_sr_a[i] * H_COMP[i] for i in range(N_COMP))
        comp_acc_a_16 = to_signed_16(comp_acc_a)

        # Mode 1: victim FIR output only (no coupling)
        mode1_comp_acc_list.append(comp_acc_v_16)

        # Save old comp_outs before NB update
        old_comp_out_v = comp_out_v
        old_comp_out_a = comp_out_a

        # Non-blocking updates for both FIRs
        comp_sr_v = [pam4_v] + comp_sr_v[:-1]
        comp_out_v = comp_acc_v_16
        comp_sr_a = [pam4_a] + comp_sr_a[:-1]
        comp_out_a = comp_acc_a_16

        # Coupling block (reads old comp_out_v/a, computes blocking coupled_val)
        coupling_shift = int(np.log2(COUPLING_COEFF_DEN))
        coupling_term = (old_comp_out_a * COUPLING_COEFF_NUM) >> coupling_shift
        coupled_val = old_comp_out_v + coupling_term

        if coupled_val > 32767:
            coupled_val = 32767
        elif coupled_val < -32768:
            coupled_val = -32768
        coupled_val = to_signed_16(coupled_val)

        # Mode 2 slicer reads blocking coupled_val (same block in Verilog)
        coupled_values.append(coupled_val)

        # FFE reads OLD coupled_out_reg (registered from previous cycle)
        ffe_buffers.append(list(ffe_sr))

        # Non-blocking FFE shift register: gets OLD coupled_out_reg
        ffe_sr = [coupled_out_reg] + ffe_sr[:-1]

        # Non-blocking coupled_out register update (takes effect next cycle)
        coupled_out_reg = coupled_val

        tx_symbols.append(tx_sym_v)
        if len(tx_symbols) >= NUM_SYMBOLS + 300:
            break

    tx_symbols = np.array(tx_symbols)
    ffe_buffers = np.array(ffe_buffers, dtype=np.float64)
    coupled_values = np.array(coupled_values, dtype=np.int64)
    mode1_values = np.array(mode1_comp_acc_list, dtype=np.int64)

    return tx_symbols, ffe_buffers, coupled_values, mode1_values


def find_thresholds_and_ser(signal, tx_symbols, label, warmup=200):
    """Find optimal thresholds and SER for a signal."""
    sig_rms = np.sqrt(np.mean(signal[warmup:].astype(np.float64) ** 2))
    th_low = int(np.round(-sig_rms * 0.895))
    th_mid = 0
    th_high = int(np.round(sig_rms * 0.895))

    det = np.zeros(len(signal), dtype=int)
    det[signal < th_low] = 0
    det[(signal >= th_low) & (signal < th_mid)] = 1
    det[(signal >= th_mid) & (signal < th_high)] = 2
    det[signal >= th_high] = 3

    best_ser = 1.0
    best_delay = 0
    for d in range(20):
        ref = tx_symbols[warmup:len(tx_symbols) - d]
        dt = det[warmup + d:]
        n = min(len(ref), len(dt))
        if n > 1000:
            errors = np.sum(ref[:n] != dt[:n])
            ser = errors / n
            if ser < best_ser:
                best_ser = ser
                best_delay = d
            if d < 12:
                print(f"    delay={d:2d}: SER={ser:.6f}")

    print(f"  {label}: delay={best_delay}, SER={best_ser:.6f} ({best_ser*100:.4f}%)")
    print(f"  Thresholds: [{th_low}, {th_mid}, {th_high}], RMS={sig_rms:.1f}")
    return best_delay, best_ser, th_low, th_mid, th_high


def main():
    tx_symbols, ffe_buffers, coupled_values, mode1_values = simulate_pipeline()

    # ================================================================
    # Mode 2: Coupled signal, threshold slicer (no FFE)
    # ================================================================
    print("\n" + "=" * 60)
    print("MODE 2: Coupled signal (no FFE)")
    print("=" * 60)
    delay_m2, ser_m2, th_low_m2, _, th_high_m2 = find_thresholds_and_ser(
        coupled_values, tx_symbols, "Mode 2")

    # ================================================================
    # Mode 3: Train FFE on coupled signal
    # ================================================================
    print("\n" + "=" * 60)
    print("MODE 3: FFE on coupled signal")
    print("=" * 60)

    # Find cursor alignment
    cursor_signal = ffe_buffers[:, N_PRE]
    pam4_levels = np.array([-3, -1, 1, 3])
    tx_pam4 = pam4_levels[tx_symbols]

    best_corr = 0
    best_d = 0
    for d in range(-2, 15):
        start = max(200, 200 + d)
        end = min(len(tx_pam4), len(cursor_signal)) - abs(d)
        if d >= 0:
            a = tx_pam4[200:end - d]
            b = cursor_signal[200 + d:end]
        else:
            a = tx_pam4[200 - d:end]
            b = cursor_signal[200:end + d]
        n = min(len(a), len(b))
        if n > 1000:
            corr = np.corrcoef(a[:n], b[:n])[0, 1]
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_d = d
    print(f"  Cursor alignment: delay={best_d}, corr={best_corr:.4f}")

    # LMS training
    cursor_gain = H_COMP[0]
    desired = np.zeros(len(ffe_buffers))
    for i in range(len(ffe_buffers)):
        sym_idx = i - best_d
        if 0 <= sym_idx < len(tx_symbols):
            desired[i] = pam4_levels[tx_symbols[sym_idx]] * cursor_gain

    sig_rms = np.sqrt(np.mean(cursor_signal[200:] ** 2))
    ffe_norm = ffe_buffers / sig_rms
    desired_norm = desired / sig_rms

    MU = 0.01
    N_TRAIN = 100000
    taps = np.zeros(N_FFE, dtype=np.float64)
    taps[N_PRE] = 1.0

    print(f"  LMS training: {N_TRAIN} symbols, mu={MU}")
    for i in range(200, 200 + N_TRAIN):
        if i >= len(ffe_norm):
            break
        buf = ffe_norm[i]
        y = np.dot(taps, buf)
        e = desired_norm[i] - y
        taps += 2 * MU * e * buf

    cursor_val = taps[N_PRE]
    taps_norm = taps / cursor_val if abs(cursor_val) > 1e-10 else taps
    ffe_q = np.round(taps_norm * (2 ** Q_FFE)).astype(int)
    print(f"  Normalized taps: {taps_norm}")
    print(f"  Q1.14 taps: {ffe_q}")

    # Verify FFE
    ffe_output = np.zeros(len(ffe_buffers), dtype=np.int64)
    for i in range(len(ffe_buffers)):
        acc = 0
        for k in range(N_FFE):
            acc += int(ffe_buffers[i][k]) * int(ffe_q[k])
        ffe_output[i] = acc >> Q_FFE

    print(f"\n  FFE verification:")
    delay_m3, ser_m3, th_low_m3, _, th_high_m3 = find_thresholds_and_ser(
        ffe_output, tx_symbols, "Mode 3")

    # Per-symbol stats
    for sym in range(4):
        d = delay_m3
        mask = tx_symbols[200:len(tx_symbols) - max(0, d)] == sym
        vals = ffe_output[200 + max(0, d):]
        n = min(len(mask), len(vals))
        sym_vals = vals[:n][mask[:n]]
        if len(sym_vals) > 0:
            print(f"    Symbol {sym}: mean={np.mean(sym_vals):.1f}, std={np.std(sym_vals):.1f}")

    # DAC offsets
    coupled_valid = coupled_values[200:]
    ffe_valid = ffe_output[200:]
    dac_offset_m2 = int(np.min(coupled_valid)) - 100
    dac_offset_m3 = int(np.min(ffe_valid)) - 100

    # ================================================================
    # Output parameters
    # ================================================================
    labels = ['PRE3', 'PRE2', 'PRE1', 'MAIN', 'POST1', 'POST2', 'POST3', 'POST4']
    print(f"\n{'='*70}")
    print(f"VERILOG PARAMETERS FOR COUPLED MODE")
    print(f"{'='*70}")
    print(f"  // Lane coupling")
    print(f"  localparam signed [15:0] COUPLING_NUM = 16'sd{COUPLING_COEFF_NUM};")
    print(f"  localparam COUPLING_SHIFT = {int(np.log2(COUPLING_COEFF_DEN))};")
    print()
    print(f"  // Mode 2 (coupled, no FFE): SYMBOL_DELAY_2 = {delay_m2}")
    print(f"  localparam signed [15:0] COUPLED_THRESH_LOW  = "
          f"{'-' if th_low_m2 < 0 else ''}16'sd{abs(th_low_m2)};")
    print(f"  localparam signed [15:0] COUPLED_THRESH_HIGH =  16'sd{abs(th_high_m2)};")
    print(f"  localparam signed [15:0] DAC_OFFSET_2 = "
          f"{'-' if dac_offset_m2 < 0 else ''}16'sd{abs(dac_offset_m2)};")
    print()
    print(f"  // Mode 3 (coupled + FFE): SYMBOL_DELAY_3 = {delay_m3}")
    for i, (label, q) in enumerate(zip(labels, ffe_q)):
        s = "" if q >= 0 else "-"
        print(f"  localparam signed [15:0] FFE_{label} = {s}16'sd{abs(q)};  "
              f"// {taps_norm[i]:+.6f}")
    print(f"  localparam signed [15:0] FFE_THRESH_LOW  = "
          f"{'-' if th_low_m3 < 0 else ''}16'sd{abs(th_low_m3)};")
    print(f"  localparam signed [15:0] FFE_THRESH_HIGH =  16'sd{abs(th_high_m3)};")
    print(f"  localparam signed [15:0] DAC_OFFSET_3 = "
          f"{'-' if dac_offset_m3 < 0 else ''}16'sd{abs(dac_offset_m3)};")
    print()
    print(f"  // Expected SER")
    print(f"  //   Mode 1 (no coupling):  ~9.45%")
    print(f"  //   Mode 2 (coupled):      {ser_m2*100:.2f}%")
    print(f"  //   Mode 3 (coupled+FFE):  {ser_m3*100:.4f}%")
    print(f"{'='*70}")

    return {
        'delay_m2': delay_m2, 'ser_m2': ser_m2,
        'th_low_m2': th_low_m2, 'th_high_m2': th_high_m2,
        'dac_offset_m2': dac_offset_m2,
        'delay_m3': delay_m3, 'ser_m3': ser_m3,
        'ffe_q': ffe_q, 'taps_norm': taps_norm,
        'th_low_m3': th_low_m3, 'th_high_m3': th_high_m3,
        'dac_offset_m3': dac_offset_m3,
    }


if __name__ == "__main__":
    main()
