#!/usr/bin/env python3
"""
Train RX FFE taps on the bit-exact Verilog pipeline signal.

This script simulates the Verilog composite FIR pipeline exactly as in
verify_step1.py, generates the FFE input buffer values that the Verilog
would see, and trains the FFE taps via LMS on that exact signal.
"""

import numpy as np

H_COMP = [288, 84, -30, -27, -24, -17, -12, -7, -4, -2, -3, -2]
N_COMP = len(H_COMP)
PAM4_MAP = {0: -3, 1: -1, 2: 1, 3: 3}

N_FFE = 8
N_PRE = 3
Q_FFE = 14
SYMBOL_DIV = 4
NUM_SYMBOLS = 120000


def to_signed_16(val):
    val = val & 0xFFFF
    return val - 0x10000 if val >= 0x8000 else val


def main():
    print("=" * 70)
    print("TRAIN FFE ON EXACT VERILOG PIPELINE")
    print("=" * 70)

    # Simulate the Verilog composite FIR pipeline exactly
    prbs7 = 0x7F
    comp_sr = [0] * N_COMP
    comp_out = 0

    # Collect the FFE input buffer and TX symbols
    ffe_sr = [0] * N_FFE
    tx_symbols = []
    ffe_buffers = []

    for clk in range((NUM_SYMBOLS + 200) * SYMBOL_DIV):
        symbol_tick = (clk % SYMBOL_DIV == SYMBOL_DIV - 1)
        if not symbol_tick:
            continue

        new_bit = ((prbs7 >> 6) ^ (prbs7 >> 5)) & 1
        prbs7 = ((prbs7 << 1) | new_bit) & 0x7F
        tx_sym = prbs7 & 0x3
        pam4 = PAM4_MAP[tx_sym]

        # Composite FIR (blocking acc, reads OLD comp_sr)
        comp_acc = sum(comp_sr[i] * H_COMP[i] for i in range(N_COMP))
        comp_acc_16 = to_signed_16(comp_acc)

        # Save old comp_out for FFE
        old_comp_out = comp_out

        # Non-blocking updates
        comp_sr = [pam4] + comp_sr[:-1]
        comp_out = comp_acc_16

        # FFE sees OLD ffe_sr (before update)
        ffe_buffers.append(list(ffe_sr))

        # Non-blocking FFE shift register update
        ffe_sr = [old_comp_out] + ffe_sr[:-1]

        tx_symbols.append(tx_sym)
        if len(tx_symbols) >= NUM_SYMBOLS + 200:
            break

    tx_symbols = np.array(tx_symbols)
    ffe_buffers = np.array(ffe_buffers, dtype=np.float64)

    print(f"  Collected {len(tx_symbols)} symbols, {len(ffe_buffers)} FFE buffers")

    # Find the alignment: which tx_symbol does ffe_buffers[i][N_PRE] correspond to?
    # The cursor tap (index N_PRE=3) in the FFE buffer at tick T should correspond
    # to a specific tx_symbol[T - delay].
    cursor_signal = ffe_buffers[:, N_PRE]  # what the main tap sees
    # Compute correlation with tx PAM4 levels at various delays
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
            if -2 <= d <= 10:
                print(f"    cursor corr at delay={d:+3d}: {corr:.4f}")

    print(f"  Best cursor alignment: delay={best_d}, corr={best_corr:.4f}")

    # The desired output of the FFE should be the PAM4 level times cursor_gain
    cursor_gain = H_COMP[0]  # 288
    desired = np.zeros(len(ffe_buffers))
    for i in range(len(ffe_buffers)):
        sym_idx = i - best_d
        if 0 <= sym_idx < len(tx_symbols):
            desired[i] = pam4_levels[tx_symbols[sym_idx]] * cursor_gain

    # Normalize for LMS
    sig_rms = np.sqrt(np.mean(cursor_signal[200:] ** 2))
    print(f"  Cursor signal RMS: {sig_rms:.1f}")

    # LMS training on normalized signals
    ffe_norm = ffe_buffers / sig_rms
    desired_norm = desired / sig_rms

    MU = 0.01
    N_TRAIN = 80000
    taps = np.zeros(N_FFE, dtype=np.float64)
    taps[N_PRE] = 1.0

    print(f"\n  LMS training: {N_TRAIN} symbols, mu={MU}")
    for i in range(200, 200 + N_TRAIN):
        buf = ffe_norm[i]
        y = np.dot(taps, buf)
        e = desired_norm[i] - y
        taps += 2 * MU * e * buf

    print(f"  Trained taps: {taps}")

    # Normalize so cursor=1.0
    cursor_val = taps[N_PRE]
    if abs(cursor_val) > 1e-10:
        taps_norm = taps / cursor_val
    else:
        taps_norm = taps

    # Quantize to Q1.14
    ffe_q = np.round(taps_norm * (2 ** Q_FFE)).astype(int)
    print(f"  Normalized: {taps_norm}")
    print(f"  Q1.14: {ffe_q}")

    # Verify: apply Q1.14 FFE to the exact Verilog pipeline
    print(f"\n  Verifying bit-exact FFE...")
    ffe_output = np.zeros(len(ffe_buffers), dtype=np.int64)
    for i in range(len(ffe_buffers)):
        acc = 0
        for k in range(N_FFE):
            acc += int(ffe_buffers[i][k]) * int(ffe_q[k])
        ffe_output[i] = acc >> Q_FFE

    # Stats per symbol
    ffe_signed = ffe_output.astype(np.int64)
    ffe_rms = np.sqrt(np.mean(ffe_signed[200:].astype(np.float64) ** 2))
    th_low = int(np.round(-ffe_rms * 0.895))
    th_mid = 0
    th_high = int(np.round(ffe_rms * 0.895))
    print(f"  FFE output RMS: {ffe_rms:.1f}")
    print(f"  FFE thresholds: [{th_low}, {th_mid}, {th_high}]")

    # Slice
    det = np.zeros(len(ffe_output), dtype=int)
    det[ffe_signed < th_low] = 0
    det[(ffe_signed >= th_low) & (ffe_signed < th_mid)] = 1
    det[(ffe_signed >= th_mid) & (ffe_signed < th_high)] = 2
    det[ffe_signed >= th_high] = 3

    # Find best SER delay
    best_ser = 1.0
    best_delay = 0
    for d in range(-2, 20):
        start = 200
        end = len(tx_symbols)
        if d >= 0:
            ref = tx_symbols[start:end - d]
            dt = det[start + d:end]
        else:
            ref = tx_symbols[start - d:end]
            dt = det[start:end + d]
        n = min(len(ref), len(dt))
        if n > 1000:
            errors = np.sum(ref[:n] != dt[:n])
            ser = errors / n
            if ser < best_ser:
                best_ser = ser
                best_delay = d
            if -2 <= d <= 12:
                print(f"    delay={d:+3d}: SER={ser:.6f} ({errors}/{n})")

    print(f"\n  OPTIMAL: delay={best_delay}, SER={best_ser:.6f} ({best_ser*100:.4f}%)")

    # Per-symbol stats at optimal delay
    for sym in range(4):
        if best_delay >= 0:
            mask = tx_symbols[200:len(tx_symbols) - best_delay] == sym
            vals = ffe_signed[200 + best_delay:len(ffe_signed)]
        else:
            mask = tx_symbols[200 - best_delay:len(tx_symbols)] == sym
            vals = ffe_signed[200:len(ffe_signed) + best_delay]
        n = min(len(mask), len(vals))
        sym_vals = vals[:n][mask[:n]]
        if len(sym_vals) > 0:
            print(f"  Symbol {sym}: mean={np.mean(sym_vals):.1f}, std={np.std(sym_vals):.1f}")

    # Output parameters
    labels = ['PRE3', 'PRE2', 'PRE1', 'MAIN', 'POST1', 'POST2', 'POST3', 'POST4']
    print(f"\n{'='*70}")
    print(f"VERILOG PARAMETERS")
    print(f"{'='*70}")
    for i, (label, q) in enumerate(zip(labels, ffe_q)):
        s = "" if q >= 0 else "-"
        print(f"  localparam signed [15:0] FFE_{label} = {s}16'sd{abs(q)};  "
              f"// {taps_norm[i]:+.6f}")
    print(f"  FFE_THRESH_LOW  = {'-' if th_low < 0 else ''}16'sd{abs(th_low)};")
    print(f"  FFE_THRESH_MID  =  16'sd{th_mid};")
    print(f"  FFE_THRESH_HIGH =  16'sd{th_high};")

    ffe_dac_offset = int(np.min(ffe_signed[200:])) - 100
    print(f"  DAC_OFFSET_2 = {'-' if ffe_dac_offset < 0 else ''}16'sd{abs(ffe_dac_offset)};")

    # The total pipeline delay from tx_symbol to det_mode2 when using
    # blocking ffe_acc in the same block:
    # best_delay is from tx_symbols to det (which uses ffe_acc, blocking).
    # But in Verilog, det_mode2 uses non-blocking (<=), adding 1 more tick.
    # So SYMBOL_DELAY_2 = best_delay + 1
    print(f"\n  FFE cursor delay from tx_symbol: {best_d}")
    print(f"  SER alignment delay (ffe_acc -> det, blocking): {best_delay}")
    print(f"  SYMBOL_DELAY_2 for Verilog (det_mode2 is NB registered): {best_delay}")
    print(f"  Expected SER: {best_ser*100:.4f}%")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
