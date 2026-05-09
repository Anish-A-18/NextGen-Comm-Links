#!/usr/bin/env python3
"""
Extract Symbol-Rate FIR Taps + RX FFE Taps for Verilog from SerdesSystem
=========================================================================

1. Composite FIR (TX_FFE * Channel * CTLE): estimated via LS from the
   SerdesSystem's (tx_symbols -> ADC output) I/O, scaled for Verilog.

2. RX FFE: trained on the *Verilog-domain* composite FIR output (integer
   signal, ~+-1500 range) using LMS, so the taps and thresholds are
   bit-exact compatible with the Verilog implementation.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Serdes_Final_Vault_ML_MLSE_DFT_v3'))

from serdes_system import ConfigSerdesSystem, SerdesSystem

CHANNEL_LENGTH = 0.30
SYMBOL_RATE = 64e9
SAMPLES_PER_SYMBOL = 32

PAM4_LEVELS = np.array([-3, -1, 1, 3])


def main():
    print("=" * 70)
    print("EXTRACT COMPOSITE FIR + RX FFE FOR VERILOG (30cm)")
    print("=" * 70)

    config = ConfigSerdesSystem(
        symbol_rate=SYMBOL_RATE,
        samples_per_symbol=SAMPLES_PER_SYMBOL,
        channel_length=CHANNEL_LENGTH,
        channel_z0=50.0,
        channel_eps_r=4.9,
        channel_cap_source=100e-15,
        channel_cap_term=150e-15,
        channel_theta_0=0.015,
        channel_k_r=120,
        channel_rdc=0.0002,
        adaptation_symbols=32000,
        total_symbols=32000 + 10000 + 10000 + 100000,
        pam4_seed=42,
        noise_seed=41,
        snr_db=30,
        adc_n_bits=8,
        adc_target_rms=0.5,
        adc_full_scale_voltage=2.0,
        non_linearity_gain_compression=0.15,
        non_linearity_third_order=0.08,
        verbose=False
    )

    system = SerdesSystem(config)
    results = system.run_ctle_adc()

    tx_symbols = results['tx_symbols']
    adc_output = results['adc_output']
    print(f"  TX symbols: {len(tx_symbols)}, ADC output: {len(adc_output)}")
    print(f"  Best CTLE config: {results['ctle_best_config']}")

    # ================================================================
    # 1. Estimate composite channel (LS fit: tx_levels -> adc_output)
    # ================================================================
    L = 11
    N_est = min(50000, len(tx_symbols))
    tx_levels = PAM4_LEVELS[tx_symbols[:N_est]]
    rx_signal = adc_output[:N_est]

    S = np.zeros((N_est - L, L + 1))
    for n in range(L, N_est):
        for k in range(L + 1):
            S[n - L, k] = tx_levels[n - k]
    R = rx_signal[L:N_est]
    h_comp = np.linalg.lstsq(S, R, rcond=None)[0]

    predicted = S @ h_comp
    mse = np.mean((R - predicted) ** 2)
    snr = 10 * np.log10(np.mean(R ** 2) / (mse + 1e-10))
    print(f"\n  Composite channel: L={L}, SNR={snr:.1f} dB")

    # ================================================================
    # 2. Quantize composite FIR for Verilog
    # ================================================================
    target_half_range = 1500
    max_out_per_unit = np.sum(np.abs(h_comp)) * 3
    SCALE = target_half_range / max_out_per_unit
    h_q = np.round(h_comp * SCALE).astype(int)

    N_ver = min(100000, len(tx_symbols))
    tx_int = PAM4_LEVELS[tx_symbols[:N_ver]]
    fir_out = np.convolve(tx_int, h_q)[:N_ver]

    fir_rms = np.sqrt(np.mean(fir_out[100:].astype(np.float64) ** 2))
    th_low  = int(np.round(-fir_rms * 0.895))
    th_mid  = 0
    th_high = int(np.round( fir_rms * 0.895))

    print(f"  SCALE={SCALE:.2f}")
    print(f"  FIR range: [{int(np.min(fir_out[100:]))}, {int(np.max(fir_out[100:]))}]")
    print(f"  FIR RMS: {fir_rms:.1f}")
    print(f"  Thresholds: [{th_low}, {th_mid}, {th_high}]")

    # Composite SER
    det_q = np.zeros(N_ver, dtype=int)
    det_q[fir_out < th_low] = 0
    det_q[(fir_out >= th_low) & (fir_out < th_mid)] = 1
    det_q[(fir_out >= th_mid) & (fir_out < th_high)] = 2
    det_q[fir_out >= th_high] = 3

    best_ser_q = 1.0
    best_delay_q = 0
    for d in range(20):
        ref = tx_symbols[100:N_ver - d]
        det = det_q[100 + d:N_ver]
        n = min(len(ref), len(det))
        if n > 1000:
            ser = np.sum(ref[:n] != det[:n]) / n
            if ser < best_ser_q:
                best_ser_q = ser
                best_delay_q = d
    print(f"  Composite SER: {best_ser_q:.4f} (delay={best_delay_q})")

    # DAC offset
    dac_offset = int(np.min(fir_out[100:])) - 100
    print(f"  DAC offset: {dac_offset}")

    # ================================================================
    # 3. RX FFE: Train on integer-domain FIR output
    # ================================================================
    print("\n" + "=" * 70)
    print("RX FFE (LMS, 8 taps, trained on Verilog-domain integer signal)")
    print("=" * 70)

    N_FFE_TAPS = 8
    N_PRE = 3
    N_POST = 4
    Q_FFE = 14
    N_TRAIN = 80000

    # Simulate the Verilog composite FIR pipeline (with 1-symbol delay
    # from non-blocking shift register) to produce the exact signal the
    # FFE will see in hardware.
    print(f"  Simulating Verilog FIR pipeline for FFE training signal...")
    comp_sr_sim = [0] * len(h_q)
    fir_pipeline = []
    for i in range(len(tx_symbols)):
        pam4 = PAM4_LEVELS[tx_symbols[i]]
        comp_acc_sim = sum(comp_sr_sim[k] * int(h_q[k]) for k in range(len(h_q)))
        fir_pipeline.append(comp_acc_sim & 0xFFFF)
        comp_sr_sim = [pam4] + comp_sr_sim[:-1]

    fir_pipeline = np.array(fir_pipeline, dtype=np.int64)
    # Convert to signed 16-bit
    fir_pipeline = np.where(fir_pipeline >= 0x8000,
                            fir_pipeline.astype(np.int64) - 0x10000,
                            fir_pipeline.astype(np.int64))

    # Now fir_pipeline[i] is what comp_out would be after tick i in Verilog.
    # But comp_out is registered (non-blocking), so the FFE's ffe_sr[0] gets
    # old_comp_out from the PREVIOUS tick: 1 delay.
    # Then ffe_acc reads OLD ffe_sr (before non-blocking update): another 1 delay.
    # Total: ffe_acc at tick T reads comp_out values from ticks T-2, T-3, ...
    #
    # In the verify script:
    #   old_comp_out = comp_out  (from previous tick)
    #   ffe_acc = FIR on OLD ffe_sr
    #   ffe_sr = [old_comp_out] + ffe_sr[:-1]
    #
    # So ffe_acc at tick T uses ffe_sr from before the update.
    # ffe_sr[0] was set to old_comp_out at tick T-1 (i.e. comp_out from T-2).
    # ffe_sr[1] was set to old_comp_out at tick T-2 (i.e. comp_out from T-3).
    # etc.
    #
    # So the FFE input buffer at tick T contains:
    #   ffe_sr[0] = comp_out[T-2], ffe_sr[1] = comp_out[T-3], ...
    # i.e. 2-tick delay from comp_out to FFE processing.
    #
    # comp_out[T] = comp_acc from FIR on OLD comp_sr at tick T
    #             ~ FIR response of pam4_level[T-1] (1 delay from comp_sr NB)
    #
    # So overall: ffe_sr[k] at tick T corresponds to pam4_level[T - 2 - 1 - k]
    #                                                = pam4_level[T - 3 - k]
    # The FFE cursor (tap index N_PRE=3) reads ffe_sr[3], corresponding to
    # pam4_level[T - 3 - 3] = pam4_level[T-6].
    # And det_mode2 reads OLD ffe_out (1 more delay), making it T-7.
    # So total delay from tx_symbol to det_mode2 = 7.

    fir_for_ffe = np.zeros_like(fir_pipeline)
    fir_for_ffe[2:] = fir_pipeline[:-2]  # 2-tick delay (comp_out reg + ffe_sr NB)

    fir_signal = fir_for_ffe.astype(np.float64)

    # Normalize for stable LMS
    sig_rms = np.sqrt(np.mean(fir_signal[200:N_TRAIN] ** 2))
    fir_norm = fir_signal / sig_rms

    # Desired: PAM4 levels scaled by cursor gain, accounting for pipeline delay.
    # comp_acc at tick i corresponds to tx_symbols[i-1] as the "cursor" (due
    # to the non-blocking shift register delay). Then ffe_sr[0] gets comp_out
    # from tick i (available at tick i+1). So fir_for_ffe[i+1] corresponds to
    # tx_symbols[i-1]. The FFE cursor (N_PRE=3 taps back) aligns to
    # fir_for_ffe[i+1-3] ~ tx_symbols[i-1-3] = tx_symbols[i-4].
    #
    # More directly: let's find the best alignment empirically.
    cursor_gain = h_q[0]
    best_train_delay = 0
    best_train_corr = 0
    for d in range(-5, 10):
        start = max(200, 200 + d)
        end = min(N_TRAIN, N_TRAIN - abs(d))
        if d >= 0:
            des = PAM4_LEVELS[tx_symbols[200:end - d]] * cursor_gain
            sig = fir_for_ffe[200 + d:end]
        else:
            des = PAM4_LEVELS[tx_symbols[200 - d:end]] * cursor_gain
            sig = fir_for_ffe[200:end + d]
        n = min(len(des), len(sig))
        if n > 1000:
            corr = np.abs(np.corrcoef(des[:n], sig[:n])[0, 1])
            if corr > best_train_corr:
                best_train_corr = corr
                best_train_delay = d
    print(f"  Best FFE training alignment: delay={best_train_delay}, corr={best_train_corr:.4f}")

    # Build desired signal with correct alignment
    desired = np.zeros(len(fir_signal))
    for i in range(len(tx_symbols)):
        sym_idx = i - best_train_delay
        if 0 <= sym_idx < len(tx_symbols):
            desired[i] = PAM4_LEVELS[tx_symbols[sym_idx]] * cursor_gain
    desired_norm = desired / sig_rms

    MU = 0.005

    taps_f = np.zeros(N_FFE_TAPS, dtype=np.float64)
    taps_f[N_PRE] = 1.0

    buf = np.zeros(N_FFE_TAPS, dtype=np.float64)

    print(f"  Training {N_TRAIN} symbols, mu={MU}, sig_rms={sig_rms:.1f}")
    mse_history = []
    for i in range(N_PRE, N_TRAIN + N_PRE):
        for k in range(N_FFE_TAPS):
            idx = i - N_PRE + k
            if 0 <= idx < len(fir_norm):
                buf[k] = fir_norm[idx]
            else:
                buf[k] = 0

        y = np.dot(taps_f, buf)
        e = desired_norm[i] - y
        taps_f += 2 * MU * e * buf
        if i % 1000 == 0:
            mse_history.append(e * e)

    print(f"  Final MSE: {mse_history[-1]:.4f}")
    print(f"  Float taps (normalized training): {taps_f}")

    # Normalize so cursor tap = 1.0
    cursor_val = taps_f[N_PRE]
    if abs(cursor_val) > 1e-10:
        taps_norm = taps_f / cursor_val
    else:
        taps_norm = taps_f

    # Quantize to Q1.14
    ffe_q = np.round(taps_norm * (2 ** Q_FFE)).astype(int)
    print(f"  Normalized taps: {taps_norm}")
    print(f"  Q1.14 taps: {ffe_q}")

    # ================================================================
    # 4. Verify FFE in integer domain (bit-exact)
    # ================================================================
    print("\n" + "=" * 70)
    print("FFE VERIFICATION (bit-exact integer, using Verilog pipeline signal)")
    print("=" * 70)

    # Apply FFE to the pipeline-accurate FIR signal
    N_test = min(100000, len(fir_for_ffe))
    ffe_output = np.zeros(N_test, dtype=np.int64)
    for i in range(N_PRE, N_test - N_POST):
        acc = 0
        for k in range(N_FFE_TAPS):
            idx = i - N_PRE + k
            if 0 <= idx < N_test:
                acc += int(fir_for_ffe[idx]) * int(ffe_q[k])
        ffe_output[i] = acc >> Q_FFE

    # Compute thresholds for FFE output
    ffe_valid = ffe_output[200:N_test - N_POST]
    ffe_rms = np.sqrt(np.mean(ffe_valid.astype(np.float64) ** 2))
    ffe_th_low  = int(np.round(-ffe_rms * 0.895))
    ffe_th_mid  = 0
    ffe_th_high = int(np.round( ffe_rms * 0.895))
    print(f"  FFE output RMS: {ffe_rms:.1f}")
    print(f"  FFE output range: [{int(np.min(ffe_valid))}, {int(np.max(ffe_valid))}]")
    print(f"  FFE thresholds: [{ffe_th_low}, {ffe_th_mid}, {ffe_th_high}]")

    # Slice
    det_ffe = np.zeros(N_test, dtype=int)
    det_ffe[ffe_output < ffe_th_low] = 0
    det_ffe[(ffe_output >= ffe_th_low) & (ffe_output < ffe_th_mid)] = 1
    det_ffe[(ffe_output >= ffe_th_mid) & (ffe_output < ffe_th_high)] = 2
    det_ffe[ffe_output >= ffe_th_high] = 3

    best_ser_ffe = 1.0
    best_delay_ffe = 0
    for d in range(-5, 20):
        start = max(200, 200 + d)
        end = N_test - N_POST
        if d >= 0:
            ref = tx_symbols[200:end - d]
            det = det_ffe[200 + d:end]
        else:
            ref = tx_symbols[200 - d:end]
            det = det_ffe[200:end + d]
        n = min(len(ref), len(det))
        if n > 1000:
            ser = np.sum(ref[:n] != det[:n]) / n
            if ser < best_ser_ffe:
                best_ser_ffe = ser
                best_delay_ffe = d
            if -3 <= d <= 10:
                print(f"    delay={d:+3d}: SER={ser:.6f} ({int(ser*n)}/{n})")

    print(f"\n  FFE optimal: delay={best_delay_ffe}, SER={best_ser_ffe:.6f} ({best_ser_ffe*100:.4f}%)")

    # Per-symbol FFE stats
    for sym in range(4):
        d = best_delay_ffe
        if d >= 0:
            mask = tx_symbols[200:N_test - N_POST - d] == sym
            vals = ffe_output[200 + d:N_test - N_POST][mask[:len(ffe_output[200+d:N_test - N_POST])]]
        else:
            mask = tx_symbols[200 - d:N_test - N_POST] == sym
            vals = ffe_output[200:N_test - N_POST + d][mask[:len(ffe_output[200:N_test - N_POST + d])]]
        if len(vals) > 0:
            print(f"  Symbol {sym}: mean={np.mean(vals):.1f}, std={np.std(vals):.1f}")

    # DAC offset for FFE output
    ffe_dac_offset = int(np.min(ffe_valid)) - 100

    # ================================================================
    # 5. Write Verilog parameters
    # ================================================================
    print("\n" + "=" * 70)
    print("WRITING VERILOG PARAMETERS")
    print("=" * 70)

    lines = []
    lines.append("// ============================================")
    lines.append("// AUTO-GENERATED VERILOG PARAMETERS")
    lines.append(f"// Channel length: {CHANNEL_LENGTH*100:.0f} cm")
    lines.append(f"// CTLE config: {results['ctle_best_config']}")
    lines.append(f"// Composite SER: {best_ser_q:.4e} (delay={best_delay_q})")
    lines.append(f"// FFE SER: {best_ser_ffe:.4e} (delay={best_delay_ffe})")
    lines.append(f"// SCALE={SCALE:.2f}")
    lines.append("// ============================================")
    lines.append("")

    lines.append(f"// Composite Channel FIR ({len(h_q)} taps)")
    for i, q in enumerate(h_q):
        s = "" if q >= 0 else "-"
        lines.append(f"localparam signed [15:0] H_COMP_{i} = {s}16'sd{abs(q)};  "
                      f"// {h_comp[i]:+.6f}")

    lines.append("")
    lines.append(f"// Step 1 Threshold slicer levels")
    lines.append(f"localparam signed [15:0] THRESH_LOW  = {'-' if th_low < 0 else ''}16'sd{abs(th_low)};")
    lines.append(f"localparam signed [15:0] THRESH_MID  =  16'sd{th_mid};")
    lines.append(f"localparam signed [15:0] THRESH_HIGH =  16'sd{th_high};")
    lines.append(f"localparam SYMBOL_DELAY = {best_delay_q};")

    lines.append("")
    lines.append(f"// DAC output mapping (Step 1)")
    lines.append(f"localparam signed [15:0] DAC_OFFSET = {'-' if dac_offset < 0 else ''}16'sd{abs(dac_offset)};")

    lines.append("")
    labels_ffe = ['PRE3', 'PRE2', 'PRE1', 'MAIN', 'POST1', 'POST2', 'POST3', 'POST4']
    lines.append(f"// RX FFE ({N_FFE_TAPS} taps, Q1.{Q_FFE}, trained on integer FIR output)")
    for i, (label, q) in enumerate(zip(labels_ffe, ffe_q)):
        s = "" if q >= 0 else "-"
        lines.append(f"localparam signed [15:0] FFE_{label} = {s}16'sd{abs(q)};  "
                      f"// {taps_norm[i]:+.6f}")

    lines.append("")
    lines.append(f"// Step 2 FFE Threshold slicer levels")
    lines.append(f"localparam signed [15:0] FFE_THRESH_LOW  = {'-' if ffe_th_low < 0 else ''}16'sd{abs(ffe_th_low)};")
    lines.append(f"localparam signed [15:0] FFE_THRESH_MID  =  16'sd{ffe_th_mid};")
    lines.append(f"localparam signed [15:0] FFE_THRESH_HIGH =  16'sd{abs(ffe_th_high)};")
    lines.append(f"localparam FFE_SYMBOL_DELAY = {best_delay_ffe};")

    lines.append("")
    lines.append(f"// DAC output mapping (Step 2)")
    lines.append(f"localparam signed [15:0] FFE_DAC_OFFSET = {'-' if ffe_dac_offset < 0 else ''}16'sd{abs(ffe_dac_offset)};")

    text = "\n".join(lines)
    outpath = os.path.join(os.path.dirname(__file__), "verilog_params_v2.txt")
    with open(outpath, 'w') as f:
        f.write(text)
    print(f"  Saved to {outpath}")

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Composite FIR: {len(h_q)} taps, SER={best_ser_q*100:.2f}% (delay={best_delay_q})")
    print(f"  FFE: {N_FFE_TAPS} taps, SER={best_ser_ffe*100:.4f}% (delay={best_delay_ffe})")
    print(f"  Step 1 thresholds: [{th_low}, {th_mid}, {th_high}]")
    print(f"  Step 2 thresholds: [{ffe_th_low}, {ffe_th_mid}, {ffe_th_high}]")
    print(f"{'='*70}")

    return {
        'h_comp': h_comp, 'h_q': h_q, 'SCALE': SCALE,
        'th_low': th_low, 'th_mid': th_mid, 'th_high': th_high,
        'best_delay_q': best_delay_q, 'best_ser_q': best_ser_q,
        'dac_offset': dac_offset,
        'ffe_q': ffe_q, 'ffe_taps_norm': taps_norm,
        'ffe_th_low': ffe_th_low, 'ffe_th_mid': ffe_th_mid, 'ffe_th_high': ffe_th_high,
        'best_delay_ffe': best_delay_ffe, 'best_ser_ffe': best_ser_ffe,
        'ffe_dac_offset': ffe_dac_offset,
    }


if __name__ == "__main__":
    main()
