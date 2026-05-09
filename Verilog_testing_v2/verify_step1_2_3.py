#!/usr/bin/env python3
"""
Bit-Exact Python Verification for pam4_serdes_step1_2_3.v
==========================================================

Models the Verilog pipeline cycle-by-cycle with integer arithmetic.
- Mode 1: Composite FIR -> threshold slicer -> SER  (~9.45%)
- Mode 2: Composite FIR + lane coupling -> slicer -> SER  (~14.2%)
- Mode 3: Composite FIR + coupling -> RX FFE -> slicer -> SER  (~0%)
"""

import numpy as np

SYMBOL_DIV = 4

H_COMP = [288, 84, -30, -27, -24, -17, -12, -7, -4, -2, -3, -2]
N_COMP = len(H_COMP)

# Mode 1 thresholds
THRESH_LOW  = -608
THRESH_MID  = 0
THRESH_HIGH = 608

# Coupling parameters
COUPLING_NUM = 17
COUPLING_SHIFT = 6  # 17/64 = 0.265625

# Mode 2 thresholds (coupled, no FFE)
COUPLED_THRESH_LOW  = -681
COUPLED_THRESH_MID  = 0
COUPLED_THRESH_HIGH = 681

# RX FFE taps (Q1.14, trained on coupled pipeline)
FFE_TAPS = [-274, -145, 474, 16384, -4674, 3018, 598, 1794]
N_FFE = len(FFE_TAPS)
N_FFE_PRE = 3
Q_FFE = 14

# Mode 3 thresholds (coupled + FFE)
FFE_THRESH_LOW  = -600
FFE_THRESH_MID  = 0
FFE_THRESH_HIGH = 600

PAM4_MAP = {0: -3, 1: -1, 2: 1, 3: 3}
NUM_SYMBOLS = 200000
WARMUP = 100
SER_WINDOW = 100000


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


def simulate():
    # Victim PRBS-7 (seed 0x7F)
    prbs7_v = 0x7F
    comp_sr_v = [0] * N_COMP
    comp_out_v = 0

    # Aggressor PRBS-7 (seed 0x55)
    prbs7_a = 0x55
    comp_sr_a = [0] * N_COMP
    comp_out_a = 0

    # Coupling output (registered)
    coupled_out = 0

    # FFE
    ffe_sr = [0] * N_FFE
    ffe_out = 0

    # Reference
    ref_sr = [0] * 32

    tx_symbols = []
    det_mode1_list = []
    det_mode2_list = []
    det_mode3_list = []

    print("Running bit-exact simulation of pam4_serdes_step1_2_3.v")
    print(f"  Coupling = {COUPLING_NUM}/{2**COUPLING_SHIFT} = "
          f"{COUPLING_NUM / 2**COUPLING_SHIFT:.4f}")
    print(f"  Mode1 thresholds: [{THRESH_LOW}, {THRESH_MID}, {THRESH_HIGH}]")
    print(f"  Mode2 thresholds: [{COUPLED_THRESH_LOW}, {COUPLED_THRESH_MID}, "
          f"{COUPLED_THRESH_HIGH}]")
    print(f"  Mode3 thresholds: [{FFE_THRESH_LOW}, {FFE_THRESH_MID}, "
          f"{FFE_THRESH_HIGH}]")
    print(f"  FFE taps: {FFE_TAPS}")

    symbol_count = 0

    for clk in range((NUM_SYMBOLS + 300) * SYMBOL_DIV):
        symbol_tick = (clk % SYMBOL_DIV == SYMBOL_DIV - 1)
        if not symbol_tick:
            continue

        # --- Victim PRBS-7 ---
        new_bit_v = ((prbs7_v >> 6) ^ (prbs7_v >> 5)) & 1
        prbs7_v = ((prbs7_v << 1) | new_bit_v) & 0x7F
        tx_sym_v = prbs7_v & 0x3
        pam4_v = PAM4_MAP[tx_sym_v]

        # --- Aggressor PRBS-7 ---
        new_bit_a = ((prbs7_a >> 6) ^ (prbs7_a >> 5)) & 1
        prbs7_a = ((prbs7_a << 1) | new_bit_a) & 0x7F
        tx_sym_a = prbs7_a & 0x3
        pam4_a = PAM4_MAP[tx_sym_a]

        # --- Victim FIR block (blocking acc, reads OLD comp_sr_v) ---
        comp_acc_v = sum(comp_sr_v[i] * H_COMP[i] for i in range(N_COMP))
        comp_acc_v_16 = to_signed_16(comp_acc_v)

        # Mode 1 slicer: uses blocking comp_acc_v (same block)
        det1 = slicer(comp_acc_v_16, THRESH_LOW, THRESH_MID, THRESH_HIGH)

        # Save old registered values
        old_comp_out_v = comp_out_v
        old_comp_out_a = comp_out_a

        # NB: victim FIR shift register + comp_out
        comp_sr_v = [pam4_v] + comp_sr_v[:-1]
        comp_out_v = comp_acc_v_16

        # --- Aggressor FIR block (blocking acc, reads OLD comp_sr_a) ---
        comp_acc_a = sum(comp_sr_a[i] * H_COMP[i] for i in range(N_COMP))
        comp_acc_a_16 = to_signed_16(comp_acc_a)

        # NB: aggressor FIR shift register + comp_out
        comp_sr_a = [pam4_a] + comp_sr_a[:-1]
        comp_out_a = comp_acc_a_16

        # --- Coupling block (reads OLD comp_out_v/a, blocking coupled_val) ---
        coupling_term = (old_comp_out_a * COUPLING_NUM) >> COUPLING_SHIFT
        coupled_val = old_comp_out_v + coupling_term
        if coupled_val > 32767:
            coupled_val = 32767
        elif coupled_val < -32768:
            coupled_val = -32768
        coupled_val = to_signed_16(coupled_val)

        # Mode 2 slicer: blocking coupled_val, same block
        det2 = slicer(coupled_val, COUPLED_THRESH_LOW, COUPLED_THRESH_MID,
                       COUPLED_THRESH_HIGH)

        # Save old coupled_out for FFE
        old_coupled_out = coupled_out

        # NB: coupled_out register
        coupled_out = coupled_val

        # --- FFE block (reads OLD coupled_out via ffe_sr, blocking ffe_acc) ---
        ffe_acc = sum(ffe_sr[i] * FFE_TAPS[i] for i in range(N_FFE))
        ffe_16 = to_signed_16(ffe_acc >> Q_FFE)

        # Mode 3 slicer: blocking ffe_acc, same block
        det3 = slicer(ffe_16, FFE_THRESH_LOW, FFE_THRESH_MID, FFE_THRESH_HIGH)

        # NB: FFE shift register (gets OLD coupled_out)
        ffe_sr = [old_coupled_out] + ffe_sr[:-1]
        ffe_out = ffe_16

        # Reference shift register
        ref_sr = [tx_sym_v] + ref_sr[:-1]

        tx_symbols.append(tx_sym_v)
        det_mode1_list.append(det1)
        det_mode2_list.append(det2)
        det_mode3_list.append(det3)

        symbol_count += 1
        if symbol_count >= NUM_SYMBOLS + 300:
            break

    tx_symbols = np.array(tx_symbols)
    det_mode1 = np.array(det_mode1_list)
    det_mode2 = np.array(det_mode2_list)
    det_mode3 = np.array(det_mode3_list)

    # ================================================================
    # MODE 1
    # ================================================================
    print("\n" + "=" * 60)
    print("MODE 1 (Composite FIR -> Slicer, no coupling)")
    print("=" * 60)
    best_delay1, best_ser1 = 0, 1.0
    for delay in range(20):
        ref = tx_symbols[WARMUP:len(tx_symbols) - delay]
        det = det_mode1[WARMUP + delay:]
        n = min(len(ref), len(det))
        if n > 1000:
            errors = np.sum(ref[:n] != det[:n])
            ser = errors / n
            if ser < best_ser1:
                best_ser1 = ser
                best_delay1 = delay
            if delay < 8:
                print(f"  delay={delay:2d}: SER={ser:.6f} ({errors}/{n})")

    print(f"\n  OPTIMAL: delay={best_delay1}, SER={best_ser1:.6f} "
          f"({best_ser1*100:.4f}%)")

    print(f"\n  100K-window SER:")
    total = 0; errs = 0; wi = 0
    for i in range(WARMUP, len(tx_symbols)):
        di = i + best_delay1
        if di >= len(det_mode1):
            break
        if tx_symbols[i] != det_mode1[di]:
            errs += 1
        total += 1
        if total == SER_WINDOW:
            ppm = errs * 10
            print(f"    Window {wi}: {errs}/{SER_WINDOW}, ser_ppm={ppm}, "
                  f"display: 1  {ppm//100000%10}{ppm//10000%10}."
                  f"{ppm//1000%10}{ppm//100%10}{ppm//10%10}{ppm%10}")
            total = 0; errs = 0; wi += 1

    # ================================================================
    # MODE 2
    # ================================================================
    print("\n" + "=" * 60)
    print("MODE 2 (Composite FIR + FEXT coupling -> Slicer)")
    print("=" * 60)
    best_delay2, best_ser2 = 0, 1.0
    for delay in range(20):
        ref = tx_symbols[WARMUP:len(tx_symbols) - delay]
        det = det_mode2[WARMUP + delay:]
        n = min(len(ref), len(det))
        if n > 1000:
            errors = np.sum(ref[:n] != det[:n])
            ser = errors / n
            if ser < best_ser2:
                best_ser2 = ser
                best_delay2 = delay
            if delay < 8:
                print(f"  delay={delay:2d}: SER={ser:.6f} ({errors}/{n})")

    print(f"\n  OPTIMAL: delay={best_delay2}, SER={best_ser2:.6f} "
          f"({best_ser2*100:.4f}%)")

    print(f"\n  100K-window SER:")
    total = 0; errs = 0; wi = 0
    for i in range(WARMUP, len(tx_symbols)):
        di = i + best_delay2
        if di >= len(det_mode2):
            break
        if tx_symbols[i] != det_mode2[di]:
            errs += 1
        total += 1
        if total == SER_WINDOW:
            ppm = errs * 10
            print(f"    Window {wi}: {errs}/{SER_WINDOW}, ser_ppm={ppm}, "
                  f"display: 2  {ppm//100000%10}{ppm//10000%10}."
                  f"{ppm//1000%10}{ppm//100%10}{ppm//10%10}{ppm%10}")
            total = 0; errs = 0; wi += 1

    # ================================================================
    # MODE 3
    # ================================================================
    print("\n" + "=" * 60)
    print("MODE 3 (Composite FIR + FEXT coupling -> RX FFE -> Slicer)")
    print("=" * 60)
    best_delay3, best_ser3 = 0, 1.0
    for delay in range(20):
        ref = tx_symbols[WARMUP:len(tx_symbols) - delay]
        det = det_mode3[WARMUP + delay:]
        n = min(len(ref), len(det))
        if n > 1000:
            errors = np.sum(ref[:n] != det[:n])
            ser = errors / n
            if ser < best_ser3:
                best_ser3 = ser
                best_delay3 = delay
            if delay < 12:
                print(f"  delay={delay:2d}: SER={ser:.6f} ({errors}/{n})")

    print(f"\n  OPTIMAL: delay={best_delay3}, SER={best_ser3:.6f} "
          f"({best_ser3*100:.4f}%)")

    print(f"\n  100K-window SER:")
    total = 0; errs = 0; wi = 0
    for i in range(WARMUP, len(tx_symbols)):
        di = i + best_delay3
        if di >= len(det_mode3):
            break
        if tx_symbols[i] != det_mode3[di]:
            errs += 1
        total += 1
        if total == SER_WINDOW:
            ppm = errs * 10
            print(f"    Window {wi}: {errs}/{SER_WINDOW}, ser_ppm={ppm}, "
                  f"display: 3  {ppm//100000%10}{ppm//10000%10}."
                  f"{ppm//1000%10}{ppm//100%10}{ppm//10%10}{ppm%10}")
            total = 0; errs = 0; wi += 1

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*60}")
    print(f"SUMMARY / VERILOG PARAMETERS")
    print(f"{'='*60}")
    print(f"  SYMBOL_DELAY_1 = {best_delay1}  -> Mode 1 SER: {best_ser1*100:.4f}%")
    print(f"  SYMBOL_DELAY_2 = {best_delay2}  -> Mode 2 SER: {best_ser2*100:.4f}%")
    print(f"  SYMBOL_DELAY_3 = {best_delay3}  -> Mode 3 SER: {best_ser3*100:.4f}%")
    print(f"{'='*60}")

    return best_delay1, best_ser1, best_delay2, best_ser2, best_delay3, best_ser3


if __name__ == "__main__":
    simulate()
