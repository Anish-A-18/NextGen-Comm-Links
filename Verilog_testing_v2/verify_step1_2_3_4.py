#!/usr/bin/env python3
"""
Bit-Exact Python Verification for pam4_serdes_step1_2_3.v (4 modes)
=====================================================================

Models the Verilog pipeline cycle-by-cycle with integer arithmetic.
- Mode 1: Composite FIR -> threshold slicer -> SER  (~9.45%)
- Mode 2: Composite FIR + lane coupling -> slicer -> SER  (~19.7%)
- Mode 3: Composite FIR + coupling -> RX FFE -> slicer -> SER  (~0.79%)
- Mode 4: Composite FIR + coupling -> ML NN  -> argmax -> SER  (~0%)
"""

import numpy as np

SYMBOL_DIV = 4

H_COMP = [288, 84, -30, -27, -24, -17, -12, -7, -4, -2, -3, -2]
N_COMP = len(H_COMP)

THRESH_LOW  = -608
THRESH_MID  = 0
THRESH_HIGH = 608

COUPLING_NUM = 17
COUPLING_SHIFT = 6

COUPLED_THRESH_LOW  = -681
COUPLED_THRESH_MID  = 0
COUPLED_THRESH_HIGH = 681

FFE_TAPS = [-274, -145, 474, 16384, -4674, 3018, 598, 1794]
N_FFE = len(FFE_TAPS)
Q_FFE = 14

FFE_THRESH_LOW  = -600
FFE_THRESH_MID  = 0
FFE_THRESH_HIGH = 600

# ML Network parameters (Q6.10, trained on raw shifted integers)
ML_INPUT_SHIFT = 6
Q_ML = 10

# Layer 1 weights (23 x 4), row-major
ML_W1 = np.array([
    [ -4,  10, -11, -18], [  3, -14,  14,   0], [-26,  27,  -4,  -4],
    [ 95, -59,  16,  10], [234, 153,  50, 369], [  2,   9, 135,  40],
    [-10,   1, -21,  -9], [ 57,  21,  14,  56], [-12, -28, -12, -28],
    [-52, 100,  86,  67], [674,-290,-167, 139], [162, 219, 543,-111],
    [ -1,   0,  36, -84], [ 64,  -4,  24,  52], [ 59,   1,  24,  -4],
    [ 22,   6,  37, -28], [-90,  38,  32,  -6], [-13, -18, -85,  42],
    [ 31, -28, -22,  26], [  0,  24,  17,  -4], [-14, -12,   5,  -8],
    [ 24,   3, -19,  14], [ 24,  -5,  25, -13],
], dtype=np.int64)
ML_B1 = np.array([74, 24, 53, 31], dtype=np.int64)

ML_W2 = np.array([
    [  25, 1062, -361,  823],
    [-810, -983,  468, -274],
    [ 492,-1429,  626, 1940],
    [-260,  382,  164,-1337],
], dtype=np.int64)
ML_B2 = np.array([1540, 0, -10, 0], dtype=np.int64)

ML_W3 = np.array([
    [1480,-1474, -604,  335],
    [-590, 1058,  226,   72],
    [-1056, -158, 1452,-1520],
    [-314, -448,   79,  713],
], dtype=np.int64)
ML_B3 = np.array([1904, 91, -866, -1129], dtype=np.int64)

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


def asr(val, shift):
    """Arithmetic right shift matching Verilog >>> on signed values."""
    if val >= 0:
        return val >> shift
    else:
        return -((-val - 1) >> shift) - 1


def ml_forward(ml_sr_snapshot):
    """Fixed-point ML inference matching the Verilog ML block."""
    # Pre-scale inputs
    scaled = [asr(int(v), ML_INPUT_SHIFT) for v in ml_sr_snapshot]

    # Layer 1: 23->4 ReLU
    h1 = [0] * 4
    for k in range(4):
        acc = 0
        for j in range(23):
            acc += scaled[j] * int(ML_W1[j, k])
        acc += int(ML_B1[k])
        val = asr(acc, Q_ML)
        h1[k] = max(0, val)

    # Layer 2: 4->4 ReLU
    h2 = [0] * 4
    for k in range(4):
        acc = 0
        for j in range(4):
            acc += h1[j] * int(ML_W2[j, k])
        acc += int(ML_B2[k])
        val = asr(acc, Q_ML)
        h2[k] = max(0, val)

    # Output layer: 4->4 logits
    logits = [0] * 4
    for k in range(4):
        acc = 0
        for j in range(4):
            acc += h2[j] * int(ML_W3[j, k])
        acc += int(ML_B3[k])
        logits[k] = asr(acc, Q_ML)

    # Argmax (same tie-breaking as Verilog)
    if logits[0] >= logits[1] and logits[0] >= logits[2] and logits[0] >= logits[3]:
        return 0
    elif logits[1] >= logits[2] and logits[1] >= logits[3]:
        return 1
    elif logits[2] >= logits[3]:
        return 2
    else:
        return 3


def simulate():
    prbs7_v = 0x7F
    comp_sr_v = [0] * N_COMP
    comp_out_v = 0

    prbs7_a = 0x55
    comp_sr_a = [0] * N_COMP
    comp_out_a = 0

    coupled_out = 0

    ffe_sr = [0] * N_FFE
    ml_sr = [0] * 23

    ref_sr = [0] * 32

    det_mode1_list = []
    det_mode2_list = []
    det_mode3_list = []
    det_mode4_list = []
    tx_symbols = []

    print("Running bit-exact simulation (4 modes)")
    print(f"  Coupling = {COUPLING_NUM}/{2**COUPLING_SHIFT}")

    symbol_count = 0

    for clk in range((NUM_SYMBOLS + 300) * SYMBOL_DIV):
        if clk % SYMBOL_DIV != SYMBOL_DIV - 1:
            continue

        # Victim PRBS-7
        new_bit_v = ((prbs7_v >> 6) ^ (prbs7_v >> 5)) & 1
        prbs7_v = ((prbs7_v << 1) | new_bit_v) & 0x7F
        tx_sym_v = prbs7_v & 0x3
        pam4_v = PAM4_MAP[tx_sym_v]

        # Aggressor PRBS-7
        new_bit_a = ((prbs7_a >> 6) ^ (prbs7_a >> 5)) & 1
        prbs7_a = ((prbs7_a << 1) | new_bit_a) & 0x7F
        pam4_a = PAM4_MAP[prbs7_a & 0x3]

        # Victim FIR (blocking acc, reads OLD comp_sr_v)
        comp_acc_v = sum(comp_sr_v[i] * H_COMP[i] for i in range(N_COMP))
        comp_acc_v_16 = to_signed_16(comp_acc_v)

        det1 = slicer(comp_acc_v_16, THRESH_LOW, THRESH_MID, THRESH_HIGH)

        old_comp_out_v = comp_out_v
        old_comp_out_a = comp_out_a

        comp_sr_v = [pam4_v] + comp_sr_v[:-1]
        comp_out_v = comp_acc_v_16

        # Aggressor FIR
        comp_acc_a = sum(comp_sr_a[i] * H_COMP[i] for i in range(N_COMP))
        comp_acc_a_16 = to_signed_16(comp_acc_a)

        comp_sr_a = [pam4_a] + comp_sr_a[:-1]
        comp_out_a = comp_acc_a_16

        # Coupling (reads old comp_out_v/a)
        coupling_term = asr(old_comp_out_a * COUPLING_NUM, COUPLING_SHIFT)
        coupled_val = old_comp_out_v + coupling_term
        if coupled_val > 32767:
            coupled_val = 32767
        elif coupled_val < -32768:
            coupled_val = -32768
        coupled_val = to_signed_16(coupled_val)

        det2 = slicer(coupled_val, COUPLED_THRESH_LOW, COUPLED_THRESH_MID,
                       COUPLED_THRESH_HIGH)

        old_coupled_out = coupled_out
        coupled_out = coupled_val

        # FFE (reads OLD coupled_out via ffe_sr)
        ffe_acc = sum(ffe_sr[i] * FFE_TAPS[i] for i in range(N_FFE))
        ffe_16 = to_signed_16(ffe_acc >> Q_FFE)
        det3 = slicer(ffe_16, FFE_THRESH_LOW, FFE_THRESH_MID, FFE_THRESH_HIGH)
        ffe_sr = [old_coupled_out] + ffe_sr[:-1]

        # ML (reads OLD coupled_out via ml_sr — same as FFE)
        det4 = ml_forward(ml_sr)
        ml_sr = [old_coupled_out] + ml_sr[:-1]

        ref_sr = [tx_sym_v] + ref_sr[:-1]

        tx_symbols.append(tx_sym_v)
        det_mode1_list.append(det1)
        det_mode2_list.append(det2)
        det_mode3_list.append(det3)
        det_mode4_list.append(det4)

        symbol_count += 1
        if symbol_count >= NUM_SYMBOLS + 300:
            break

    tx_symbols = np.array(tx_symbols)
    det_mode1 = np.array(det_mode1_list)
    det_mode2 = np.array(det_mode2_list)
    det_mode3 = np.array(det_mode3_list)
    det_mode4 = np.array(det_mode4_list)

    modes = [
        (1, "Composite FIR -> Slicer (no coupling)", det_mode1),
        (2, "Composite FIR + FEXT coupling -> Slicer", det_mode2),
        (3, "Composite FIR + coupling -> RX FFE -> Slicer", det_mode3),
        (4, "Composite FIR + coupling -> ML NN -> Argmax", det_mode4),
    ]

    delays = {}
    sers = {}

    for mode_num, label, det in modes:
        print(f"\n{'='*60}")
        print(f"MODE {mode_num} ({label})")
        print(f"{'='*60}")

        best_delay, best_ser = 0, 1.0
        max_scan = 20 if mode_num <= 3 else 25
        for delay in range(max_scan):
            ref = tx_symbols[WARMUP:len(tx_symbols) - delay]
            d = det[WARMUP + delay:]
            n = min(len(ref), len(d))
            if n > 1000:
                errors = np.sum(ref[:n] != d[:n])
                ser = errors / n
                if ser < best_ser:
                    best_ser = ser
                    best_delay = delay
                if delay < 12:
                    print(f"  delay={delay:2d}: SER={ser:.6f} ({errors}/{n})")

        print(f"\n  OPTIMAL: delay={best_delay}, SER={best_ser:.6f} "
              f"({best_ser*100:.4f}%)")
        delays[mode_num] = best_delay
        sers[mode_num] = best_ser

        print(f"\n  100K-window SER:")
        total = 0; errs = 0; wi = 0
        for i in range(WARMUP, len(tx_symbols)):
            di = i + best_delay
            if di >= len(det):
                break
            if tx_symbols[i] != det[di]:
                errs += 1
            total += 1
            if total == SER_WINDOW:
                ppm = errs * 10
                print(f"    Window {wi}: {errs}/{SER_WINDOW}, ser_ppm={ppm}, "
                      f"display: {mode_num}  {ppm//100000%10}{ppm//10000%10}."
                      f"{ppm//1000%10}{ppm//100%10}{ppm//10%10}{ppm%10}")
                total = 0; errs = 0; wi += 1

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    for m in [1, 2, 3, 4]:
        print(f"  SYMBOL_DELAY_{m} = {delays[m]:2d}  -> Mode {m} SER: "
              f"{sers[m]*100:.4f}%")
    print(f"{'='*60}")

    return delays, sers


if __name__ == "__main__":
    simulate()
