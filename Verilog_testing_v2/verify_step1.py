#!/usr/bin/env python3
"""
Bit-Exact Python Verification for pam4_serdes_step1.v (Step 1 + Step 2)
========================================================================

Models the Verilog pipeline cycle-by-cycle with integer arithmetic.
- Mode 1: Composite FIR -> threshold slicer -> SER
- Mode 2: Composite FIR -> RX FFE (8 tap, Q1.14) -> threshold slicer -> SER

Each registered output uses non-blocking semantics (takes effect next cycle).
"""

import numpy as np

SYMBOL_DIV = 4

H_COMP = [288, 84, -30, -27, -24, -17, -12, -7, -4, -2, -3, -2]
N_COMP = len(H_COMP)

# Step 1 thresholds
THRESH_LOW  = -608
THRESH_MID  = 0
THRESH_HIGH = 608

# RX FFE taps (Q1.14): PRE3, PRE2, PRE1, MAIN, POST1..POST4
FFE_TAPS = [-77, -16, -37, 16384, -4942, 3157, -312, 1482]
N_FFE = len(FFE_TAPS)
N_FFE_PRE = 3
Q_FFE = 14

# Step 2 thresholds
FFE_THRESH_LOW  = -576
FFE_THRESH_MID  = 0
FFE_THRESH_HIGH = 576

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
    prbs7 = 0x7F

    # Composite FIR shift register (non-blocking model)
    comp_sr = [0] * N_COMP
    comp_out = 0       # registered output (1 cycle delay)

    # FFE shift register (non-blocking model, holds comp_out values)
    ffe_sr = [0] * N_FFE
    ffe_out = 0        # registered output (1 cycle delay)

    # Reference shift register
    ref_sr = [0] * 32

    tx_symbols = []
    det_mode1_list = []
    det_mode2_list = []
    comp_values = []
    ffe_values = []
    ref_sr_history = []

    print("Running bit-exact simulation of pam4_serdes_step1.v (Mode 1 + Mode 2)")
    print(f"  N_COMP={N_COMP}, N_FFE={N_FFE}, Q_FFE={Q_FFE}")
    print(f"  Mode1 thresholds: [{THRESH_LOW},{THRESH_MID},{THRESH_HIGH}]")
    print(f"  Mode2 thresholds: [{FFE_THRESH_LOW},{FFE_THRESH_MID},{FFE_THRESH_HIGH}]")

    # Cycle-accurate simulation
    # On each symbol_tick, the Verilog does (in parallel across always blocks):
    #
    # Block 1 (Composite FIR):
    #   comp_sr[i] <= comp_sr[i-1]  (non-blocking)
    #   comp_sr[0] <= pam4_level    (non-blocking)
    #   comp_acc = blocking FIR on OLD comp_sr
    #   comp_out <= comp_acc[15:0]  (non-blocking)
    #
    # Block 2 (Mode 1 slicer): reads blocking comp_acc from Block 1
    #   det_mode1 <= slicer(comp_acc)   (non-blocking)
    #   NOTE: This is in a SEPARATE always block from Block 1.
    #   comp_acc is declared as reg and assigned with blocking (=) in Block 1.
    #   In simulation, the blocking assignment in Block 1 happens within that
    #   block's execution. Block 2 executes in arbitrary order, so it may see
    #   the old OR new comp_acc. In practice, synthesis maps comp_acc to
    #   combinational logic feeding the register, so both blocks "see" the
    #   same value. We model it as seeing the NEW comp_acc.
    #
    # Block 3 (FFE):
    #   ffe_sr[i] <= ffe_sr[i-1]    (non-blocking, uses old values)
    #   ffe_sr[0] <= comp_out       (non-blocking, uses OLD comp_out - from prev tick)
    #   ffe_acc = blocking FIR on OLD ffe_sr
    #   ffe_out <= ffe_acc>>14      (non-blocking)
    #
    # Block 4 (Mode 2 slicer): reads ffe_out (from PREVIOUS tick, non-blocking)
    #   Actually ffe_out is read combinationally. Since ffe_out was assigned
    #   non-blocking in Block 3, the slicer in Block 4 reads the OLD ffe_out.
    #   Wait - let me re-read the Verilog...
    #
    # Actually in the Verilog, Block 4 (Mode 2 slicer) reads $signed(ffe_out),
    # which is the registered value from the PREVIOUS tick. But the slicer
    # assignment is also non-blocking (det_mode2 <=), so det_mode2 gets the
    # OLD ffe_out's slicer result. This adds another pipeline stage.
    #
    # Let me trace the full pipeline for Mode 2:
    #   Tick T:   comp_sr updated, comp_acc computed from old comp_sr
    #   Tick T:   comp_out <= comp_acc[15:0] (available at T+1)
    #   Tick T+1: ffe_sr shifts, ffe_sr[0] <= comp_out_old (T's comp_out)
    #             ffe_acc computed from old ffe_sr (T-1's values)
    #             ffe_out <= ffe_acc>>14 (available at T+2)
    #   Tick T+2: det_mode2 reads old ffe_out (T+1's ffe_out)
    #             det_mode2 <= slicer(old ffe_out)
    #
    # Wait, but det_mode2 is assigned on symbol_tick in Block 4, reading ffe_out.
    # ffe_out was last updated on the previous symbol_tick via non-blocking.
    # So det_mode2 on tick T reads ffe_out from tick T-1.
    #
    # Similarly for Mode 1: det_mode1 reads comp_acc which is a blocking
    # variable computed in Block 1. Since Block 2 is a SEPARATE always block,
    # there's a race condition on comp_acc. In synthesis, both blocks share
    # the same combinational logic for comp_acc, so Block 2 sees the NEW value.
    # But to be safe and match hardware, let's model it both ways and check.

    # Actually, I realize the Mode 2 slicer in the Verilog reads ffe_out (not
    # ffe_acc). ffe_out is a registered (non-blocking) value. So on any tick,
    # the slicer reads the ffe_out from the PREVIOUS tick's computation.

    symbol_count = 0

    for clk in range((NUM_SYMBOLS + 300) * SYMBOL_DIV):
        symbol_tick = (clk % SYMBOL_DIV == SYMBOL_DIV - 1)

        if not symbol_tick:
            continue

        # PRBS-7 (models Verilog non-blocking: we compute new bits, apply immediately)
        new_bit = ((prbs7 >> 6) ^ (prbs7 >> 5)) & 1
        prbs7 = ((prbs7 << 1) | new_bit) & 0x7F
        tx_sym = prbs7 & 0x3
        pam4 = PAM4_MAP[tx_sym]

        # --- Block 1: Composite FIR ---
        # Blocking comp_acc uses OLD comp_sr (before non-blocking update)
        comp_acc = sum(comp_sr[i] * H_COMP[i] for i in range(N_COMP))
        comp_acc_16 = to_signed_16(comp_acc)

        # Mode 1 slicer: uses comp_acc (blocking, same cycle)
        det1 = slicer(comp_acc_16, THRESH_LOW, THRESH_MID, THRESH_HIGH)

        # Non-blocking: shift register update + comp_out update
        old_comp_out = comp_out
        comp_sr = [pam4] + comp_sr[:-1]
        comp_out = comp_acc_16

        # --- FFE block (includes Mode 2 slicer in same always block) ---
        # Blocking ffe_acc uses OLD ffe_sr (before non-blocking update)
        ffe_acc = sum(ffe_sr[i] * FFE_TAPS[i] for i in range(N_FFE))
        ffe_acc_shifted = ffe_acc >> Q_FFE
        ffe_16 = to_signed_16(ffe_acc_shifted)

        # Mode 2 slicer: uses blocking ffe_acc (same cycle, same block)
        det2 = slicer(ffe_16, FFE_THRESH_LOW, FFE_THRESH_MID, FFE_THRESH_HIGH)

        # Non-blocking: ffe shift register + ffe_out update
        # ffe_sr[0] gets OLD comp_out (the non-blocking value from previous tick)
        ffe_sr = [old_comp_out] + ffe_sr[:-1]
        ffe_out = ffe_16

        # Reference shift register (non-blocking)
        ref_sr = [tx_sym] + ref_sr[:-1]

        tx_symbols.append(tx_sym)
        det_mode1_list.append(det1)
        det_mode2_list.append(det2)
        comp_values.append(comp_acc)
        ffe_values.append(ffe_acc_shifted)

        symbol_count += 1
        if symbol_count >= NUM_SYMBOLS + 300:
            break

    tx_symbols = np.array(tx_symbols)
    det_mode1 = np.array(det_mode1_list)
    det_mode2 = np.array(det_mode2_list)
    comp_values = np.array(comp_values)
    ffe_values = np.array(ffe_values)

    # ================================================================
    # Find optimal delays for both modes
    # ================================================================
    print("\n" + "=" * 60)
    print("MODE 1 (Composite FIR -> Slicer)")
    print("=" * 60)
    best_delay1 = 0
    best_ser1 = 1.0
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

    print(f"\n  OPTIMAL: delay={best_delay1}, SER={best_ser1:.6f} ({best_ser1*100:.4f}%)")

    # Mode 1 window SER
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
                  f"display: 1  {ppm//100000%10}{ppm//10000%10}.{ppm//1000%10}{ppm//100%10}{ppm//10%10}{ppm%10}")
            total = 0; errs = 0; wi += 1

    # ================================================================
    print("\n" + "=" * 60)
    print("MODE 2 (Composite FIR -> RX FFE -> Slicer)")
    print("=" * 60)
    best_delay2 = 0
    best_ser2 = 1.0
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
            if delay < 12:
                print(f"  delay={delay:2d}: SER={ser:.6f} ({errors}/{n})")

    print(f"\n  OPTIMAL: delay={best_delay2}, SER={best_ser2:.6f} ({best_ser2*100:.4f}%)")

    # Mode 2 window SER
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
                  f"display: 2  {ppm//100000%10}{ppm//10000%10}.{ppm//1000%10}{ppm//100%10}{ppm//10%10}{ppm%10}")
            total = 0; errs = 0; wi += 1

    # FFE output stats
    print(f"\n  FFE output stats:")
    for sym in range(4):
        d = best_delay2
        mask = tx_symbols[200:len(tx_symbols) - d] == sym
        vals = ffe_values[200 + d:len(ffe_values)]
        if len(mask) > 0 and len(vals) >= len(mask):
            sym_vals = vals[:len(mask)][mask]
            if len(sym_vals) > 0:
                print(f"    Symbol {sym}: mean={np.mean(sym_vals):.1f}, std={np.std(sym_vals):.1f}")

    # First 30 symbols for Mode 2
    print(f"\n  First 30 symbols (Mode 2, delay={best_delay2}):")
    for i in range(30):
        tx = tx_symbols[i]
        di = i + best_delay2
        d2 = det_mode2[di] if di < len(det_mode2) else -1
        fv = ffe_values[di] if di < len(ffe_values) else 0
        match = "OK" if tx == d2 else "ERR"
        print(f"    [{i:3d}] TX={tx} DET2={d2} FFE={fv:+6d} {match}")

    print(f"\n{'='*60}")
    print(f"VERILOG PARAMETERS TO SET")
    print(f"{'='*60}")
    print(f"  SYMBOL_DELAY_1 = {best_delay1}  (Mode 1)")
    print(f"  SYMBOL_DELAY_2 = {best_delay2}  (Mode 2)")
    print(f"  Mode 1 SER: {best_ser1*100:.4f}%")
    print(f"  Mode 2 SER: {best_ser2*100:.4f}%")
    print(f"{'='*60}")

    return best_delay1, best_ser1, best_delay2, best_ser2


if __name__ == "__main__":
    simulate()
