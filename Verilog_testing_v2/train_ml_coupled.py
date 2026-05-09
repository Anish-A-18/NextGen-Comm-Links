#!/usr/bin/env python3
"""
Train ML Neural Network Equalizer for the lane-coupled Verilog pipeline.

Architecture:  23 inputs (11+1+11 context) -> 4 ReLU -> 4 ReLU -> 4 softmax
Input signal:  coupled_out (same signal the FFE reads, registered from coupling block)

The ML network operates on the same coupled_out shift register as the FFE,
but uses a wider 23-tap window and nonlinear activation to capture patterns
that a linear FFE cannot.

Pipeline in Verilog:
  ml_sr[0:22] shift register fed from coupled_out (registered, NB)
  On each symbol_tick: ml_sr shifts, then NN reads OLD ml_sr values (blocking MAC)
  -> Layer 1: 23x4 MAC + bias + ReLU
  -> Layer 2: 4x4 MAC + bias + ReLU
  -> Layer 3: 4x4 MAC + bias (logits, no softmax needed for argmax)
  -> Argmax of 4 logits -> det_mode4

Fixed-point: weights/biases in Q8.8 (16-bit signed, 8 fractional bits)
             Intermediate accumulators in 32-bit
"""

import numpy as np

H_COMP = [288, 84, -30, -27, -24, -17, -12, -7, -4, -2, -3, -2]
N_COMP = len(H_COMP)
PAM4_MAP = {0: -3, 1: -1, 2: 1, 3: 3}
SYMBOL_DIV = 4
NUM_SYMBOLS = 200000

COUPLING_COEFF_NUM = 17
COUPLING_COEFF_DEN = 64

ML_CONTEXT = 11  # 11 pre + 1 cursor + 11 post = 23
ML_INPUT_SIZE = 2 * ML_CONTEXT + 1  # 23
ML_HIDDEN = 4
ML_OUTPUT = 4
Q_ML = 10  # fractional bits for weights/biases
INPUT_SHIFT = 6  # right-shift input by 6 bits (16-bit -> 10-bit, range ~ [-24, 21])
TRAIN_ON_RAW = True  # train directly on shifted integers, no normalization


def to_signed_16(val):
    val = val & 0xFFFF
    return val - 0x10000 if val >= 0x8000 else val


def simulate_pipeline():
    """Simulate Verilog pipeline, collect coupled_out shift register buffers."""
    print("=" * 70)
    print("ML TRAINING: Simulating Verilog pipeline")
    print(f"  Coupling = {COUPLING_COEFF_NUM}/{COUPLING_COEFF_DEN}")
    print("=" * 70)

    prbs_v = 0x7F; prbs_a = 0x55
    comp_sr_v = [0] * N_COMP; comp_sr_a = [0] * N_COMP
    comp_out_v = 0; comp_out_a = 0
    coupled_out_reg = 0

    # ML shift register: 23 entries, models ml_sr[0:22] in Verilog
    ml_sr = [0] * ML_INPUT_SIZE

    tx_symbols = []
    ml_buffers = []  # OLD ml_sr snapshots (what NN reads in Verilog)

    cshift = int(np.log2(COUPLING_COEFF_DEN))

    for clk in range((NUM_SYMBOLS + 300) * SYMBOL_DIV):
        if clk % SYMBOL_DIV != SYMBOL_DIV - 1:
            continue

        new_bit_v = ((prbs_v >> 6) ^ (prbs_v >> 5)) & 1
        prbs_v = ((prbs_v << 1) | new_bit_v) & 0x7F
        tx_sym_v = prbs_v & 0x3
        pam4_v = PAM4_MAP[tx_sym_v]

        new_bit_a = ((prbs_a >> 6) ^ (prbs_a >> 5)) & 1
        prbs_a = ((prbs_a << 1) | new_bit_a) & 0x7F
        pam4_a = PAM4_MAP[prbs_a & 0x3]

        comp_acc_v = sum(comp_sr_v[i] * H_COMP[i] for i in range(N_COMP))
        comp_acc_v_16 = to_signed_16(comp_acc_v)
        comp_acc_a = sum(comp_sr_a[i] * H_COMP[i] for i in range(N_COMP))
        comp_acc_a_16 = to_signed_16(comp_acc_a)

        old_v = comp_out_v; old_a = comp_out_a
        comp_sr_v = [pam4_v] + comp_sr_v[:-1]; comp_out_v = comp_acc_v_16
        comp_sr_a = [pam4_a] + comp_sr_a[:-1]; comp_out_a = comp_acc_a_16

        ct = (old_a * COUPLING_COEFF_NUM) >> cshift
        cv = old_v + ct
        if cv > 32767: cv = 32767
        elif cv < -32768: cv = -32768
        cv = to_signed_16(cv)

        # ML reads OLD ml_sr (before NB shift), same as FFE
        ml_buffers.append(list(ml_sr))

        # NB shift: ml_sr[0] gets OLD coupled_out_reg
        ml_sr = [coupled_out_reg] + ml_sr[:-1]
        coupled_out_reg = cv

        tx_symbols.append(tx_sym_v)
        if len(tx_symbols) >= NUM_SYMBOLS + 300:
            break

    return np.array(tx_symbols), np.array(ml_buffers, dtype=np.float64)


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    ex = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return ex / np.sum(ex, axis=-1, keepdims=True)


def forward(x, weights, biases):
    """Forward pass: 2 hidden ReLU layers + output (logits)."""
    h = x
    for i in range(len(weights) - 1):
        h = relu(h @ weights[i] + biases[i])
    logits = h @ weights[-1] + biases[-1]
    return logits


def train_network(ml_buffers, tx_symbols):
    """Train the 23->4->4->4 neural network on integer-shifted inputs."""
    print("\n" + "=" * 60)
    print("TRAINING ML NETWORK (23 -> 4 ReLU -> 4 ReLU -> 4 softmax)")
    print(f"  Input pre-shift: >>> {INPUT_SHIFT} bits")
    print("=" * 60)

    # Apply the same input shift that Verilog will do
    # In Verilog: ml_input[j] = ml_sr[j] >>> INPUT_SHIFT (arithmetic right shift)
    shifted = np.floor(ml_buffers / (2 ** INPUT_SHIFT)).astype(np.float64)

    # Find alignment delay (cursor is at index ML_CONTEXT=11 in the context)
    cursor_signal = shifted[:, ML_CONTEXT]
    pam4_levels = np.array([-3, -1, 1, 3])
    tx_pam4 = pam4_levels[tx_symbols]

    best_corr = 0; best_d = 0
    for d in range(0, 25):
        a = tx_pam4[200:len(tx_pam4) - d]
        b = cursor_signal[200 + d:]
        n = min(len(a), len(b))
        if n > 1000:
            c = np.corrcoef(a[:n], b[:n])[0, 1]
            if abs(c) > abs(best_corr):
                best_corr = c; best_d = d
    print(f"  Cursor alignment: delay={best_d}, corr={best_corr:.4f}")

    warmup = 200
    N = min(len(tx_symbols) - best_d, len(shifted)) - warmup
    X = shifted[warmup + best_d:warmup + best_d + N].copy()
    y = tx_symbols[warmup:warmup + N].copy()

    x_scale = np.max(np.abs(X)) + 1e-8
    print(f"  Input range after shift: [{X.min():.0f}, {X.max():.0f}], scale={x_scale:.1f}")

    # Train on raw shifted integers — the quantized weights will be exact
    X_norm = X.copy()  # no normalization
    x_scale = 1.0  # no folding needed

    y_onehot = np.zeros((N, ML_OUTPUT))
    y_onehot[np.arange(N), y] = 1.0

    n_train = int(N * 0.8)
    idx = np.random.RandomState(42).permutation(N)
    X_train = X_norm[idx[:n_train]]
    y_train = y_onehot[idx[:n_train]]
    y_train_labels = y[idx[:n_train]]
    X_val = X_norm[idx[n_train:]]
    y_val_labels = y[idx[n_train:]]

    layer_sizes = [ML_INPUT_SIZE, ML_HIDDEN, ML_HIDDEN, ML_OUTPUT]
    rng = np.random.RandomState(123)
    weights = []
    biases = []
    for i in range(len(layer_sizes) - 1):
        fan_in = layer_sizes[i]
        # Kaiming init scaled for integer input magnitudes
        w = rng.randn(fan_in, layer_sizes[i+1]) * np.sqrt(2.0 / fan_in)
        if i == 0:
            w *= 0.01  # scale down first layer for large integer inputs
        b = np.zeros(layer_sizes[i+1])
        weights.append(w)
        biases.append(b)

    lr = 0.0001  # smaller LR for integer-scale inputs
    batch_size = 128
    epochs = 500

    print(f"  Training on RAW shifted integers (no normalization)")
    print(f"  Training: {n_train} samples, {epochs} epochs, lr={lr}, batch={batch_size}")

    for epoch in range(epochs):
        shuf = np.random.RandomState(epoch).permutation(n_train)
        X_s = X_train[shuf]; y_s = y_train[shuf]

        for b_start in range(0, n_train, batch_size):
            b_end = min(b_start + batch_size, n_train)
            xb = X_s[b_start:b_end]
            yb = y_s[b_start:b_end]
            bs = b_end - b_start

            h0 = xb
            z1 = h0 @ weights[0] + biases[0]; h1 = relu(z1)
            z2 = h1 @ weights[1] + biases[1]; h2 = relu(z2)
            z3 = h2 @ weights[2] + biases[2]; prob = softmax(z3)

            dz3 = (prob - yb) / bs
            dw2 = h2.T @ dz3; db2 = np.sum(dz3, axis=0)
            dh2 = dz3 @ weights[2].T; dz2 = dh2 * (z2 > 0)
            dw1 = h1.T @ dz2; db1 = np.sum(dz2, axis=0)
            dh1 = dz2 @ weights[1].T; dz1 = dh1 * (z1 > 0)
            dw0 = h0.T @ dz1; db0 = np.sum(dz1, axis=0)

            weights[0] -= lr * dw0; biases[0] -= lr * db0
            weights[1] -= lr * dw1; biases[1] -= lr * db1
            weights[2] -= lr * dw2; biases[2] -= lr * db2

        if epoch % 50 == 0 or epoch == epochs - 1:
            logits_val = forward(X_val, weights, biases)
            pred_val = np.argmax(logits_val, axis=1)
            val_acc = np.mean(pred_val == y_val_labels)
            logits_train = forward(X_train, weights, biases)
            pred_train = np.argmax(logits_train, axis=1)
            train_acc = np.mean(pred_train == y_train_labels)
            print(f"    Epoch {epoch:3d}: train_acc={train_acc:.6f}  val_acc={val_acc:.6f}")

    logits_all = forward(X_norm, weights, biases)
    pred_all = np.argmax(logits_all, axis=1)
    float_acc = np.mean(pred_all == y)
    float_ser = 1.0 - float_acc
    print(f"\n  Float accuracy: {float_acc:.6f}, SER: {float_ser*100:.4f}%")

    return weights, biases, x_scale, best_d


def quantize_network(weights, biases, x_scale, ml_buffers, tx_symbols, delay):
    """Quantize to fixed-point and verify."""
    print("\n" + "=" * 60)
    print(f"QUANTIZATION: Q{16-Q_ML}.{Q_ML} (16-bit signed, {Q_ML} frac bits)")
    print(f"  Input shift: >>> {INPUT_SHIFT}, then scale by 1/{x_scale:.1f}")
    print("=" * 60)

    scale = 2 ** Q_ML

    # The float network was trained on X_norm = (shifted_input) / x_scale
    # In Verilog we can't divide by x_scale easily, so fold it into layer 1:
    # z1 = W0 * (input_shifted / x_scale) + b0
    #    = (W0 / x_scale) * input_shifted + b0
    w0_folded = weights[0] / x_scale  # (23, 4)
    b0_folded = biases[0]  # unchanged

    w0_q = np.round(w0_folded * scale).astype(np.int64)
    b0_q = np.round(b0_folded * scale).astype(np.int64)
    w1_q = np.round(weights[1] * scale).astype(np.int64)
    b1_q = np.round(biases[1] * scale).astype(np.int64)
    w2_q = np.round(weights[2] * scale).astype(np.int64)
    b2_q = np.round(biases[2] * scale).astype(np.int64)

    for arr in [w0_q, b0_q, w1_q, b1_q, w2_q, b2_q]:
        np.clip(arr, -32768, 32767, out=arr)

    print(f"  W0 range: [{w0_q.min()}, {w0_q.max()}]")
    print(f"  B0 range: [{b0_q.min()}, {b0_q.max()}]")
    print(f"  W1 range: [{w1_q.min()}, {w1_q.max()}]")
    print(f"  B1 range: [{b1_q.min()}, {b1_q.max()}]")
    print(f"  W2 range: [{w2_q.min()}, {w2_q.max()}]")
    print(f"  B2 range: [{b2_q.min()}, {b2_q.max()}]")

    # Bit-exact fixed-point inference matching Verilog
    # Input path: ml_sr values are 16-bit signed, shifted right by INPUT_SHIFT
    warmup = 200
    N = min(len(tx_symbols) - delay, len(ml_buffers)) - warmup
    det_ml = np.zeros(N, dtype=int)

    for i in range(N):
        raw_buf = ml_buffers[warmup + delay + i]  # 23 x 16-bit signed values

        # Arithmetic right shift (matching Verilog >>>)
        shifted_buf = np.array([int(v) >> INPUT_SHIFT for v in raw_buf], dtype=np.int64)

        # Layer 1: shifted_input * w0_q + b0_q, then >> Q_ML, ReLU
        h1 = np.zeros(ML_HIDDEN, dtype=np.int64)
        for k in range(ML_HIDDEN):
            acc = 0
            for j in range(ML_INPUT_SIZE):
                acc += int(shifted_buf[j]) * int(w0_q[j, k])
            acc += int(b0_q[k])
            val = acc >> Q_ML
            h1[k] = max(0, val)

        # Layer 2: h1 * w1_q + b1_q, >> Q_ML, ReLU
        h2 = np.zeros(ML_HIDDEN, dtype=np.int64)
        for k in range(ML_HIDDEN):
            acc = 0
            for j in range(ML_HIDDEN):
                acc += int(h1[j]) * int(w1_q[j, k])
            acc += int(b1_q[k])
            val = acc >> Q_ML
            h2[k] = max(0, val)

        # Layer 3: h2 * w2_q + b2_q, >> Q_ML (logits)
        logits = np.zeros(ML_OUTPUT, dtype=np.int64)
        for k in range(ML_OUTPUT):
            acc = 0
            for j in range(ML_HIDDEN):
                acc += int(h2[j]) * int(w2_q[j, k])
            acc += int(b2_q[k])
            logits[k] = acc >> Q_ML

        det_ml[i] = np.argmax(logits)

    ref = tx_symbols[warmup:warmup + N]
    errors = np.sum(det_ml != ref)
    ser = errors / N
    print(f"\n  Quantized SER: {ser*100:.4f}% ({errors}/{N})")

    for w in range(N // 100000):
        s = w * 100000; e = s + 100000
        we = np.sum(det_ml[s:e] != ref[s:e])
        print(f"    Window {w}: {we}/100000 = {we/1000:.4f}%")

    return w0_q, b0_q, w1_q, b1_q, w2_q, b2_q, delay, ser


def export_verilog(w0_q, b0_q, w1_q, b1_q, w2_q, b2_q, delay, ser):
    """Print Verilog localparam declarations."""
    print(f"\n{'='*70}")
    print("VERILOG PARAMETERS FOR ML EQUALIZER (Mode 4)")
    print(f"{'='*70}")
    print(f"    // ML Equalizer: 23 -> 4 ReLU -> 4 ReLU -> 4 argmax")
    print(f"    // Weights in Q{16-Q_ML}.{Q_ML} (16-bit signed, {Q_ML} fractional bits)")
    print(f"    // Normalization folded into Layer 1 weights/biases")
    print(f"    localparam ML_CONTEXT    = {ML_CONTEXT};")
    print(f"    localparam ML_INPUT      = {ML_INPUT_SIZE};")
    print(f"    localparam ML_HIDDEN     = {ML_HIDDEN};")
    print(f"    localparam ML_OUTPUT     = {ML_OUTPUT};")
    print(f"    localparam Q_ML          = {Q_ML};")
    print(f"    localparam ML_INPUT_SHIFT = {INPUT_SHIFT};")
    print(f"    localparam SYMBOL_DELAY_4 = {delay};")
    print()

    # Layer 1 weights: w0_q is (23, 4)
    print(f"    // Layer 1 weights (23 inputs x 4 hidden), row-major")
    for j in range(ML_INPUT_SIZE):
        for k in range(ML_HIDDEN):
            v = int(w0_q[j, k])
            s = "-" if v < 0 else " "
            print(f"    localparam signed [15:0] ML_W1_{j:02d}_{k} = "
                  f"{'-' if v<0 else ''}16'sd{abs(v)};")
    print()

    print(f"    // Layer 1 biases (4)")
    for k in range(ML_HIDDEN):
        v = int(b0_q[k])
        print(f"    localparam signed [15:0] ML_B1_{k} = "
              f"{'-' if v<0 else ''}16'sd{abs(v)};")
    print()

    # Layer 2 weights: w1_q is (4, 4)
    print(f"    // Layer 2 weights (4 x 4)")
    for j in range(ML_HIDDEN):
        for k in range(ML_HIDDEN):
            v = int(w1_q[j, k])
            print(f"    localparam signed [15:0] ML_W2_{j}_{k} = "
                  f"{'-' if v<0 else ''}16'sd{abs(v)};")
    print()

    print(f"    // Layer 2 biases (4)")
    for k in range(ML_HIDDEN):
        v = int(b1_q[k])
        print(f"    localparam signed [15:0] ML_B2_{k} = "
              f"{'-' if v<0 else ''}16'sd{abs(v)};")
    print()

    # Output layer weights: w2_q is (4, 4)
    print(f"    // Output layer weights (4 x 4)")
    for j in range(ML_HIDDEN):
        for k in range(ML_OUTPUT):
            v = int(w2_q[j, k])
            print(f"    localparam signed [15:0] ML_W3_{j}_{k} = "
                  f"{'-' if v<0 else ''}16'sd{abs(v)};")
    print()

    print(f"    // Output layer biases (4)")
    for k in range(ML_OUTPUT):
        v = int(b2_q[k])
        print(f"    localparam signed [15:0] ML_B3_{k} = "
              f"{'-' if v<0 else ''}16'sd{abs(v)};")
    print()

    print(f"    // Expected Mode 4 SER: {ser*100:.4f}%")
    print(f"{'='*70}")


def main():
    tx_symbols, ml_buffers = simulate_pipeline()
    weights, biases, x_scale, delay = train_network(
        ml_buffers, tx_symbols)
    w0_q, b0_q, w1_q, b1_q, w2_q, b2_q, delay, ser = quantize_network(
        weights, biases, x_scale, ml_buffers, tx_symbols, delay)
    export_verilog(w0_q, b0_q, w1_q, b1_q, w2_q, b2_q, delay, ser)

    return {
        'w0_q': w0_q, 'b0_q': b0_q,
        'w1_q': w1_q, 'b1_q': b1_q,
        'w2_q': w2_q, 'b2_q': b2_q,
        'delay': delay, 'ser': ser,
    }


if __name__ == "__main__":
    main()
