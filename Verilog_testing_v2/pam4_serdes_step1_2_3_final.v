`timescale 1ns / 1ps
// ============================================================================
// PAM4 SerDes Demo - Step 1+2+3: Composite Channel + Coupling + RX FFE
// ============================================================================
// Target: Nexys A7 FPGA (Artix-7, 100 MHz clock)
//
// Three modes, cycling every 30 seconds:
//   Mode 1: Composite FIR (TX*Ch*CTLE) -> threshold slicer -> SER  (~9.45%)
//   Mode 2: Composite FIR + lane coupling (FEXT) -> slicer -> SER  (~19.7%)
//   Mode 3: Composite FIR + lane coupling -> RX FFE -> slicer -> SER (~0.79%)
//
// Lane coupling: A second PRBS-7 (seed 0x55) drives an aggressor FIR.
//   The aggressor output is scaled by COUPLING_NUM/2^COUPLING_SHIFT and
//   added to the victim composite output (FEXT model).
//
// DAC Outputs (DAC8568, 32-bit SPI, 3 channels):
//   Channel A: Interpolated output (FIR / coupled / FFE, scaled to 12-bit)
//   Channel B: Error waveform (0xFFF = error, 0x000 = correct)
//   Channel E: Threshold slicer output (4 discrete PAM4 levels)
//
// 7-Segment Display: "M  XX.XXXX" where M = mode number (1, 2, or 3)
//   SER computed over 100,000-symbol windows, displayed as XX.XXXX%.
// ============================================================================

module pam4_serdes_step1 (
    input  wire        clk,
    output reg  [15:0] led,
    output wire        dac_cs,
    output wire        dac_mosi,
    output wire        dac_sck,
    output reg  [6:0]  seg,
    output reg         dp,
    output reg  [7:0]  an
);

    // ========================================================================
    // PARAMETERS
    // ========================================================================

    // Composite FIR taps (TX_FFE * Channel * CTLE, 12 taps)
    localparam signed [15:0] H_COMP_0  =  16'sd288;
    localparam signed [15:0] H_COMP_1  =  16'sd84;
    localparam signed [15:0] H_COMP_2  = -16'sd30;
    localparam signed [15:0] H_COMP_3  = -16'sd27;
    localparam signed [15:0] H_COMP_4  = -16'sd24;
    localparam signed [15:0] H_COMP_5  = -16'sd17;
    localparam signed [15:0] H_COMP_6  = -16'sd12;
    localparam signed [15:0] H_COMP_7  = -16'sd7;
    localparam signed [15:0] H_COMP_8  = -16'sd4;
    localparam signed [15:0] H_COMP_9  = -16'sd2;
    localparam signed [15:0] H_COMP_10 = -16'sd3;
    localparam signed [15:0] H_COMP_11 = -16'sd2;
    localparam N_COMP = 12;

    // Mode 1 thresholds (victim FIR only, no coupling)
    localparam signed [15:0] THRESH_LOW  = -16'sd608;
    localparam signed [15:0] THRESH_MID  =  16'sd0;
    localparam signed [15:0] THRESH_HIGH =  16'sd608;
    localparam SYMBOL_DELAY_1 = 1;
    localparam signed [15:0] DAC_OFFSET_1 = -16'sd1592;

    // Lane coupling: FEXT = aggressor_comp_out * COUPLING_NUM >>> COUPLING_SHIFT
    localparam signed [15:0] COUPLING_NUM   = 16'sd17;
    localparam                COUPLING_SHIFT = 6;  // 17/64 = 0.2656

    // Mode 2 thresholds (coupled, no FFE)
    localparam signed [15:0] COUPLED_THRESH_LOW  = -16'sd681;
    localparam signed [15:0] COUPLED_THRESH_MID  =  16'sd0;
    localparam signed [15:0] COUPLED_THRESH_HIGH =  16'sd681;
    localparam SYMBOL_DELAY_2 = 2;
    localparam signed [15:0] DAC_OFFSET_2 = -16'sd1610;

    // RX FFE taps (8 taps, Q1.14, trained on coupled Verilog pipeline)
    localparam signed [15:0] FFE_PRE3  = -16'sd274;
    localparam signed [15:0] FFE_PRE2  = -16'sd145;
    localparam signed [15:0] FFE_PRE1  =  16'sd474;
    localparam signed [15:0] FFE_MAIN  =  16'sd16384;
    localparam signed [15:0] FFE_POST1 = -16'sd4674;
    localparam signed [15:0] FFE_POST2 =  16'sd3018;
    localparam signed [15:0] FFE_POST3 =  16'sd598;
    localparam signed [15:0] FFE_POST4 =  16'sd1794;
    localparam N_FFE     = 8;
    localparam N_FFE_PRE = 3;
    localparam Q_FFE     = 14;

    // Mode 3 thresholds (coupled + FFE)
    localparam signed [15:0] FFE_THRESH_LOW  = -16'sd600;
    localparam signed [15:0] FFE_THRESH_MID  =  16'sd0;
    localparam signed [15:0] FFE_THRESH_HIGH =  16'sd600;
    localparam SYMBOL_DELAY_3 = 7;
    localparam signed [15:0] DAC_OFFSET_3 = -16'sd1224;

    // Timing
    localparam SYMBOL_DIV = 4;
    localparam [31:0] SER_WINDOW = 32'd100000;
    localparam SPI_CLK_DIV = 4;

    // 30-second mode switch at 100 MHz
    localparam [31:0] MODE_SWITCH_TICKS = 32'd3_000_000_000;

    // DAC channel addresses
    localparam [3:0] DAC_ADDR_A = 4'b0000;
    localparam [3:0] DAC_ADDR_B = 4'b0001;
    localparam [3:0] DAC_ADDR_E = 4'b0100;

    // Slicer DAC levels
    localparam [11:0] SLICER_DAC_00 = 12'h000;
    localparam [11:0] SLICER_DAC_01 = 12'h555;
    localparam [11:0] SLICER_DAC_10 = 12'hAAA;
    localparam [11:0] SLICER_DAC_11 = 12'hFFF;

    // ========================================================================
    // MODE SWITCH TIMER (30 seconds per mode, cycles 0->1->2->0...)
    // ========================================================================
    reg [31:0] mode_timer = 0;
    reg [1:0]  active_mode = 0;  // 0 = Mode 1, 1 = Mode 2, 2 = Mode 3

    always @(posedge clk) begin
        if (mode_timer >= MODE_SWITCH_TICKS - 1) begin
            mode_timer <= 0;
            if (active_mode == 2'd2)
                active_mode <= 2'd0;
            else
                active_mode <= active_mode + 1;
        end else begin
            mode_timer <= mode_timer + 1;
        end
    end

    // ========================================================================
    // CLOCK / SYMBOL TICK
    // ========================================================================
    reg [3:0] tick_counter = 0;
    reg       symbol_tick  = 0;

    always @(posedge clk) begin
        if (tick_counter == SYMBOL_DIV - 1) begin
            tick_counter <= 0;
            symbol_tick  <= 1;
        end else begin
            tick_counter <= tick_counter + 1;
            symbol_tick  <= 0;
        end
    end

    // ========================================================================
    // VICTIM PRBS-7 GENERATOR (seed 0x7F)
    // ========================================================================
    reg [6:0] prbs7_v = 7'h7F;

    always @(posedge clk) begin
        if (symbol_tick) begin
            prbs7_v[6:1] <= prbs7_v[5:0];
            prbs7_v[0]   <= prbs7_v[6] ^ prbs7_v[5];
        end
    end

    wire [1:0] tx_symbol_v = prbs7_v[1:0];

    reg signed [2:0] pam4_level_v;
    always @(*) begin
        case (tx_symbol_v)
            2'b00: pam4_level_v = -3;
            2'b01: pam4_level_v = -1;
            2'b10: pam4_level_v =  1;
            2'b11: pam4_level_v =  3;
        endcase
    end

    // ========================================================================
    // AGGRESSOR PRBS-7 GENERATOR (seed 0x55)
    // ========================================================================
    reg [6:0] prbs7_a = 7'h55;

    always @(posedge clk) begin
        if (symbol_tick) begin
            prbs7_a[6:1] <= prbs7_a[5:0];
            prbs7_a[0]   <= prbs7_a[6] ^ prbs7_a[5];
        end
    end

    wire [1:0] tx_symbol_a = prbs7_a[1:0];

    reg signed [2:0] pam4_level_a;
    always @(*) begin
        case (tx_symbol_a)
            2'b00: pam4_level_a = -3;
            2'b01: pam4_level_a = -1;
            2'b10: pam4_level_a =  1;
            2'b11: pam4_level_a =  3;
        endcase
    end

    // ========================================================================
    // VICTIM COMPOSITE FIR + MODE 1 SLICER (all in ONE always block)
    // ========================================================================
    reg signed [2:0]  comp_sr_v [0:11];
    reg signed [31:0] comp_acc_v;
    reg signed [15:0] comp_out_v;
    reg [1:0]  det_mode1;
    reg [11:0] dac_a_mode1;
    reg [11:0] dac_e_mode1;
    reg signed [15:0] dac_temp1;
    integer ci;

    always @(posedge clk) begin
        if (symbol_tick) begin
            for (ci = 11; ci > 0; ci = ci - 1)
                comp_sr_v[ci] <= comp_sr_v[ci-1];
            comp_sr_v[0] <= pam4_level_v;

            comp_acc_v = $signed(comp_sr_v[0])  * H_COMP_0
                       + $signed(comp_sr_v[1])  * H_COMP_1
                       + $signed(comp_sr_v[2])  * H_COMP_2
                       + $signed(comp_sr_v[3])  * H_COMP_3
                       + $signed(comp_sr_v[4])  * H_COMP_4
                       + $signed(comp_sr_v[5])  * H_COMP_5
                       + $signed(comp_sr_v[6])  * H_COMP_6
                       + $signed(comp_sr_v[7])  * H_COMP_7
                       + $signed(comp_sr_v[8])  * H_COMP_8
                       + $signed(comp_sr_v[9])  * H_COMP_9
                       + $signed(comp_sr_v[10]) * H_COMP_10
                       + $signed(comp_sr_v[11]) * H_COMP_11;

            comp_out_v <= comp_acc_v[15:0];

            // Mode 1 slicer (blocking comp_acc_v, same block)
            if ($signed(comp_acc_v[15:0]) < THRESH_LOW) begin
                det_mode1   <= 2'b00;
                dac_e_mode1 <= SLICER_DAC_00;
            end else if ($signed(comp_acc_v[15:0]) < THRESH_MID) begin
                det_mode1   <= 2'b01;
                dac_e_mode1 <= SLICER_DAC_01;
            end else if ($signed(comp_acc_v[15:0]) < THRESH_HIGH) begin
                det_mode1   <= 2'b10;
                dac_e_mode1 <= SLICER_DAC_10;
            end else begin
                det_mode1   <= 2'b11;
                dac_e_mode1 <= SLICER_DAC_11;
            end

            dac_temp1 = $signed(comp_acc_v[15:0]) - DAC_OFFSET_1;
            if (dac_temp1 < 0)
                dac_a_mode1 <= 12'd0;
            else if (dac_temp1 > 4095)
                dac_a_mode1 <= 12'hFFF;
            else
                dac_a_mode1 <= dac_temp1[11:0];
        end
    end

    // ========================================================================
    // AGGRESSOR COMPOSITE FIR (runs in parallel, same structure)
    // ========================================================================
    reg signed [2:0]  comp_sr_a [0:11];
    reg signed [31:0] comp_acc_a;
    reg signed [15:0] comp_out_a;
    integer ai;

    always @(posedge clk) begin
        if (symbol_tick) begin
            for (ai = 11; ai > 0; ai = ai - 1)
                comp_sr_a[ai] <= comp_sr_a[ai-1];
            comp_sr_a[0] <= pam4_level_a;

            comp_acc_a = $signed(comp_sr_a[0])  * H_COMP_0
                       + $signed(comp_sr_a[1])  * H_COMP_1
                       + $signed(comp_sr_a[2])  * H_COMP_2
                       + $signed(comp_sr_a[3])  * H_COMP_3
                       + $signed(comp_sr_a[4])  * H_COMP_4
                       + $signed(comp_sr_a[5])  * H_COMP_5
                       + $signed(comp_sr_a[6])  * H_COMP_6
                       + $signed(comp_sr_a[7])  * H_COMP_7
                       + $signed(comp_sr_a[8])  * H_COMP_8
                       + $signed(comp_sr_a[9])  * H_COMP_9
                       + $signed(comp_sr_a[10]) * H_COMP_10
                       + $signed(comp_sr_a[11]) * H_COMP_11;

            comp_out_a <= comp_acc_a[15:0];
        end
    end

    // ========================================================================
    // LANE COUPLING + MODE 2 SLICER + FFE INPUT (all in ONE always block)
    //
    // comp_out_v and comp_out_a are registered (NB) from the FIR blocks.
    // We compute the coupled signal as blocking, then:
    //   - Mode 2 slicer reads the blocking coupled value (no extra delay)
    //   - coupled_out is registered (NB) for the FFE to read next cycle
    // ========================================================================
    reg signed [15:0] coupled_out;
    reg signed [31:0] coupling_product;
    reg signed [15:0] coupled_val;
    reg [1:0]  det_mode2;
    reg [11:0] dac_a_mode2;
    reg [11:0] dac_e_mode2;
    reg signed [15:0] dac_temp2;

    always @(posedge clk) begin
        if (symbol_tick) begin
            // Blocking coupled computation (arithmetic right shift for signed)
            coupling_product = $signed(comp_out_a) * COUPLING_NUM;
            coupled_val = comp_out_v + (coupling_product >>> COUPLING_SHIFT);

            // Register for FFE to read next cycle
            coupled_out <= coupled_val;

            // Mode 2 slicer (blocking coupled_val, same block, no extra delay)
            if ($signed(coupled_val) < COUPLED_THRESH_LOW) begin
                det_mode2   <= 2'b00;
                dac_e_mode2 <= SLICER_DAC_00;
            end else if ($signed(coupled_val) < COUPLED_THRESH_MID) begin
                det_mode2   <= 2'b01;
                dac_e_mode2 <= SLICER_DAC_01;
            end else if ($signed(coupled_val) < COUPLED_THRESH_HIGH) begin
                det_mode2   <= 2'b10;
                dac_e_mode2 <= SLICER_DAC_10;
            end else begin
                det_mode2   <= 2'b11;
                dac_e_mode2 <= SLICER_DAC_11;
            end

            dac_temp2 = $signed(coupled_val) - DAC_OFFSET_2;
            if (dac_temp2 < 0)
                dac_a_mode2 <= 12'd0;
            else if (dac_temp2 > 4095)
                dac_a_mode2 <= 12'hFFF;
            else
                dac_a_mode2 <= dac_temp2[11:0];
        end
    end

    // ========================================================================
    // RX FFE (8-tap FIR on coupled_out, Q1.14) + MODE 3 SLICER
    //
    // ffe_sr[0] <= coupled_out reads the REGISTERED (old) coupled_out.
    // FFE acc reads OLD ffe_sr (non-blocking). Slicer reads blocking ffe_acc.
    // ========================================================================
    reg signed [15:0] ffe_sr [0:7];
    reg signed [31:0] ffe_acc;
    reg signed [15:0] ffe_out;
    reg [1:0]  det_mode3;
    reg [11:0] dac_a_mode3;
    reg [11:0] dac_e_mode3;
    reg signed [15:0] dac_temp3;
    integer fi;

    always @(posedge clk) begin
        if (symbol_tick) begin
            for (fi = 7; fi > 0; fi = fi - 1)
                ffe_sr[fi] <= ffe_sr[fi-1];
            ffe_sr[0] <= coupled_out;

            ffe_acc = $signed(ffe_sr[0]) * FFE_PRE3
                    + $signed(ffe_sr[1]) * FFE_PRE2
                    + $signed(ffe_sr[2]) * FFE_PRE1
                    + $signed(ffe_sr[3]) * FFE_MAIN
                    + $signed(ffe_sr[4]) * FFE_POST1
                    + $signed(ffe_sr[5]) * FFE_POST2
                    + $signed(ffe_sr[6]) * FFE_POST3
                    + $signed(ffe_sr[7]) * FFE_POST4;

            ffe_out <= ffe_acc[29:14];

            if ($signed(ffe_acc[29:14]) < FFE_THRESH_LOW) begin
                det_mode3   <= 2'b00;
                dac_e_mode3 <= SLICER_DAC_00;
            end else if ($signed(ffe_acc[29:14]) < FFE_THRESH_MID) begin
                det_mode3   <= 2'b01;
                dac_e_mode3 <= SLICER_DAC_01;
            end else if ($signed(ffe_acc[29:14]) < FFE_THRESH_HIGH) begin
                det_mode3   <= 2'b10;
                dac_e_mode3 <= SLICER_DAC_10;
            end else begin
                det_mode3   <= 2'b11;
                dac_e_mode3 <= SLICER_DAC_11;
            end

            dac_temp3 = $signed(ffe_acc[29:14]) - DAC_OFFSET_3;
            if (dac_temp3 < 0)
                dac_a_mode3 <= 12'd0;
            else if (dac_temp3 > 4095)
                dac_a_mode3 <= 12'hFFF;
            else
                dac_a_mode3 <= dac_temp3[11:0];
        end
    end

    // ========================================================================
    // REFERENCE SYMBOL DELAY (separate for each mode)
    // ========================================================================
    reg [1:0] ref_sr [0:31];
    integer di;

    always @(posedge clk) begin
        if (symbol_tick) begin
            for (di = 31; di > 0; di = di - 1)
                ref_sr[di] <= ref_sr[di-1];
            ref_sr[0] <= tx_symbol_v;
        end
    end

    wire [1:0] ref_mode1 = ref_sr[SYMBOL_DELAY_1];
    wire [1:0] ref_mode2 = ref_sr[SYMBOL_DELAY_2];
    wire [1:0] ref_mode3 = ref_sr[SYMBOL_DELAY_3];

    // ========================================================================
    // ACTIVE MODE MUX (3-way)
    // ========================================================================
    reg [1:0]  active_det;
    reg [1:0]  active_ref;
    reg [11:0] active_dac_a;
    reg [11:0] active_dac_e;

    always @(*) begin
        case (active_mode)
            2'd0: begin
                active_det   = det_mode1;
                active_ref   = ref_mode1;
                active_dac_a = dac_a_mode1;
                active_dac_e = dac_e_mode1;
            end
            2'd1: begin
                active_det   = det_mode2;
                active_ref   = ref_mode2;
                active_dac_a = dac_a_mode2;
                active_dac_e = dac_e_mode2;
            end
            default: begin
                active_det   = det_mode3;
                active_ref   = ref_mode3;
                active_dac_a = dac_a_mode3;
                active_dac_e = dac_e_mode3;
            end
        endcase
    end

    wire symbol_error = (active_det != active_ref);

    reg [11:0] dac_ch_b;
    always @(posedge clk) begin
        if (symbol_tick)
            dac_ch_b <= symbol_error ? 12'hFFF : 12'h000;
    end

    // ========================================================================
    // SER COUNTER (100,000 symbol windows, resets on mode switch)
    // ========================================================================
    reg [31:0] total_symbols = 0;
    reg [31:0] error_symbols = 0;
    reg        warmup_done   = 0;
    reg [15:0] warmup_count  = 0;
    reg [31:0] ser_errors_latched = 0;
    reg [1:0]  prev_mode = 0;

    always @(posedge clk) begin
        if (symbol_tick) begin
            if (active_mode != prev_mode) begin
                prev_mode     <= active_mode;
                total_symbols <= 0;
                error_symbols <= 0;
                warmup_done   <= 0;
                warmup_count  <= 0;
            end else if (!warmup_done) begin
                warmup_count <= warmup_count + 1;
                if (warmup_count >= 16'd100)
                    warmup_done <= 1;
            end else begin
                total_symbols <= total_symbols + 1;
                if (active_det != active_ref)
                    error_symbols <= error_symbols + 1;

                if (total_symbols == SER_WINDOW - 1) begin
                    if (active_det != active_ref)
                        ser_errors_latched <= error_symbols + 1;
                    else
                        ser_errors_latched <= error_symbols;
                    total_symbols <= 0;
                    error_symbols <= 0;
                end
            end
        end
    end

    // ========================================================================
    // SER PERCENTAGE: XX.XXXX% (ser_ppm = errors * 10 for 100K window)
    // ========================================================================
    reg [31:0] ser_ppm = 0;

    always @(posedge clk) begin
        if (symbol_tick && warmup_done && total_symbols == 0)
            ser_ppm <= ser_errors_latched * 10;
    end

    wire [3:0] ser_d5 = (ser_ppm / 100000) % 10;
    wire [3:0] ser_d4 = (ser_ppm / 10000)  % 10;
    wire [3:0] ser_d3 = (ser_ppm / 1000)   % 10;
    wire [3:0] ser_d2 = (ser_ppm / 100)    % 10;
    wire [3:0] ser_d1 = (ser_ppm / 10)     % 10;
    wire [3:0] ser_d0 = ser_ppm            % 10;

    reg [3:0] mode_digit;
    always @(*) begin
        case (active_mode)
            2'd0:    mode_digit = 4'd1;
            2'd1:    mode_digit = 4'd2;
            default: mode_digit = 4'd3;
        endcase
    end

    // ========================================================================
    // DAC SPI OUTPUT (DAC8568-style, 32-bit, 3 channels)
    // ========================================================================
    reg [15:0] spi_counter = 0;
    wire spi_tick = (spi_counter == 0);

    always @(posedge clk) begin
        if (spi_counter >= SPI_CLK_DIV)
            spi_counter <= 0;
        else
            spi_counter <= spi_counter + 1;
    end

    localparam DAC_IDLE     = 0;
    localparam DAC_SEND_REF = 1;
    localparam DAC_WAIT_REF = 2;
    localparam DAC_SEND_A   = 3;
    localparam DAC_SEND_B   = 4;
    localparam DAC_SEND_E   = 5;
    localparam DAC_WAIT     = 6;

    reg [2:0]  dac_state     = DAC_IDLE;
    reg [6:0]  dac_bit_count = 0;
    reg [31:0] dac_shift_reg = 0;
    reg [23:0] dac_wait_count = 0;
    reg        dac_first_init = 1;

    reg        cs_internal   = 1;
    reg        sck_internal  = 0;
    reg        mosi_internal = 0;

    reg [11:0] dac_a_reg = 0;
    reg [11:0] dac_b_reg = 0;
    reg [11:0] dac_e_reg = 0;

    always @(posedge clk) begin
        if (symbol_tick) begin
            dac_a_reg <= active_dac_a;
            dac_b_reg <= dac_ch_b;
            dac_e_reg <= active_dac_e;
        end
    end

    wire [31:0] cmd_ref_enable = {4'b0000, 4'b1000, 23'b0, 1'b1};
    wire [31:0] cmd_write_a    = {4'b0000, 4'b0011, DAC_ADDR_A, dac_a_reg, 8'b0};
    wire [31:0] cmd_write_b    = {4'b0000, 4'b0011, DAC_ADDR_B, dac_b_reg, 8'b0};
    wire [31:0] cmd_write_e    = {4'b0000, 4'b0011, DAC_ADDR_E, dac_e_reg, 8'b0};

    assign dac_cs   = cs_internal;
    assign dac_sck  = sck_internal;
    assign dac_mosi = mosi_internal;

    always @(posedge clk) begin
        case (dac_state)
            DAC_IDLE: begin
                cs_internal  <= 1;
                sck_internal <= 0;
                dac_bit_count <= 0;
                dac_wait_count <= 0;
                if (dac_first_init) begin
                    dac_shift_reg <= cmd_ref_enable;
                    dac_first_init <= 0;
                    dac_state <= DAC_SEND_REF;
                end else begin
                    dac_shift_reg <= cmd_write_a;
                    dac_state <= DAC_SEND_A;
                end
            end

            DAC_SEND_REF: begin
                if (spi_tick) begin
                    if (dac_bit_count == 0) begin
                        cs_internal <= 0;
                        dac_bit_count <= dac_bit_count + 1;
                    end else if (dac_bit_count <= 64) begin
                        if (sck_internal == 0) begin
                            mosi_internal <= dac_shift_reg[31];
                            sck_internal <= 1;
                        end else begin
                            dac_shift_reg <= {dac_shift_reg[30:0], 1'b0};
                            sck_internal <= 0;
                            dac_bit_count <= dac_bit_count + 1;
                        end
                    end else begin
                        cs_internal <= 1;
                        sck_internal <= 0;
                        dac_bit_count <= 0;
                        dac_state <= DAC_WAIT_REF;
                    end
                end
            end

            DAC_WAIT_REF: begin
                dac_wait_count <= dac_wait_count + 1;
                if (dac_wait_count >= 24'd1000000)
                    dac_state <= DAC_IDLE;
            end

            DAC_SEND_A: begin
                if (spi_tick) begin
                    if (dac_bit_count == 0) begin
                        cs_internal <= 0;
                        dac_bit_count <= dac_bit_count + 1;
                    end else if (dac_bit_count <= 64) begin
                        if (sck_internal == 0) begin
                            mosi_internal <= dac_shift_reg[31];
                            sck_internal <= 1;
                        end else begin
                            dac_shift_reg <= {dac_shift_reg[30:0], 1'b0};
                            sck_internal <= 0;
                            dac_bit_count <= dac_bit_count + 1;
                        end
                    end else begin
                        cs_internal <= 1;
                        sck_internal <= 0;
                        dac_bit_count <= 0;
                        dac_shift_reg <= cmd_write_b;
                        dac_state <= DAC_SEND_B;
                    end
                end
            end

            DAC_SEND_B: begin
                if (spi_tick) begin
                    if (dac_bit_count == 0) begin
                        cs_internal <= 0;
                        dac_bit_count <= dac_bit_count + 1;
                    end else if (dac_bit_count <= 64) begin
                        if (sck_internal == 0) begin
                            mosi_internal <= dac_shift_reg[31];
                            sck_internal <= 1;
                        end else begin
                            dac_shift_reg <= {dac_shift_reg[30:0], 1'b0};
                            sck_internal <= 0;
                            dac_bit_count <= dac_bit_count + 1;
                        end
                    end else begin
                        cs_internal <= 1;
                        sck_internal <= 0;
                        dac_bit_count <= 0;
                        dac_shift_reg <= cmd_write_e;
                        dac_state <= DAC_SEND_E;
                    end
                end
            end

            DAC_SEND_E: begin
                if (spi_tick) begin
                    if (dac_bit_count == 0) begin
                        cs_internal <= 0;
                        dac_bit_count <= dac_bit_count + 1;
                    end else if (dac_bit_count <= 64) begin
                        if (sck_internal == 0) begin
                            mosi_internal <= dac_shift_reg[31];
                            sck_internal <= 1;
                        end else begin
                            dac_shift_reg <= {dac_shift_reg[30:0], 1'b0};
                            sck_internal <= 0;
                            dac_bit_count <= dac_bit_count + 1;
                        end
                    end else begin
                        cs_internal <= 1;
                        sck_internal <= 0;
                        dac_bit_count <= 0;
                        dac_state <= DAC_WAIT;
                    end
                end
            end

            DAC_WAIT: begin
                if (symbol_tick)
                    dac_state <= DAC_IDLE;
            end

            default: dac_state <= DAC_IDLE;
        endcase
    end

    // ========================================================================
    // 7-SEGMENT DISPLAY: "M  XX.XXXX"
    // ========================================================================
    reg [2:0]  mux_digit = 0;
    reg [18:0] refresh_counter = 0;

    always @(posedge clk) begin
        refresh_counter <= refresh_counter + 1;
        if (refresh_counter == 0)
            mux_digit <= mux_digit + 1;
    end

    function [6:0] seg_encode;
        input [3:0] d;
        case (d)
            4'd0: seg_encode = 7'b1000000;
            4'd1: seg_encode = 7'b1111001;
            4'd2: seg_encode = 7'b0100100;
            4'd3: seg_encode = 7'b0110000;
            4'd4: seg_encode = 7'b0011001;
            4'd5: seg_encode = 7'b0010010;
            4'd6: seg_encode = 7'b0000010;
            4'd7: seg_encode = 7'b1111000;
            4'd8: seg_encode = 7'b0000000;
            4'd9: seg_encode = 7'b0010000;
            default: seg_encode = 7'b1111111;
        endcase
    endfunction

    always @(*) begin
        an  = 8'hFF;
        seg = 7'b1111111;
        dp  = 1;

        case (mux_digit)
            3'd7: begin an = 8'b01111111; seg = seg_encode(mode_digit); dp = 1; end
            3'd6: begin an = 8'b10111111; seg = 7'b1111111;            dp = 1; end
            3'd5: begin an = 8'b11011111; seg = seg_encode(ser_d5);    dp = 1; end
            3'd4: begin an = 8'b11101111; seg = seg_encode(ser_d4);    dp = 0; end
            3'd3: begin an = 8'b11110111; seg = seg_encode(ser_d3);    dp = 1; end
            3'd2: begin an = 8'b11111011; seg = seg_encode(ser_d2);    dp = 1; end
            3'd1: begin an = 8'b11111101; seg = seg_encode(ser_d1);    dp = 1; end
            3'd0: begin an = 8'b11111110; seg = seg_encode(ser_d0);    dp = 1; end
        endcase
    end

    // ========================================================================
    // LED INDICATORS
    // ========================================================================
    reg [7:0] error_shift = 0;
    always @(posedge clk) begin
        if (symbol_tick && warmup_done)
            error_shift <= {error_shift[6:0], symbol_error};
    end

    always @(posedge clk) begin
        led[15:8] <= error_shift;
        led[7:6]  <= tx_symbol_v;
        led[5:4]  <= active_ref;
        led[3:2]  <= active_det;
        led[1:0]  <= active_mode;
    end

    // ========================================================================
    // INITIALIZATION
    // ========================================================================
    integer ii;
    initial begin
        for (ii = 0; ii < N_COMP; ii = ii + 1) comp_sr_v[ii] = 0;
        for (ii = 0; ii < N_COMP; ii = ii + 1) comp_sr_a[ii] = 0;
        for (ii = 0; ii < N_FFE; ii = ii + 1)  ffe_sr[ii] = 0;
        for (ii = 0; ii < 32; ii = ii + 1)     ref_sr[ii] = 0;
        comp_out_v = 0;
        comp_out_a = 0;
        coupled_out = 0;
        ffe_out = 0;
        det_mode1 = 0;
        det_mode2 = 0;
        det_mode3 = 0;
        dac_a_mode1 = 0;
        dac_a_mode2 = 0;
        dac_a_mode3 = 0;
        dac_e_mode1 = 0;
        dac_e_mode2 = 0;
        dac_e_mode3 = 0;
        dac_ch_b = 0;
    end

endmodule
