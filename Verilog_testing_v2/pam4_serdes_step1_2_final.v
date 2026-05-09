`timescale 1ns / 1ps
// ============================================================================
// PAM4 SerDes Demo - Step 1+2: Composite Channel + optional RX FFE
// ============================================================================
// Target: Nexys A7 FPGA (Artix-7, 100 MHz clock)
//
// Two modes, switching every 30 seconds:
//   Mode 1: Composite FIR (TX*Ch*CTLE) -> threshold slicer -> SER
//   Mode 2: Composite FIR -> RX FFE (8 tap) -> threshold slicer -> SER
//
// DAC Outputs (DAC8568, 32-bit SPI, 3 channels):
//   Channel A: Interpolated output (FIR or FFE, scaled to 12-bit)
//   Channel B: Error waveform (0xFFF = error, 0x000 = correct)
//   Channel E: Threshold slicer output (4 discrete PAM4 levels)
//
// 7-Segment Display: "M  XX.XXXX" where M = mode number (1 or 2)
//   SER computed over 100,000-symbol windows, displayed as XX.XXXX%.
// ============================================================================

module pam4_serdes_step1 (
    input  wire        clk,           // 100 MHz system clock
    output reg  [15:0] led,           // 16 LEDs
    output wire        dac_cs,        // DAC SPI chip select
    output wire        dac_mosi,      // DAC SPI data
    output wire        dac_sck,       // DAC SPI clock
    output reg  [6:0]  seg,           // 7-seg cathodes
    output reg         dp,            // 7-seg decimal point
    output reg  [7:0]  an             // 7-seg anodes
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

    // Step 1 thresholds
    localparam signed [15:0] THRESH_LOW  = -16'sd608;
    localparam signed [15:0] THRESH_MID  =  16'sd0;
    localparam signed [15:0] THRESH_HIGH =  16'sd608;
    localparam SYMBOL_DELAY_1 = 1;
    localparam signed [15:0] DAC_OFFSET_1 = -16'sd1592;  // from extract_fir_taps.py

    // RX FFE taps (8 taps, Q1.14, trained on exact Verilog pipeline signal)
    localparam signed [15:0] FFE_PRE3  = -16'sd77;
    localparam signed [15:0] FFE_PRE2  = -16'sd16;
    localparam signed [15:0] FFE_PRE1  = -16'sd37;
    localparam signed [15:0] FFE_MAIN  =  16'sd16384;
    localparam signed [15:0] FFE_POST1 = -16'sd4942;
    localparam signed [15:0] FFE_POST2 =  16'sd3157;
    localparam signed [15:0] FFE_POST3 = -16'sd312;
    localparam signed [15:0] FFE_POST4 =  16'sd1482;
    localparam N_FFE = 8;
    localparam N_FFE_PRE = 3;
    localparam Q_FFE = 14;

    // Step 2 thresholds
    localparam signed [15:0] FFE_THRESH_LOW  = -16'sd576;
    localparam signed [15:0] FFE_THRESH_MID  =  16'sd0;
    localparam signed [15:0] FFE_THRESH_HIGH =  16'sd576;
    localparam signed [15:0] DAC_OFFSET_2 = -16'sd1082;

    // Timing
    localparam SYMBOL_DIV = 4;
    localparam [31:0] SER_WINDOW = 32'd100000;
    localparam SPI_CLK_DIV = 4;

    // 30-second mode switch at 100 MHz
    localparam [31:0] MODE_SWITCH_TICKS = 32'd3_000_000_000;
    // (3 billion ticks = 30 seconds at 100 MHz)

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
    // MODE SWITCH TIMER (30 seconds per mode)
    // ========================================================================
    reg [31:0] mode_timer = 0;
    reg        active_mode = 0;  // 0 = Mode 1, 1 = Mode 2

    always @(posedge clk) begin
        if (mode_timer >= MODE_SWITCH_TICKS - 1) begin
            mode_timer  <= 0;
            active_mode <= ~active_mode;
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
    // PRBS-7 GENERATOR
    // ========================================================================
    reg [6:0] prbs7 = 7'h7F;

    always @(posedge clk) begin
        if (symbol_tick) begin
            prbs7[6:1] <= prbs7[5:0];
            prbs7[0]   <= prbs7[6] ^ prbs7[5];
        end
    end

    wire [1:0] tx_symbol = prbs7[1:0];

    reg signed [2:0] pam4_level;
    always @(*) begin
        case (tx_symbol)
            2'b00: pam4_level = -3;
            2'b01: pam4_level = -1;
            2'b10: pam4_level =  1;
            2'b11: pam4_level =  3;
        endcase
    end

    // ========================================================================
    // COMPOSITE FIR + MODE 1 SLICER (all in ONE always block)
    // ========================================================================
    reg signed [2:0]  comp_sr [0:11];
    reg signed [31:0] comp_acc;
    reg signed [15:0] comp_out;
    reg [1:0]  det_mode1;
    reg [11:0] dac_a_mode1;
    reg [11:0] dac_e_mode1;
    reg signed [15:0] dac_temp1;
    integer ci;

    always @(posedge clk) begin
        if (symbol_tick) begin
            // Shift register (non-blocking)
            for (ci = 11; ci > 0; ci = ci - 1)
                comp_sr[ci] <= comp_sr[ci-1];
            comp_sr[0] <= pam4_level;

            // FIR accumulate (blocking, reads OLD comp_sr)
            comp_acc = $signed(comp_sr[0])  * H_COMP_0
                     + $signed(comp_sr[1])  * H_COMP_1
                     + $signed(comp_sr[2])  * H_COMP_2
                     + $signed(comp_sr[3])  * H_COMP_3
                     + $signed(comp_sr[4])  * H_COMP_4
                     + $signed(comp_sr[5])  * H_COMP_5
                     + $signed(comp_sr[6])  * H_COMP_6
                     + $signed(comp_sr[7])  * H_COMP_7
                     + $signed(comp_sr[8])  * H_COMP_8
                     + $signed(comp_sr[9])  * H_COMP_9
                     + $signed(comp_sr[10]) * H_COMP_10
                     + $signed(comp_sr[11]) * H_COMP_11;

            comp_out <= comp_acc[15:0];

            // Mode 1 slicer (uses blocking comp_acc, same block)
            if ($signed(comp_acc[15:0]) < THRESH_LOW) begin
                det_mode1   <= 2'b00;
                dac_e_mode1 <= SLICER_DAC_00;
            end else if ($signed(comp_acc[15:0]) < THRESH_MID) begin
                det_mode1   <= 2'b01;
                dac_e_mode1 <= SLICER_DAC_01;
            end else if ($signed(comp_acc[15:0]) < THRESH_HIGH) begin
                det_mode1   <= 2'b10;
                dac_e_mode1 <= SLICER_DAC_10;
            end else begin
                det_mode1   <= 2'b11;
                dac_e_mode1 <= SLICER_DAC_11;
            end

            // DAC Channel A for Mode 1
            dac_temp1 = $signed(comp_acc[15:0]) - DAC_OFFSET_1;
            if (dac_temp1 < 0)
                dac_a_mode1 <= 12'd0;
            else if (dac_temp1 > 4095)
                dac_a_mode1 <= 12'hFFF;
            else
                dac_a_mode1 <= dac_temp1[11:0];
        end
    end

    // ========================================================================
    // RX FFE (8-tap FIR on comp_out, Q1.14)
    // ========================================================================
    // The FFE has 3 pre-cursor taps, so the "cursor" corresponds to
    // ffe_sr[3] (the sample 3 ticks ago). This means the FFE output
    // is naturally delayed by N_FFE_PRE = 3 symbol ticks relative to
    // the composite FIR input.
    //
    // ffe_sr[0] = newest comp_out (PRE3 = earliest / most future tap)
    // ffe_sr[3] = cursor sample (MAIN tap)
    // ffe_sr[7] = oldest (POST4)
    // ========================================================================
    reg signed [15:0] ffe_sr [0:7];
    reg signed [31:0] ffe_acc;
    reg signed [15:0] ffe_out;
    reg [1:0]  det_mode2;
    reg [11:0] dac_a_mode2;
    reg [11:0] dac_e_mode2;
    reg signed [15:0] dac_temp2;
    integer fi;

    always @(posedge clk) begin
        if (symbol_tick) begin
            // FFE shift register (non-blocking)
            for (fi = 7; fi > 0; fi = fi - 1)
                ffe_sr[fi] <= ffe_sr[fi-1];
            ffe_sr[0] <= comp_out;

            // FFE FIR (blocking, reads OLD ffe_sr values)
            ffe_acc = $signed(ffe_sr[0]) * FFE_PRE3
                    + $signed(ffe_sr[1]) * FFE_PRE2
                    + $signed(ffe_sr[2]) * FFE_PRE1
                    + $signed(ffe_sr[3]) * FFE_MAIN
                    + $signed(ffe_sr[4]) * FFE_POST1
                    + $signed(ffe_sr[5]) * FFE_POST2
                    + $signed(ffe_sr[6]) * FFE_POST3
                    + $signed(ffe_sr[7]) * FFE_POST4;

            ffe_out <= ffe_acc[29:14];

            // Mode 2 slicer (uses blocking ffe_acc, same cycle, same block)
            if ($signed(ffe_acc[29:14]) < FFE_THRESH_LOW) begin
                det_mode2   <= 2'b00;
                dac_e_mode2 <= SLICER_DAC_00;
            end else if ($signed(ffe_acc[29:14]) < FFE_THRESH_MID) begin
                det_mode2   <= 2'b01;
                dac_e_mode2 <= SLICER_DAC_01;
            end else if ($signed(ffe_acc[29:14]) < FFE_THRESH_HIGH) begin
                det_mode2   <= 2'b10;
                dac_e_mode2 <= SLICER_DAC_10;
            end else begin
                det_mode2   <= 2'b11;
                dac_e_mode2 <= SLICER_DAC_11;
            end

            // DAC Channel A for Mode 2
            dac_temp2 = $signed(ffe_acc[29:14]) - DAC_OFFSET_2;
            if (dac_temp2 < 0)
                dac_a_mode2 <= 12'd0;
            else if (dac_temp2 > 4095)
                dac_a_mode2 <= 12'hFFF;
            else
                dac_a_mode2 <= dac_temp2[11:0];
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
            ref_sr[0] <= tx_symbol;
        end
    end

    // Mode 1: composite FIR has 1-cycle shift register delay
    wire [1:0] ref_mode1 = ref_sr[SYMBOL_DELAY_1];

    // Mode 2: FFE adds N_FFE_PRE(=3) cursor delay + 1 pipeline stage
    // (comp_out registered) + 1 (ffe_sr non-blocking) + 1 (ffe_out registered)
    // Total from tx_symbol to det_mode2:
    //   1 (comp_sr non-blocking) + 1 (comp_out reg) + 1 (ffe_sr non-blocking)
    //   + N_FFE_PRE(=3) cursor offset + 1 (ffe_out reg) + 1 (det_mode2 reg)
    // We'll find the correct delay via simulation and set it here.
    // For now we parameterize it.
    localparam SYMBOL_DELAY_2 = 6;
    wire [1:0] ref_mode2 = ref_sr[SYMBOL_DELAY_2];

    // ========================================================================
    // ACTIVE MODE MUX
    // ========================================================================
    wire [1:0] active_det     = active_mode ? det_mode2     : det_mode1;
    wire [1:0] active_ref     = active_mode ? ref_mode2     : ref_mode1;
    wire [11:0] active_dac_a  = active_mode ? dac_a_mode2   : dac_a_mode1;
    wire [11:0] active_dac_e  = active_mode ? dac_e_mode2   : dac_e_mode1;

    wire symbol_error = (active_det != active_ref);

    // Channel B: error waveform
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
    reg        prev_mode = 0;

    always @(posedge clk) begin
        if (symbol_tick) begin
            // Reset counters on mode switch
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

    wire [3:0] mode_digit = active_mode ? 4'd2 : 4'd1;

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
        led[7:6]  <= tx_symbol;
        led[5:4]  <= active_ref;
        led[3:2]  <= active_det;
        led[1]    <= active_mode;
        led[0]    <= warmup_done;
    end

    // ========================================================================
    // INITIALIZATION
    // ========================================================================
    integer ii;
    initial begin
        for (ii = 0; ii < N_COMP; ii = ii + 1) comp_sr[ii] = 0;
        for (ii = 0; ii < N_FFE; ii = ii + 1)  ffe_sr[ii] = 0;
        for (ii = 0; ii < 32; ii = ii + 1)     ref_sr[ii] = 0;
        comp_out = 0;
        ffe_out = 0;
        det_mode1 = 0;
        det_mode2 = 0;
        dac_a_mode1 = 0;
        dac_a_mode2 = 0;
        dac_e_mode1 = 0;
        dac_e_mode2 = 0;
        dac_ch_b = 0;
    end

endmodule
