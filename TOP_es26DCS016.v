module CNN(
    input                   clk,
    input                   rst_n,
    input                   in_valid,
    input                   mode,
    input   signed  [7:0]   in_data_ch1,
    input   signed  [7:0]   in_data_ch2,
    input   signed  [7:0]   kernel_ch1,
    input   signed  [7:0]   kernel_ch2,
    input   signed  [7:0]   weight,
    output  reg             out_valid,
    output  reg signed [19:0]  out_data
);

//==================================================================
// parameters / state encoding
//==================================================================

localparam [3:0] ST_IDLE     = 4'd0;  
localparam [3:0] ST_CONV     = 4'd1;
localparam [3:0] ST_CONV_ACT = 4'd2;  // pipeline stage: final acc -> activation
localparam [3:0] ST_QUANT    = 4'd3;
localparam [3:0] ST_FC       = 4'd4;  // FC multiply stage
localparam [3:0] ST_OUT      = 4'd5;
localparam [3:0] ST_FC_WB    = 4'd6;
localparam [3:0] ST_CONV_WB  = 4'd7;  // pipeline stage: activation -> pool/writeback
localparam [3:0] ST_FC_PAIR  = 4'd8;  // FC product add stage
localparam [3:0] ST_FC_ACC   = 4'd9;  // FC accumulation stage
localparam [3:0] ST_CONV_POOL = 4'd10; // clk=6: split pooling max/writeback
localparam [3:0] ST_FC_FETCH  = 4'd11; // clk=6: fetch FC operands before multiply

integer ii;

//==================================================================
// Registers
//==================================================================

reg        [3:0]   st;

reg                in_valid_d;
reg                in_valid_p_d;
reg        [6:0]   load_cnt;
reg                img0_done;
reg                load_done;
reg                mode_r;
reg                pat_hold;

reg signed [7:0]    img_ch1 [0:71];
reg signed [7:0]    img_ch2 [0:71];
reg signed [7:0]    ker1    [0:8];
reg signed [7:0]    ker2    [0:8];
reg signed [7:0]    w_mem   [0:31];

reg                conv_img;
reg        [3:0]   conv_pos;
reg        [4:0]   conv_mac;
reg signed [31:0]  conv_acc0;
reg signed [31:0]  conv_acc1;
reg signed [31:0]  conv_acc2;
reg signed [31:0]  conv_acc3;

reg signed [19:0]  act0 [0:15];
reg signed [19:0]  act1 [0:15];

reg                pool_img;
reg        [1:0]   pool_pos;
reg signed [19:0]  pool0 [0:3];
reg signed [19:0]  pool1 [0:3];

reg signed [7:0]   feat [0:7];

reg        [1:0]   fc_out_idx;
reg        [2:0]   fc_in_idx;
reg signed [31:0]  fc_acc;
reg signed [19:0]  fc_y [0:3];

reg        [1:0]   out_cnt;

reg        [1:0]   conv_oy;
reg        [1:0]   conv_ox0;
reg        [1:0]   conv_ox1;
reg        [1:0]   conv_ox2;
reg        [1:0]   conv_ox3;
reg        [3:0]   conv_rem9;
reg        [1:0]   conv_ky;
reg        [1:0]   conv_kx;
reg        [6:0]   conv_img_base;
reg        [2:0]   conv_row6;
reg        [2:0]   conv_col6_0;
reg        [2:0]   conv_col6_1;
reg        [2:0]   conv_col6_2;
reg        [2:0]   conv_col6_3;
reg        [5:0]   conv_pix6_0;
reg        [5:0]   conv_pix6_1;
reg        [5:0]   conv_pix6_2;
reg        [5:0]   conv_pix6_3;
reg        [6:0]   conv_idx72_0;
reg        [6:0]   conv_idx72_1;
reg        [6:0]   conv_idx72_2;
reg        [6:0]   conv_idx72_3;
reg signed [7:0]   conv_img0_ch1;
reg signed [7:0]   conv_img0_ch2;
reg signed [7:0]   conv_img1_ch1;
reg signed [7:0]   conv_img1_ch2;
reg signed [7:0]   conv_img2_ch1;
reg signed [7:0]   conv_img2_ch2;
reg signed [7:0]   conv_img3_ch1;
reg signed [7:0]   conv_img3_ch2;
reg signed [7:0]   conv_ker_v1;
reg signed [7:0]   conv_ker_v2;
reg signed [7:0]   conv_op_img0_ch1_r;
reg signed [7:0]   conv_op_img0_ch2_r;
reg signed [7:0]   conv_op_img1_ch1_r;
reg signed [7:0]   conv_op_img1_ch2_r;
reg signed [7:0]   conv_op_img2_ch1_r;
reg signed [7:0]   conv_op_img2_ch2_r;
reg signed [7:0]   conv_op_img3_ch1_r;
reg signed [7:0]   conv_op_img3_ch2_r;
reg signed [7:0]   conv_op_ker1_r;
reg signed [7:0]   conv_op_ker2_r;
reg signed [15:0]  conv_prod0_ch1_r;
reg signed [15:0]  conv_prod0_ch2_r;
reg signed [15:0]  conv_prod1_ch1_r;
reg signed [15:0]  conv_prod1_ch2_r;
reg signed [15:0]  conv_prod2_ch1_r;
reg signed [15:0]  conv_prod2_ch2_r;
reg signed [15:0]  conv_prod3_ch1_r;
reg signed [15:0]  conv_prod3_ch2_r;
reg signed [16:0]  conv_prod_sum0;
reg signed [16:0]  conv_prod_sum1;
reg signed [16:0]  conv_prod_sum2;
reg signed [16:0]  conv_prod_sum3;
reg signed [31:0]  conv_prod_sum_ext0;
reg signed [31:0]  conv_prod_sum_ext1;
reg signed [31:0]  conv_prod_sum_ext2;
reg signed [31:0]  conv_prod_sum_ext3;
reg                conv_op_valid;
reg                conv_prod_valid;

reg        [3:0]   conv_act_idx0;
reg        [3:0]   conv_act_idx1;
reg        [3:0]   conv_act_idx2;
reg        [3:0]   conv_act_idx3;

reg        [1:0]   pool_py;
reg        [1:0]   pool_px;
reg        [3:0]   pool_p0;
reg        [3:0]   pool_p1;
reg        [3:0]   pool_p2;
reg        [3:0]   pool_p3;
reg signed [19:0]  pool_a00;
reg signed [19:0]  pool_a01;
reg signed [19:0]  pool_a10;
reg signed [19:0]  pool_a11;
reg signed [19:0]  pool_m_v;

reg signed [7:0]   fc_w_v;
reg signed [7:0]   fc_x_v;
reg signed [15:0]  fc_prod;
reg signed [31:0]  fc_sum_final;

reg signed [7:0]   fc_w_v0;
reg signed [7:0]   fc_w_v1;
reg signed [7:0]   fc_x_v0;
reg signed [7:0]   fc_x_v1;
reg signed [15:0]  fc_prod0;
reg signed [15:0]  fc_prod1;
reg signed [16:0]  fc_pair_sum;
reg signed [31:0]  fc_sum_final2;

reg signed [16:0]  fc_pair_sum_r;
// clk=8 timing optimization: split FC into mult -> pair-sum -> acc
reg signed [15:0]  fc_prod0_r;
reg signed [15:0]  fc_prod1_r;
// clk=6 timing optimization: register FC operands before multiplier
reg signed [7:0]   fc_w_v0_r;
reg signed [7:0]   fc_w_v1_r;
reg signed [7:0]   fc_x_v0_r;
reg signed [7:0]   fc_x_v1_r;
reg                fc_pair_valid;
reg                fc_pair_last;
reg                fc_drain;

reg signed [19:0]  conv_actv0;
reg signed [19:0]  conv_actv1;
reg signed [19:0]  conv_actv2;
reg signed [19:0]  conv_actv3;
reg signed [31:0]  conv_final_acc0;
reg signed [31:0]  conv_final_acc1;
reg signed [31:0]  conv_final_acc2;
reg signed [31:0]  conv_final_acc3;
reg signed [19:0]  conv_final_actv0;
reg signed [19:0]  conv_final_actv1;
reg signed [19:0]  conv_final_actv2;
reg signed [19:0]  conv_final_actv3;

// timing optimization: register final activation before pooling/writeback
reg signed [19:0]  conv_wb_actv0;
reg signed [19:0]  conv_wb_actv1;
reg signed [19:0]  conv_wb_actv2;
reg signed [19:0]  conv_wb_actv3;
reg                conv_wb_img;
reg        [1:0]   conv_wb_oy;
reg        [1:0]   conv_wb_pool_idx_base;

// timing optimization: register final accumulated values before activation
reg signed [31:0]  conv_pipe_acc0;
reg signed [31:0]  conv_pipe_acc1;
reg signed [31:0]  conv_pipe_acc2;
reg signed [31:0]  conv_pipe_acc3;
reg                conv_pipe_img;
reg        [1:0]   conv_pipe_oy;
reg        [1:0]   conv_pipe_pool_idx_base;

// clk=6 timing optimization: split 2x2 max-pooling into two cycles
reg                pool_pipe_img;
reg        [1:0]   pool_pipe_base;
reg signed [19:0]  pool_pipe_m0a;
reg signed [19:0]  pool_pipe_m0b;
reg signed [19:0]  pool_pipe_m1a;
reg signed [19:0]  pool_pipe_m1b;

reg signed [19:0]  prev0_r0;
reg signed [19:0]  prev0_r1;
reg signed [19:0]  prev0_r2;
reg signed [19:0]  prev0_r3;
reg signed [19:0]  prev1_r0;
reg signed [19:0]  prev1_r1;
reg signed [19:0]  prev1_r2;
reg signed [19:0]  prev1_r3;
reg        [1:0]   pool_idx_base;

// latency optimization: calculate quantized features and all FC outputs in ST_QUANT
reg signed [7:0]   q_feat0, q_feat1, q_feat2, q_feat3;
reg signed [7:0]   q_feat4, q_feat5, q_feat6, q_feat7;
reg signed [31:0]  fc_calc0, fc_calc1, fc_calc2, fc_calc3;

// parallel FC engine: four output neurons are accumulated at the same time
reg signed [31:0]  fc_acc0_all, fc_acc1_all, fc_acc2_all, fc_acc3_all;
reg signed [7:0]   fc_w00_r, fc_w01_r, fc_w10_r, fc_w11_r;
reg signed [7:0]   fc_w20_r, fc_w21_r, fc_w30_r, fc_w31_r;
reg signed [7:0]   fc_x0_all_r, fc_x1_all_r;
reg signed [15:0]  fc_p00_r, fc_p01_r, fc_p10_r, fc_p11_r;
reg signed [15:0]  fc_p20_r, fc_p21_r, fc_p30_r, fc_p31_r;
reg signed [16:0]  fc_pair0_all_r, fc_pair1_all_r, fc_pair2_all_r, fc_pair3_all_r;
reg signed [31:0]  fc_acc0_next, fc_acc1_next, fc_acc2_next, fc_acc3_next;

wire signed [16:0]  fc_pair0_now;
wire signed [16:0]  fc_pair1_now;
wire signed [16:0]  fc_pair2_now;
wire signed [16:0]  fc_pair3_now;
wire signed [31:0]  fc_acc0_now;
wire signed [31:0]  fc_acc1_now;
wire signed [31:0]  fc_acc2_now;
wire signed [31:0]  fc_acc3_now;


//==================================================================
// Wires
//==================================================================

function automatic signed [19:0] fn_abs20(input signed [31:0] v);
    reg signed [31:0] t;
begin
    if (v < 0) begin
        if (v == -32'sd524288) t = 32'sd524287;
        else t = -v;
    end else begin
        t = v;
    end
    fn_abs20 = t[19:0];
end
endfunction

function automatic signed [19:0] fn_relu20(input signed [31:0] v);
begin
    if (v < 0) fn_relu20 = 20'sd0;
    else fn_relu20 = v[19:0];
end
endfunction

function automatic signed [19:0] fn_max2_20(input signed [19:0] a, input signed [19:0] b);
begin
    if (a >= b) fn_max2_20 = a;
    else fn_max2_20 = b;
end
endfunction

function automatic signed [19:0] fn_max4_20(
    input signed [19:0] a,
    input signed [19:0] b,
    input signed [19:0] c,
    input signed [19:0] d
);
    reg signed [19:0] m1;
    reg signed [19:0] m2;
begin
    m1 = fn_max2_20(a, b);
    m2 = fn_max2_20(c, d);
    fn_max4_20 = fn_max2_20(m1, m2);
end
endfunction

function automatic signed [7:0] fn_quant8(input signed [19:0] v);
    reg signed [31:0] q;
begin
    q = $signed(v) >>> 9;
    if (q > 32'sd127) fn_quant8 = 8'sd127;
    else if (q < -32'sd128) fn_quant8 = -8'sd128;
    else fn_quant8 = q[7:0];
end
endfunction

function automatic signed [31:0] fn_mul8_sext(input signed [7:0] a, input signed [7:0] b);
    reg signed [15:0] p;
begin
    p = a * b;
    fn_mul8_sext = {{16{p[15]}}, p};
end
endfunction


//==================================================================
// Design
//==================================================================

assign fc_pair0_now = $signed(fc_p00_r) + $signed(fc_p01_r);
assign fc_pair1_now = $signed(fc_p10_r) + $signed(fc_p11_r);
assign fc_pair2_now = $signed(fc_p20_r) + $signed(fc_p21_r);
assign fc_pair3_now = $signed(fc_p30_r) + $signed(fc_p31_r);

assign fc_acc0_now = fc_acc0_all + {{15{fc_pair0_now[16]}}, fc_pair0_now};
assign fc_acc1_now = fc_acc1_all + {{15{fc_pair1_now[16]}}, fc_pair1_now};
assign fc_acc2_now = fc_acc2_all + {{15{fc_pair2_now[16]}}, fc_pair2_now};
assign fc_acc3_now = fc_acc3_all + {{15{fc_pair3_now[16]}}, fc_pair3_now};

always @* begin
    conv_oy = conv_pos[1:0];
    conv_ox0 = 2'd0;
    conv_ox1 = 2'd1;
    conv_ox2 = 2'd2;
    conv_ox3 = 2'd3;

    conv_rem9 = (conv_mac <= 5'd8) ? conv_mac[3:0] : 4'd0;

    if (conv_rem9 >= 4'd6) conv_ky = 2'd2;
    else if (conv_rem9 >= 4'd3) conv_ky = 2'd1;
    else conv_ky = 2'd0;

    conv_kx = conv_rem9 - (conv_ky * 2'd3);
    conv_row6 = conv_oy + conv_ky;
    conv_col6_0 = conv_ox0 + conv_kx;
    conv_col6_1 = conv_ox1 + conv_kx;
    conv_col6_2 = conv_ox2 + conv_kx;
    conv_col6_3 = conv_ox3 + conv_kx;
    conv_pix6_0 = (conv_row6 << 2) + (conv_row6 << 1) + conv_col6_0;
    conv_pix6_1 = (conv_row6 << 2) + (conv_row6 << 1) + conv_col6_1;
    conv_pix6_2 = (conv_row6 << 2) + (conv_row6 << 1) + conv_col6_2;
    conv_pix6_3 = (conv_row6 << 2) + (conv_row6 << 1) + conv_col6_3;

    conv_img_base = conv_img ? 7'd36 : 7'd0;
    conv_idx72_0 = conv_img_base + conv_pix6_0;
    conv_idx72_1 = conv_img_base + conv_pix6_1;
    conv_idx72_2 = conv_img_base + conv_pix6_2;
    conv_idx72_3 = conv_img_base + conv_pix6_3;

    conv_act_idx0 = {conv_oy, 2'd0};
    conv_act_idx1 = {conv_oy, 2'd1};
    conv_act_idx2 = {conv_oy, 2'd2};
    conv_act_idx3 = {conv_oy, 2'd3};

    conv_img0_ch1 = img_ch1[conv_idx72_0];
    conv_img0_ch2 = img_ch2[conv_idx72_0];
    conv_img1_ch1 = img_ch1[conv_idx72_1];
    conv_img1_ch2 = img_ch2[conv_idx72_1];
    conv_img2_ch1 = img_ch1[conv_idx72_2];
    conv_img2_ch2 = img_ch2[conv_idx72_2];
    conv_img3_ch1 = img_ch1[conv_idx72_3];
    conv_img3_ch2 = img_ch2[conv_idx72_3];
    conv_ker_v1 = ker1[conv_rem9];
    conv_ker_v2 = ker2[conv_rem9];

    conv_prod_sum0 = $signed(conv_prod0_ch1_r) + $signed(conv_prod0_ch2_r);
    conv_prod_sum1 = $signed(conv_prod1_ch1_r) + $signed(conv_prod1_ch2_r);
    conv_prod_sum2 = $signed(conv_prod2_ch1_r) + $signed(conv_prod2_ch2_r);
    conv_prod_sum3 = $signed(conv_prod3_ch1_r) + $signed(conv_prod3_ch2_r);
    conv_prod_sum_ext0 = {{15{conv_prod_sum0[16]}}, conv_prod_sum0};
    conv_prod_sum_ext1 = {{15{conv_prod_sum1[16]}}, conv_prod_sum1};
    conv_prod_sum_ext2 = {{15{conv_prod_sum2[16]}}, conv_prod_sum2};
    conv_prod_sum_ext3 = {{15{conv_prod_sum3[16]}}, conv_prod_sum3};

    pool_py = pool_pos[1];
    pool_px = pool_pos[0];
    pool_p0 = (pool_py << 3) + (pool_px << 1);
    pool_p1 = pool_p0 + 4'd1;
    pool_p2 = pool_p0 + 4'd4;
    pool_p3 = pool_p0 + 4'd5;

    if (!pool_img) begin
        pool_a00 = act0[pool_p0];
        pool_a01 = act0[pool_p1];
        pool_a10 = act0[pool_p2];
        pool_a11 = act0[pool_p3];
    end else begin
        pool_a00 = act1[pool_p0];
        pool_a01 = act1[pool_p1];
        pool_a10 = act1[pool_p2];
        pool_a11 = act1[pool_p3];
    end
    pool_m_v = fn_max4_20(pool_a00, pool_a01, pool_a10, pool_a11);

    fc_w_v = w_mem[{fc_out_idx, fc_in_idx}];
    fc_x_v = feat[fc_in_idx];
    fc_prod = $signed(fc_w_v) * $signed(fc_x_v);
    fc_sum_final = fc_acc + fc_prod;

    fc_w_v0 = w_mem[{fc_out_idx, fc_in_idx}];
    fc_w_v1 = w_mem[{fc_out_idx, (fc_in_idx + 3'd1)}];
    fc_x_v0 = feat[fc_in_idx];
    fc_x_v1 = feat[fc_in_idx + 3'd1];
    fc_prod0 = $signed(fc_w_v0) * $signed(fc_x_v0);
    fc_prod1 = $signed(fc_w_v1) * $signed(fc_x_v1);
    fc_pair_sum = $signed(fc_prod0) + $signed(fc_prod1);
    fc_sum_final2 = fc_acc + {{15{fc_pair_sum[16]}}, fc_pair_sum};

    if (mode_r) begin
        conv_actv0 = fn_abs20(conv_acc0);
        conv_actv1 = fn_abs20(conv_acc1);
        conv_actv2 = fn_abs20(conv_acc2);
        conv_actv3 = fn_abs20(conv_acc3);
    end else begin
        conv_actv0 = fn_relu20(conv_acc0);
        conv_actv1 = fn_relu20(conv_acc1);
        conv_actv2 = fn_relu20(conv_acc2);
        conv_actv3 = fn_relu20(conv_acc3);
    end

    conv_final_acc0 = conv_acc0 + conv_prod_sum_ext0;
    conv_final_acc1 = conv_acc1 + conv_prod_sum_ext1;
    conv_final_acc2 = conv_acc2 + conv_prod_sum_ext2;
    conv_final_acc3 = conv_acc3 + conv_prod_sum_ext3;

    if (mode_r) begin
        conv_final_actv0 = fn_abs20(conv_final_acc0);
        conv_final_actv1 = fn_abs20(conv_final_acc1);
        conv_final_actv2 = fn_abs20(conv_final_acc2);
        conv_final_actv3 = fn_abs20(conv_final_acc3);
    end else begin
        conv_final_actv0 = fn_relu20(conv_final_acc0);
        conv_final_actv1 = fn_relu20(conv_final_acc1);
        conv_final_actv2 = fn_relu20(conv_final_acc2);
        conv_final_actv3 = fn_relu20(conv_final_acc3);
    end

    pool_idx_base = {conv_oy[1], 1'b0};

    q_feat0 = fn_quant8(pool0[0]);
    q_feat1 = fn_quant8(pool0[1]);
    q_feat2 = fn_quant8(pool0[2]);
    q_feat3 = fn_quant8(pool0[3]);
    q_feat4 = fn_quant8(pool1[0]);
    q_feat5 = fn_quant8(pool1[1]);
    q_feat6 = fn_quant8(pool1[2]);
    q_feat7 = fn_quant8(pool1[3]);

    fc_calc0 = fn_mul8_sext(w_mem[0], q_feat0) +
               fn_mul8_sext(w_mem[1], q_feat1) +
               fn_mul8_sext(w_mem[2], q_feat2) +
               fn_mul8_sext(w_mem[3], q_feat3) +
               fn_mul8_sext(w_mem[4], q_feat4) +
               fn_mul8_sext(w_mem[5], q_feat5) +
               fn_mul8_sext(w_mem[6], q_feat6) +
               fn_mul8_sext(w_mem[7], q_feat7);
    fc_calc1 = fn_mul8_sext(w_mem[8], q_feat0) +
               fn_mul8_sext(w_mem[9], q_feat1) +
               fn_mul8_sext(w_mem[10], q_feat2) +
               fn_mul8_sext(w_mem[11], q_feat3) +
               fn_mul8_sext(w_mem[12], q_feat4) +
               fn_mul8_sext(w_mem[13], q_feat5) +
               fn_mul8_sext(w_mem[14], q_feat6) +
               fn_mul8_sext(w_mem[15], q_feat7);
    fc_calc2 = fn_mul8_sext(w_mem[16], q_feat0) +
               fn_mul8_sext(w_mem[17], q_feat1) +
               fn_mul8_sext(w_mem[18], q_feat2) +
               fn_mul8_sext(w_mem[19], q_feat3) +
               fn_mul8_sext(w_mem[20], q_feat4) +
               fn_mul8_sext(w_mem[21], q_feat5) +
               fn_mul8_sext(w_mem[22], q_feat6) +
               fn_mul8_sext(w_mem[23], q_feat7);
    fc_calc3 = fn_mul8_sext(w_mem[24], q_feat0) +
               fn_mul8_sext(w_mem[25], q_feat1) +
               fn_mul8_sext(w_mem[26], q_feat2) +
               fn_mul8_sext(w_mem[27], q_feat3) +
               fn_mul8_sext(w_mem[28], q_feat4) +
               fn_mul8_sext(w_mem[29], q_feat5) +
               fn_mul8_sext(w_mem[30], q_feat6) +
               fn_mul8_sext(w_mem[31], q_feat7);

    fc_acc0_next = fc_acc0_all + {{15{fc_pair0_all_r[16]}}, fc_pair0_all_r};
    fc_acc1_next = fc_acc1_all + {{15{fc_pair1_all_r[16]}}, fc_pair1_all_r};
    fc_acc2_next = fc_acc2_all + {{15{fc_pair2_all_r[16]}}, fc_pair2_all_r};
    fc_acc3_next = fc_acc3_all + {{15{fc_pair3_all_r[16]}}, fc_pair3_all_r};
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        in_valid_d <= 1'b0;
        load_cnt   <= 7'd0;
        img0_done  <= 1'b0;
        load_done  <= 1'b0;
        mode_r     <= 1'b0;
        for (ii = 0; ii < 72; ii = ii + 1) begin
            img_ch1[ii] <= 8'sd0;
            img_ch2[ii] <= 8'sd0;
        end
        for (ii = 0; ii < 9; ii = ii + 1) begin
            ker1[ii] <= 8'sd0;
            ker2[ii] <= 8'sd0;
        end
        for (ii = 0; ii < 32; ii = ii + 1) begin
            w_mem[ii] <= 8'sd0;
        end
    end else begin
        in_valid_d <= in_valid;

        if (!in_valid_d && in_valid) begin
            load_cnt  <= 7'd0;
            img0_done <= 1'b0;
            load_done <= 1'b0;
        end

        if (in_valid) begin
            if (!in_valid_d) begin
                mode_r <= mode;
                ker1[0] <= kernel_ch1;
                ker2[0] <= kernel_ch2;
                w_mem[0] <= weight;
                img_ch1[0] <= in_data_ch1;
                img_ch2[0] <= in_data_ch2;
                load_cnt <= 7'd1;
            end else begin
                if (load_cnt < 7'd9) begin
                    ker1[load_cnt] <= kernel_ch1;
                    ker2[load_cnt] <= kernel_ch2;
                end
                if (load_cnt < 7'd32) begin
                    w_mem[load_cnt] <= weight;
                end
                if (load_cnt < 7'd72) begin
                    img_ch1[load_cnt] <= in_data_ch1;
                    img_ch2[load_cnt] <= in_data_ch2;
                end
                if (load_cnt == 7'd35) begin
                    img0_done <= 1'b1;
                end
                if (load_cnt == 7'd71) begin
                    load_done <= 1'b1;
                end
                load_cnt <= load_cnt + 7'd1;
            end
        end
    end
end

always @(posedge clk or negedge rst_n) begin

    if (!rst_n) begin
        st <= ST_IDLE;
        out_valid <= 1'b0;
        out_data <= 20'sd0;
        in_valid_p_d <= 1'b0;
        pat_hold <= 1'b0;
        conv_img <= 1'b0;
        conv_pos <= 4'd0;
        conv_mac <= 5'd0;
        conv_acc0 <= 32'sd0;
        conv_acc1 <= 32'sd0;
        conv_acc2 <= 32'sd0;
        conv_acc3 <= 32'sd0;
        conv_op_img0_ch1_r <= 8'sd0;
        conv_op_img0_ch2_r <= 8'sd0;
        conv_op_img1_ch1_r <= 8'sd0;
        conv_op_img1_ch2_r <= 8'sd0;
        conv_op_img2_ch1_r <= 8'sd0;
        conv_op_img2_ch2_r <= 8'sd0;
        conv_op_img3_ch1_r <= 8'sd0;
        conv_op_img3_ch2_r <= 8'sd0;
        conv_op_ker1_r <= 8'sd0;
        conv_op_ker2_r <= 8'sd0;
        conv_prod0_ch1_r <= 16'sd0;
        conv_prod0_ch2_r <= 16'sd0;
        conv_prod1_ch1_r <= 16'sd0;
        conv_prod1_ch2_r <= 16'sd0;
        conv_prod2_ch1_r <= 16'sd0;
        conv_prod2_ch2_r <= 16'sd0;
        conv_prod3_ch1_r <= 16'sd0;
        conv_prod3_ch2_r <= 16'sd0;
        conv_op_valid <= 1'b0;
        conv_prod_valid <= 1'b0;
        pool_img <= 1'b0;
        pool_pos <= 2'd0;
        fc_out_idx <= 2'd0;
        fc_in_idx <= 3'd0;
        fc_acc <= 32'sd0;
        fc_pair_sum_r <= 17'sd0;
        fc_prod0_r <= 16'sd0;
        fc_prod1_r <= 16'sd0;
        fc_w_v0_r <= 8'sd0;
        fc_w_v1_r <= 8'sd0;
        fc_x_v0_r <= 8'sd0;
        fc_x_v1_r <= 8'sd0;
        fc_pair_valid <= 1'b0;
        fc_pair_last <= 1'b0;
        fc_drain <= 1'b0;
        fc_acc0_all <= 32'sd0;
        fc_acc1_all <= 32'sd0;
        fc_acc2_all <= 32'sd0;
        fc_acc3_all <= 32'sd0;
        fc_w00_r <= 8'sd0; fc_w01_r <= 8'sd0; fc_w10_r <= 8'sd0; fc_w11_r <= 8'sd0;
        fc_w20_r <= 8'sd0; fc_w21_r <= 8'sd0; fc_w30_r <= 8'sd0; fc_w31_r <= 8'sd0;
        fc_x0_all_r <= 8'sd0; fc_x1_all_r <= 8'sd0;
        fc_p00_r <= 16'sd0; fc_p01_r <= 16'sd0; fc_p10_r <= 16'sd0; fc_p11_r <= 16'sd0;
        fc_p20_r <= 16'sd0; fc_p21_r <= 16'sd0; fc_p30_r <= 16'sd0; fc_p31_r <= 16'sd0;
        fc_pair0_all_r <= 17'sd0; fc_pair1_all_r <= 17'sd0; fc_pair2_all_r <= 17'sd0; fc_pair3_all_r <= 17'sd0;
        out_cnt <= 2'd0;
        for (ii = 0; ii < 16; ii = ii + 1) begin
            act0[ii] <= 20'sd0;
            act1[ii] <= 20'sd0;
        end
        for (ii = 0; ii < 4; ii = ii + 1) begin
            pool0[ii] <= 20'sd0;
            pool1[ii] <= 20'sd0;
            fc_y[ii]  <= 20'sd0;
        end
        for (ii = 0; ii < 8; ii = ii + 1) begin
            feat[ii] <= 8'sd0;
        end

        prev0_r0 <= 20'sd0;
        prev0_r1 <= 20'sd0;
        prev0_r2 <= 20'sd0;
        prev0_r3 <= 20'sd0;
        prev1_r0 <= 20'sd0;
        prev1_r1 <= 20'sd0;
        prev1_r2 <= 20'sd0;
        prev1_r3 <= 20'sd0;
        conv_wb_actv0 <= 20'sd0;
        conv_wb_actv1 <= 20'sd0;
        conv_wb_actv2 <= 20'sd0;
        conv_wb_actv3 <= 20'sd0;
        conv_wb_img <= 1'b0;
        conv_wb_oy <= 2'd0;
        conv_wb_pool_idx_base <= 2'd0;
        conv_pipe_acc0 <= 32'sd0;
        conv_pipe_acc1 <= 32'sd0;
        conv_pipe_acc2 <= 32'sd0;
        conv_pipe_acc3 <= 32'sd0;
        conv_pipe_img <= 1'b0;
        conv_pipe_oy <= 2'd0;
        conv_pipe_pool_idx_base <= 2'd0;
        pool_pipe_img <= 1'b0;
        pool_pipe_base <= 2'd0;
        pool_pipe_m0a <= 20'sd0;
        pool_pipe_m0b <= 20'sd0;
        pool_pipe_m1a <= 20'sd0;
        pool_pipe_m1b <= 20'sd0;
    end else begin
        in_valid_p_d <= in_valid;
        if (!in_valid_p_d && in_valid) begin
            pat_hold <= 1'b0;
        end

        out_valid <= 1'b0;
        out_data  <= 20'sd0;

        case (st)
            ST_IDLE: begin
                if ((load_cnt >= 7'd18) && in_valid && !pat_hold) begin
                    conv_img <= 1'b0;
                    conv_pos <= 4'd0;
                    conv_mac <= 5'd0;
                    conv_acc0 <= 32'sd0;
                    conv_acc1 <= 32'sd0;
                    conv_acc2 <= 32'sd0;
                    conv_acc3 <= 32'sd0;
                    conv_op_valid <= 1'b0;
                    conv_prod_valid <= 1'b0;

                    prev0_r0 <= 20'sd0;
                    prev0_r1 <= 20'sd0;
                    prev0_r2 <= 20'sd0;
                    prev0_r3 <= 20'sd0;
                    prev1_r0 <= 20'sd0;
                    prev1_r1 <= 20'sd0;
                    prev1_r2 <= 20'sd0;
                    prev1_r3 <= 20'sd0;
                    pat_hold <= 1'b1;
                    st <= ST_CONV;
                end
            end

            ST_CONV: begin
                if (conv_mac <= 5'd8) begin
                    if (conv_prod_valid) begin
                        conv_acc0 <= conv_acc0 + conv_prod_sum_ext0;
                        conv_acc1 <= conv_acc1 + conv_prod_sum_ext1;
                        conv_acc2 <= conv_acc2 + conv_prod_sum_ext2;
                        conv_acc3 <= conv_acc3 + conv_prod_sum_ext3;
                    end

                    conv_prod0_ch1_r <= $signed(conv_op_img0_ch1_r) * $signed(conv_op_ker1_r);
                    conv_prod0_ch2_r <= $signed(conv_op_img0_ch2_r) * $signed(conv_op_ker2_r);
                    conv_prod1_ch1_r <= $signed(conv_op_img1_ch1_r) * $signed(conv_op_ker1_r);
                    conv_prod1_ch2_r <= $signed(conv_op_img1_ch2_r) * $signed(conv_op_ker2_r);
                    conv_prod2_ch1_r <= $signed(conv_op_img2_ch1_r) * $signed(conv_op_ker1_r);
                    conv_prod2_ch2_r <= $signed(conv_op_img2_ch2_r) * $signed(conv_op_ker2_r);
                    conv_prod3_ch1_r <= $signed(conv_op_img3_ch1_r) * $signed(conv_op_ker1_r);
                    conv_prod3_ch2_r <= $signed(conv_op_img3_ch2_r) * $signed(conv_op_ker2_r);
                    conv_prod_valid <= conv_op_valid;

                    conv_op_img0_ch1_r <= conv_img0_ch1;
                    conv_op_img0_ch2_r <= conv_img0_ch2;
                    conv_op_img1_ch1_r <= conv_img1_ch1;
                    conv_op_img1_ch2_r <= conv_img1_ch2;
                    conv_op_img2_ch1_r <= conv_img2_ch1;
                    conv_op_img2_ch2_r <= conv_img2_ch2;
                    conv_op_img3_ch1_r <= conv_img3_ch1;
                    conv_op_img3_ch2_r <= conv_img3_ch2;
                    conv_op_ker1_r <= conv_ker_v1;
                    conv_op_ker2_r <= conv_ker_v2;
                    conv_op_valid <= 1'b1;

                    conv_mac <= conv_mac + 5'd1;
                end else if (conv_mac == 5'd9) begin
                    if (conv_prod_valid) begin
                        conv_acc0 <= conv_acc0 + conv_prod_sum_ext0;
                        conv_acc1 <= conv_acc1 + conv_prod_sum_ext1;
                        conv_acc2 <= conv_acc2 + conv_prod_sum_ext2;
                        conv_acc3 <= conv_acc3 + conv_prod_sum_ext3;
                    end

                    conv_prod0_ch1_r <= $signed(conv_op_img0_ch1_r) * $signed(conv_op_ker1_r);
                    conv_prod0_ch2_r <= $signed(conv_op_img0_ch2_r) * $signed(conv_op_ker2_r);
                    conv_prod1_ch1_r <= $signed(conv_op_img1_ch1_r) * $signed(conv_op_ker1_r);
                    conv_prod1_ch2_r <= $signed(conv_op_img1_ch2_r) * $signed(conv_op_ker2_r);
                    conv_prod2_ch1_r <= $signed(conv_op_img2_ch1_r) * $signed(conv_op_ker1_r);
                    conv_prod2_ch2_r <= $signed(conv_op_img2_ch2_r) * $signed(conv_op_ker2_r);
                    conv_prod3_ch1_r <= $signed(conv_op_img3_ch1_r) * $signed(conv_op_ker1_r);
                    conv_prod3_ch2_r <= $signed(conv_op_img3_ch2_r) * $signed(conv_op_ker2_r);
                    conv_prod_valid <= conv_op_valid;

                    conv_op_valid <= 1'b0;
                    conv_mac <= 5'd10;
                end else if (conv_mac == 5'd10) begin
                    // clk=10 timing: first cut the long path here.
                    // Only register final accumulation result in this cycle.
                    conv_pipe_acc0 <= conv_final_acc0;
                    conv_pipe_acc1 <= conv_final_acc1;
                    conv_pipe_acc2 <= conv_final_acc2;
                    conv_pipe_acc3 <= conv_final_acc3;
                    conv_pipe_img  <= conv_img;
                    conv_pipe_oy   <= conv_oy;
                    conv_pipe_pool_idx_base <= pool_idx_base;

                    conv_acc0 <= 32'sd0;
                    conv_acc1 <= 32'sd0;
                    conv_acc2 <= 32'sd0;
                    conv_acc3 <= 32'sd0;
                    conv_op_valid <= 1'b0;
                    conv_prod_valid <= 1'b0;
                    conv_mac <= 5'd0;
                    st <= ST_CONV_ACT;
                end
            end

            ST_CONV_ACT: begin
                    // clk=10 timing: activation only, then register result.
                    if (mode_r) begin
                        conv_wb_actv0 <= fn_abs20(conv_pipe_acc0);
                        conv_wb_actv1 <= fn_abs20(conv_pipe_acc1);
                        conv_wb_actv2 <= fn_abs20(conv_pipe_acc2);
                        conv_wb_actv3 <= fn_abs20(conv_pipe_acc3);
                    end else begin
                        conv_wb_actv0 <= fn_relu20(conv_pipe_acc0);
                        conv_wb_actv1 <= fn_relu20(conv_pipe_acc1);
                        conv_wb_actv2 <= fn_relu20(conv_pipe_acc2);
                        conv_wb_actv3 <= fn_relu20(conv_pipe_acc3);
                    end
                    conv_wb_img <= conv_pipe_img;
                    conv_wb_oy  <= conv_pipe_oy;
                    conv_wb_pool_idx_base <= conv_pipe_pool_idx_base;
                    st <= ST_CONV_WB;
            end

            ST_CONV_WB: begin
                    // clk=6 timing: write activation / prev row only here.
                    // For odd output rows, first-stage max2 is registered, final max2 is in ST_CONV_POOL.
                    if (!conv_wb_img) begin
                        act0[{conv_wb_oy, 2'd0}] <= conv_wb_actv0;
                        act0[{conv_wb_oy, 2'd1}] <= conv_wb_actv1;
                        act0[{conv_wb_oy, 2'd2}] <= conv_wb_actv2;
                        act0[{conv_wb_oy, 2'd3}] <= conv_wb_actv3;
                    end else begin
                        act1[{conv_wb_oy, 2'd0}] <= conv_wb_actv0;
                        act1[{conv_wb_oy, 2'd1}] <= conv_wb_actv1;
                        act1[{conv_wb_oy, 2'd2}] <= conv_wb_actv2;
                        act1[{conv_wb_oy, 2'd3}] <= conv_wb_actv3;
                    end

                    if (!conv_wb_oy[0]) begin
                        if (!conv_wb_img) begin
                            prev0_r0 <= conv_wb_actv0;
                            prev0_r1 <= conv_wb_actv1;
                            prev0_r2 <= conv_wb_actv2;
                            prev0_r3 <= conv_wb_actv3;
                        end else begin
                            prev1_r0 <= conv_wb_actv0;
                            prev1_r1 <= conv_wb_actv1;
                            prev1_r2 <= conv_wb_actv2;
                            prev1_r3 <= conv_wb_actv3;
                        end

                        if (conv_pos == 4'd3) begin
                            if (!conv_img) begin
                                conv_img <= 1'b1;
                                conv_pos <= 4'd0;
                                st <= ST_CONV;
                            end else begin
                                st <= ST_QUANT;
                            end
                        end else begin
                            conv_pos <= conv_pos + 4'd1;
                            st <= ST_CONV;
                        end
                    end else begin
                        pool_pipe_img  <= conv_wb_img;
                        pool_pipe_base <= conv_wb_pool_idx_base;

                        if (!conv_wb_img) begin
                            pool_pipe_m0a <= fn_max2_20(prev0_r0, conv_wb_actv0);
                            pool_pipe_m0b <= fn_max2_20(prev0_r1, conv_wb_actv1);
                            pool_pipe_m1a <= fn_max2_20(prev0_r2, conv_wb_actv2);
                            pool_pipe_m1b <= fn_max2_20(prev0_r3, conv_wb_actv3);
                        end else begin
                            pool_pipe_m0a <= fn_max2_20(prev1_r0, conv_wb_actv0);
                            pool_pipe_m0b <= fn_max2_20(prev1_r1, conv_wb_actv1);
                            pool_pipe_m1a <= fn_max2_20(prev1_r2, conv_wb_actv2);
                            pool_pipe_m1b <= fn_max2_20(prev1_r3, conv_wb_actv3);
                        end
                        st <= ST_CONV_POOL;
                    end
            end

            ST_CONV_POOL: begin
                    // clk=6 timing: final max2 and pool register write only.
                    if (!pool_pipe_img) begin
                        pool0[pool_pipe_base]        <= fn_max2_20(pool_pipe_m0a, pool_pipe_m0b);
                        pool0[pool_pipe_base + 2'd1] <= fn_max2_20(pool_pipe_m1a, pool_pipe_m1b);
                    end else begin
                        pool1[pool_pipe_base]        <= fn_max2_20(pool_pipe_m0a, pool_pipe_m0b);
                        pool1[pool_pipe_base + 2'd1] <= fn_max2_20(pool_pipe_m1a, pool_pipe_m1b);
                    end

                    if (conv_pos == 4'd3) begin
                        if (!conv_img) begin
                            conv_img <= 1'b1;
                            conv_pos <= 4'd0;
                            st <= ST_CONV;
                        end else begin
                            st <= ST_QUANT;
                        end
                    end else begin
                        conv_pos <= conv_pos + 4'd1;
                        st <= ST_CONV;
                    end
            end

            ST_QUANT: begin
                // quantize and prepare the first FC pair in the same cycle
                feat[0] <= q_feat0;
                feat[1] <= q_feat1;
                feat[2] <= q_feat2;
                feat[3] <= q_feat3;
                feat[4] <= q_feat4;
                feat[5] <= q_feat5;
                feat[6] <= q_feat6;
                feat[7] <= q_feat7;

                fc_in_idx <= 3'd0;
                fc_acc0_all <= 32'sd0;
                fc_acc1_all <= 32'sd0;
                fc_acc2_all <= 32'sd0;
                fc_acc3_all <= 32'sd0;

                fc_w00_r <= w_mem[0];
                fc_w01_r <= w_mem[1];
                fc_w10_r <= w_mem[8];
                fc_w11_r <= w_mem[9];
                fc_w20_r <= w_mem[16];
                fc_w21_r <= w_mem[17];
                fc_w30_r <= w_mem[24];
                fc_w31_r <= w_mem[25];
                fc_x0_all_r <= q_feat0;
                fc_x1_all_r <= q_feat1;

                st <= ST_FC;
            end

            ST_FC_FETCH: begin
                // fetch two features for all four FC outputs in parallel
                fc_w00_r <= w_mem[{2'd0, fc_in_idx}];
                fc_w01_r <= w_mem[{2'd0, (fc_in_idx + 3'd1)}];
                fc_w10_r <= w_mem[{2'd1, fc_in_idx}];
                fc_w11_r <= w_mem[{2'd1, (fc_in_idx + 3'd1)}];
                fc_w20_r <= w_mem[{2'd2, fc_in_idx}];
                fc_w21_r <= w_mem[{2'd2, (fc_in_idx + 3'd1)}];
                fc_w30_r <= w_mem[{2'd3, fc_in_idx}];
                fc_w31_r <= w_mem[{2'd3, (fc_in_idx + 3'd1)}];
                fc_x0_all_r <= feat[fc_in_idx];
                fc_x1_all_r <= feat[fc_in_idx + 3'd1];
                st <= ST_FC;
            end

            ST_FC: begin
                // multiplier stage for all four FC outputs
                fc_p00_r <= $signed(fc_w00_r) * $signed(fc_x0_all_r);
                fc_p01_r <= $signed(fc_w01_r) * $signed(fc_x1_all_r);
                fc_p10_r <= $signed(fc_w10_r) * $signed(fc_x0_all_r);
                fc_p11_r <= $signed(fc_w11_r) * $signed(fc_x1_all_r);
                fc_p20_r <= $signed(fc_w20_r) * $signed(fc_x0_all_r);
                fc_p21_r <= $signed(fc_w21_r) * $signed(fc_x1_all_r);
                fc_p30_r <= $signed(fc_w30_r) * $signed(fc_x0_all_r);
                fc_p31_r <= $signed(fc_w31_r) * $signed(fc_x1_all_r);
                st <= ST_FC_PAIR;
            end

            ST_FC_PAIR: begin
                fc_acc0_all <= fc_acc0_now;
                fc_acc1_all <= fc_acc1_now;
                fc_acc2_all <= fc_acc2_now;
                fc_acc3_all <= fc_acc3_now;

                if (fc_in_idx == 3'd6) begin
                    fc_y[0] <= fc_acc0_now[19:0];
                    fc_y[1] <= fc_acc1_now[19:0];
                    fc_y[2] <= fc_acc2_now[19:0];
                    fc_y[3] <= fc_acc3_now[19:0];
                    out_valid <= 1'b1;
                    out_data  <= fc_acc0_now[19:0];
                    out_cnt <= 2'd1;
                    fc_in_idx <= 3'd0;
                    st <= ST_OUT;
                end else begin
                    fc_in_idx <= fc_in_idx + 3'd2;
                    st <= ST_FC_FETCH;
                end
            end

            ST_FC_ACC: begin
                st <= ST_FC_FETCH;
            end

            ST_FC_WB: begin
                fc_y[fc_out_idx] <= fc_acc[19:0];
                fc_acc <= 32'sd0;
                fc_in_idx <= 3'd0;
                fc_pair_sum_r <= 17'sd0;
                fc_prod0_r <= 16'sd0;
                fc_prod1_r <= 16'sd0;
                fc_w_v0_r <= 8'sd0;
                fc_w_v1_r <= 8'sd0;
                fc_x_v0_r <= 8'sd0;
                fc_x_v1_r <= 8'sd0;
                fc_pair_valid <= 1'b0;
                fc_pair_last <= 1'b0;
                fc_drain <= 1'b0;

                if (fc_out_idx == 2'd3) begin
                    out_cnt <= 2'd0;
                    st <= ST_OUT;
                end else begin
                    fc_out_idx <= fc_out_idx + 2'd1;
                    st <= ST_FC_FETCH;
                end
            end

            ST_OUT: begin
                out_valid <= 1'b1;
                out_data  <= fc_y[out_cnt];
                if (out_cnt == 2'd3) begin
                    out_cnt <= 2'd0;
                    st <= ST_IDLE;
                end else begin
                    out_cnt <= out_cnt + 2'd1;
                end
            end

            default: begin
                st <= ST_IDLE;
            end
        endcase
    end
end


endmodule

module TOP #(
    parameter ADDR_WIDTH      = 32,
    parameter DATA_WIDTH_inst = 16,
    parameter DATA_WIDTH_data = 8
)(
    input  wire clk,
    input  wire rst_n,
    output reg  IO_stall,

    output reg  [ADDR_WIDTH-1:0]      awaddr_m_inf_data,
    output reg                        awvalid_m_inf_data,
    input  wire                       awready_m_inf_data,
    output reg  [7:0]                 awlen_m_inf_data,

    output reg  [DATA_WIDTH_data-1:0] wdata_m_inf_data,
    output reg                        wvalid_m_inf_data,
    output reg                        wlast_m_inf_data,
    input  wire                       wready_m_inf_data,

    input  wire [1:0]                 bresp_m_inf_data,
    input  wire                       bvalid_m_inf_data,
    output reg                        bready_m_inf_data,

    output reg  [ADDR_WIDTH-1:0]      araddr_m_inf_data,
    output reg                        arvalid_m_inf_data,
    input  wire                       arready_m_inf_data,
    output reg  [7:0]                 arlen_m_inf_data,

    input  wire [DATA_WIDTH_data-1:0] rdata_m_inf_data,
    input  wire                       rvalid_m_inf_data,
    input  wire                       rlast_m_inf_data,
    output reg                        rready_m_inf_data,

    output reg  [ADDR_WIDTH-1:0]      araddr_m_inf_inst,
    output reg                        arvalid_m_inf_inst,
    input  wire                       arready_m_inf_inst,
    output reg  [7:0]                 arlen_m_inf_inst,

    input  wire [DATA_WIDTH_inst-1:0] rdata_m_inf_inst,
    input  wire                       rvalid_m_inf_inst,
    input  wire                       rlast_m_inf_inst,
    output reg                        rready_m_inf_inst
);

reg [4:0] state, next;

localparam S_IDLE        = 5'd0;
localparam S_IF_AR       = 5'd1;
localparam S_IF_R        = 5'd2;
localparam S_DECODE      = 5'd3;
localparam S_EXEC        = 5'd4;
localparam S_DATA_AR     = 5'd5;
localparam S_DATA_R      = 5'd6;
localparam S_DATA_AW     = 5'd7;
localparam S_DATA_W      = 5'd8;
localparam S_DATA_B      = 5'd9;
localparam S_CNN_LOAD_AR = 5'd10;
localparam S_CNN_LOAD_R  = 5'd11;
localparam S_CNN_IN      = 5'd12;
localparam S_CNN_OUT     = 5'd13;
localparam S_DONE        = 5'd14;

reg signed [7:0] reg_file [0:15];
reg signed [7:0] n_reg_file [0:15];

reg [31:0] pc, n_pc;
reg [15:0] inst_reg, n_inst_reg;

reg [15:0] inst_buf [0:7];
reg [15:0] n_inst_buf [0:7];
reg [2:0]  inst_buf_cnt, n_inst_buf_cnt;
reg [2:0]  inst_exec_cnt, n_inst_exec_cnt;

wire [2:0] opcode;
wire [3:0] rs;
wire [3:0] r_src2;
wire [3:0] r_dst;
wire [3:0] rt_i;
wire signed [4:0] imm;
wire [12:0] j_addr;

assign opcode = inst_reg[15:13];
assign rs     = inst_reg[12:9];
assign r_src2 = inst_reg[8:5];
assign r_dst  = inst_reg[4:1];
assign rt_i   = inst_reg[8:5];
assign imm    = inst_reg[4:0];
assign j_addr = inst_reg[12:0];

wire [2:0] imgA;
wire [2:0] imgB;
wire       k_sel;
wire       w_sel;

assign imgA  = inst_reg[12:10];
assign imgB  = inst_reg[9:7];
assign k_sel = inst_reg[2];
assign w_sel = inst_reg[1];

wire signed [31:0] data_offset;
assign data_offset = ($signed(reg_file[rs]) * $signed(imm)) + $signed(imm);

reg [ADDR_WIDTH-1:0]      n_awaddr_m_inf_data;
reg                       n_awvalid_m_inf_data;
reg [7:0]                 n_awlen_m_inf_data;
reg [DATA_WIDTH_data-1:0] n_wdata_m_inf_data;
reg                       n_wvalid_m_inf_data;
reg                       n_wlast_m_inf_data;
reg                       n_bready_m_inf_data;

reg [ADDR_WIDTH-1:0]      n_araddr_m_inf_data;
reg                       n_arvalid_m_inf_data;
reg [7:0]                 n_arlen_m_inf_data;
reg                       n_rready_m_inf_data;

reg [ADDR_WIDTH-1:0]      n_araddr_m_inf_inst;
reg                       n_arvalid_m_inf_inst;
reg [7:0]                 n_arlen_m_inf_inst;
reg                       n_rready_m_inf_inst;

reg                       n_IO_stall;

reg cnn_in_valid, n_cnn_in_valid;
reg cnn_mode, n_cnn_mode;
reg signed [7:0] cnn_in_data_ch1, n_cnn_in_data_ch1;
reg signed [7:0] cnn_in_data_ch2, n_cnn_in_data_ch2;
reg signed [7:0] cnn_kernel_ch1, n_cnn_kernel_ch1;
reg signed [7:0] cnn_kernel_ch2, n_cnn_kernel_ch2;
reg signed [7:0] cnn_weight, n_cnn_weight;

wire cnn_out_valid;
wire signed [19:0] cnn_out_data;

reg signed [19:0] cnn_out_data_form [0:3];
reg signed [19:0] n_cnn_out_data_form [0:3];
reg [1:0] cnn_cnt, n_cnn_cnt;

reg signed [7:0] imgA_ch1_buf [0:35];
reg signed [7:0] imgA_ch2_buf [0:35];
reg signed [7:0] imgB_ch1_buf [0:35];
reg signed [7:0] imgB_ch2_buf [0:35];
reg signed [7:0] ker_ch1_buf  [0:8];
reg signed [7:0] ker_ch2_buf  [0:8];
reg signed [7:0] weight_buf   [0:31];

reg signed [7:0] n_imgA_ch1_buf [0:35];
reg signed [7:0] n_imgA_ch2_buf [0:35];
reg signed [7:0] n_imgB_ch1_buf [0:35];
reg signed [7:0] n_imgB_ch2_buf [0:35];
reg signed [7:0] n_ker_ch1_buf  [0:8];
reg signed [7:0] n_ker_ch2_buf  [0:8];
reg signed [7:0] n_weight_buf   [0:31];

reg [7:0] cnn_load_cnt, n_cnn_load_cnt;
reg [6:0] cnn_feed_cnt, n_cnn_feed_cnt;
reg [2:0] cnn_burst_id, n_cnn_burst_id;
reg [7:0] cnn_burst_cnt, n_cnn_burst_cnt;
reg [1:0] cnn_out_cnt, n_cnn_out_cnt;

reg cnn_busy, n_cnn_busy;
reg cnn_pending, n_cnn_pending;
reg [3:0] cnn_rd, n_cnn_rd;
integer i;

CNN u_CNN(
    .clk(clk),
    .rst_n(rst_n),
    .in_valid(cnn_in_valid),
    .mode(cnn_mode),
    .in_data_ch1(cnn_in_data_ch1),
    .in_data_ch2(cnn_in_data_ch2),
    .kernel_ch1(cnn_kernel_ch1),
    .kernel_ch2(cnn_kernel_ch2),
    .weight(cnn_weight),
    .out_valid(cnn_out_valid),
    .out_data(cnn_out_data)
);

always @(*) begin
    next = state;

    n_IO_stall = 1'b1;

    n_pc = pc;
    n_inst_reg = inst_reg;
    n_inst_buf_cnt  = inst_buf_cnt;
    n_inst_exec_cnt = inst_exec_cnt;

    n_awaddr_m_inf_data  = awaddr_m_inf_data;
    n_awvalid_m_inf_data = 0;
    n_awlen_m_inf_data   = awlen_m_inf_data;

    n_wdata_m_inf_data   = wdata_m_inf_data;
    n_wvalid_m_inf_data  = 0;
    n_wlast_m_inf_data   = 0;

    n_bready_m_inf_data  = 0;

    n_araddr_m_inf_data  = araddr_m_inf_data;
    n_arvalid_m_inf_data = 0;
    n_arlen_m_inf_data   = arlen_m_inf_data;

    n_rready_m_inf_data  = 0;

    n_araddr_m_inf_inst  = araddr_m_inf_inst;
    n_arvalid_m_inf_inst = 0;
    n_arlen_m_inf_inst   = arlen_m_inf_inst;

    n_rready_m_inf_inst  = 0;

    n_cnn_in_valid = 0;
    n_cnn_mode = cnn_mode;
    n_cnn_in_data_ch1 = cnn_in_data_ch1;
    n_cnn_in_data_ch2 = cnn_in_data_ch2;
    n_cnn_kernel_ch1 = cnn_kernel_ch1;
    n_cnn_kernel_ch2 = cnn_kernel_ch2;
    n_cnn_weight = cnn_weight;

    n_cnn_cnt = cnn_cnt;
    n_cnn_load_cnt = cnn_load_cnt;
    n_cnn_feed_cnt = cnn_feed_cnt;
    n_cnn_burst_id  = cnn_burst_id;
    n_cnn_burst_cnt = cnn_burst_cnt;
    n_cnn_busy      = cnn_busy;
    n_cnn_pending   = cnn_pending;
    n_cnn_rd        = cnn_rd;
    n_cnn_out_cnt   = cnn_out_cnt;
    for(i = 0; i < 16; i = i + 1)
        n_reg_file[i] = reg_file[i];

    for(i = 0; i < 8; i = i + 1)
        n_inst_buf[i] = inst_buf[i];

    for(i = 0; i < 4; i = i + 1)
        n_cnn_out_data_form[i] = cnn_out_data_form[i];

    for(i = 0; i < 36; i = i + 1) begin
        n_imgA_ch1_buf[i] = imgA_ch1_buf[i];
        n_imgA_ch2_buf[i] = imgA_ch2_buf[i];
        n_imgB_ch1_buf[i] = imgB_ch1_buf[i];
        n_imgB_ch2_buf[i] = imgB_ch2_buf[i];
    end

    for(i = 0; i < 9; i = i + 1) begin
        n_ker_ch1_buf[i] = ker_ch1_buf[i];
        n_ker_ch2_buf[i] = ker_ch2_buf[i];
    end

    for(i = 0; i < 32; i = i + 1)
        n_weight_buf[i] = weight_buf[i];

    // Background CNN output collection for overlapped CNN execution
    if(cnn_busy && cnn_out_valid) begin
        n_cnn_out_data_form[cnn_out_cnt] = cnn_out_data;

        if(cnn_out_cnt == 2'd3) begin
            if(cnn_out_data_form[0] >= cnn_out_data_form[1] &&
               cnn_out_data_form[0] >= cnn_out_data_form[2] &&
               cnn_out_data_form[0] >= cnn_out_data) begin
                n_reg_file[cnn_rd] = 8'd0;
            end
            else if(cnn_out_data_form[1] >= cnn_out_data_form[2] &&
                    cnn_out_data_form[1] >= cnn_out_data) begin
                n_reg_file[cnn_rd] = 8'd1;
            end
            else if(cnn_out_data_form[2] >= cnn_out_data) begin
                n_reg_file[cnn_rd] = 8'd2;
            end
            else begin
                n_reg_file[cnn_rd] = 8'd3;
            end

            n_cnn_busy    = 1'b0;
            n_cnn_pending = 1'b0;
            n_cnn_out_cnt = 2'd0;
        end
        else begin
            n_cnn_out_cnt = cnn_out_cnt + 1'b1;
        end
    end

    case(state)

        S_IDLE: begin
            n_IO_stall = 1;
            n_pc = 0;
            next = S_IF_AR;
        end

        S_IF_AR: begin
            n_IO_stall = 1;
            n_araddr_m_inf_inst  = pc;
            n_arvalid_m_inf_inst = 1;
            n_arlen_m_inf_inst   = 8'd7;

            if(arvalid_m_inf_inst && arready_m_inf_inst) begin
                n_inst_buf_cnt = 0;
                next = S_IF_R;
            end
            else begin
                next = S_IF_AR;
            end
        end

        S_IF_R: begin
            n_rready_m_inf_inst = 1;

            if(rready_m_inf_inst && rvalid_m_inf_inst) begin
                n_inst_buf[inst_buf_cnt] = rdata_m_inf_inst;

                if(rlast_m_inf_inst) begin
                    n_inst_buf[inst_buf_cnt] = rdata_m_inf_inst;

                    n_inst_buf_cnt  = 0;
                    n_inst_exec_cnt = 0;

                    n_inst_reg = (inst_buf_cnt == 0) ?
                                rdata_m_inf_inst :
                                inst_buf[0];

                    next = S_DECODE;
                end
                else begin
                    n_inst_buf_cnt = inst_buf_cnt + 1'b1;
                    next = S_IF_R;
                end
            end
            else begin
                next = S_IF_R;
            end
        end

        S_DECODE: begin
            if(cnn_pending && (
                ((opcode == 3'b000) && (rs == cnn_rd || r_src2 == cnn_rd)) ||
                ((opcode == 3'b001) && (rs == cnn_rd || r_src2 == cnn_rd)) ||
                ((opcode == 3'b010) && (rs == cnn_rd)) ||
                ((opcode == 3'b011) && (rs == cnn_rd || rt_i == cnn_rd)) ||
                ((opcode == 3'b100) && (rs == cnn_rd || rt_i == cnn_rd))
            )) begin
                next = S_DECODE;
            end
            else begin
                case(opcode)
                    3'b000: next = S_EXEC;
                    3'b001: next = S_EXEC;
                    3'b010: next = S_DATA_AR;
                    3'b011: next = S_DATA_AW;
                    3'b100: next = S_EXEC;
                    3'b101: next = S_EXEC;
                    3'b111: begin
                        n_cnn_load_cnt = 0;
                        n_cnn_feed_cnt = 0;
                        n_cnn_cnt = 0;
                        n_cnn_burst_id = 0;
                        n_cnn_burst_cnt = 0;
                        next = S_CNN_LOAD_AR;
                    end
                    default: next = S_DECODE;
                endcase
            end
        end

        S_EXEC: begin
            case(opcode)
                3'b000: begin
                    if(inst_reg[0] == 1'b0)
                        n_reg_file[r_dst] = reg_file[rs] + reg_file[r_src2];
                    else
                        n_reg_file[r_dst] = reg_file[rs] - reg_file[r_src2];

                    next = S_DONE;
                end

                3'b001: begin
                    n_reg_file[r_dst] = reg_file[rs] * reg_file[r_src2];
                    next = S_DONE;
                end

                3'b100: begin
                    if(reg_file[rs] == reg_file[rt_i])
                        n_pc = pc + 1 + imm;
                    else
                        n_pc = pc + 1;

                    next = S_DONE;
                end

                3'b101: begin
                    n_pc = {19'd0, j_addr};
                    next = S_DONE;
                end

                default: begin
                    next = S_DONE;
                end
            endcase
        end

        S_DATA_AR: begin
            n_arvalid_m_inf_data = 1;
            n_araddr_m_inf_data  = 32'h0000_1000 + data_offset;
            n_arlen_m_inf_data   = 0;

            if(arvalid_m_inf_data && arready_m_inf_data)
                next = S_DATA_R;
            else
                next = S_DATA_AR;
        end

        S_DATA_R: begin
            n_rready_m_inf_data = 1;

            if(rready_m_inf_data && rvalid_m_inf_data) begin
                n_reg_file[rt_i] = rdata_m_inf_data;
                next = S_DONE;
            end
            else begin
                next = S_DATA_R;
            end
        end

        S_DATA_AW: begin
            n_awvalid_m_inf_data = 1;
            n_awaddr_m_inf_data  = 32'h0000_1000 + data_offset;
            n_awlen_m_inf_data   = 0;

            if(awvalid_m_inf_data && awready_m_inf_data)
                next = S_DATA_W;
            else
                next = S_DATA_AW;
        end

        S_DATA_W: begin
            n_wvalid_m_inf_data = 1;
            n_wdata_m_inf_data  = reg_file[rt_i];
            n_wlast_m_inf_data  = 1;

            if(wvalid_m_inf_data && wready_m_inf_data)
                next = S_DATA_B;
            else
                next = S_DATA_W;
        end

        S_DATA_B: begin
            n_bready_m_inf_data = 1;

            if(bready_m_inf_data && bvalid_m_inf_data)
                next = S_DONE;
            else
                next = S_DATA_B;
        end

        S_CNN_LOAD_AR: begin
            n_arvalid_m_inf_data = 1;

            if((imgB == (imgA + 3'd1)) && (cnn_burst_id == 3'd0)) begin
                n_araddr_m_inf_data = 32'h0000_1000 + imgA * 32'd72;
                n_arlen_m_inf_data  = 8'd143;  // imgA + imgB, 144 bytes
            end
            else begin
                case(cnn_burst_id)
                    3'd0: begin
                        n_araddr_m_inf_data = 32'h0000_1000 + imgA * 32'd72;
                        n_arlen_m_inf_data  = 8'd71;   // imgA ch1+ch2, 72 bytes
                    end
                    3'd1: begin
                        n_araddr_m_inf_data = 32'h0000_1000 + imgB * 32'd72;
                        n_arlen_m_inf_data  = 8'd71;   // imgB ch1+ch2, 72 bytes
                    end
                    3'd2: begin
                        n_araddr_m_inf_data = 32'h0000_1240 + k_sel * 32'd18;
                        n_arlen_m_inf_data  = 8'd17;   // kernel ch1+ch2, 18 bytes
                    end
                    default: begin
                        n_araddr_m_inf_data = 32'h0000_1264 + w_sel * 32'd32;
                        n_arlen_m_inf_data  = 8'd31;   // weight, 32 bytes
                    end
                endcase
            end

            if(arvalid_m_inf_data && arready_m_inf_data) begin
                n_cnn_burst_cnt = 0;
                next = S_CNN_LOAD_R;
            end
            else begin
                next = S_CNN_LOAD_AR;
            end
        end

        S_CNN_LOAD_R: begin
            n_rready_m_inf_data = 1;

            if(rready_m_inf_data && rvalid_m_inf_data) begin
                if((imgB == (imgA + 3'd1)) && (cnn_burst_id == 3'd0)) begin
                    if(cnn_burst_cnt < 8'd36)
                        n_imgA_ch1_buf[cnn_burst_cnt] = rdata_m_inf_data;
                    else if(cnn_burst_cnt < 8'd72)
                        n_imgA_ch2_buf[cnn_burst_cnt - 8'd36] = rdata_m_inf_data;
                    else if(cnn_burst_cnt < 8'd108)
                        n_imgB_ch1_buf[cnn_burst_cnt - 8'd72] = rdata_m_inf_data;
                    else
                        n_imgB_ch2_buf[cnn_burst_cnt - 8'd108] = rdata_m_inf_data;
                end
                else begin
                    case(cnn_burst_id)
                        3'd0: begin
                            if(cnn_burst_cnt < 8'd36)
                                n_imgA_ch1_buf[cnn_burst_cnt] = rdata_m_inf_data;
                            else
                                n_imgA_ch2_buf[cnn_burst_cnt - 8'd36] = rdata_m_inf_data;
                        end

                        3'd1: begin
                            if(cnn_burst_cnt < 8'd36)
                                n_imgB_ch1_buf[cnn_burst_cnt] = rdata_m_inf_data;
                            else
                                n_imgB_ch2_buf[cnn_burst_cnt - 8'd36] = rdata_m_inf_data;
                        end

                        3'd2: begin
                            if(cnn_burst_cnt < 8'd9)
                                n_ker_ch1_buf[cnn_burst_cnt] = rdata_m_inf_data;
                            else
                                n_ker_ch2_buf[cnn_burst_cnt - 8'd9] = rdata_m_inf_data;
                        end

                        default: begin
                            n_weight_buf[cnn_burst_cnt] = rdata_m_inf_data;
                        end
                    endcase
                end

                if(rlast_m_inf_data) begin
                    n_cnn_burst_cnt = 0;

                    if((imgB == (imgA + 3'd1)) && (cnn_burst_id == 3'd0)) begin
                        n_cnn_burst_id = 3'd2;
                        next = S_CNN_LOAD_AR;
                    end
                    else if(cnn_burst_id == 3'd3) begin
                        n_cnn_burst_id = 0;
                        n_cnn_load_cnt = 0;
                        n_cnn_feed_cnt = 0;
                        next = S_CNN_IN;
                    end
                    else begin
                        n_cnn_burst_id = cnn_burst_id + 1'b1;
                        next = S_CNN_LOAD_AR;
                    end
                end
                else begin
                    n_cnn_burst_cnt = cnn_burst_cnt + 1'b1;
                    next = S_CNN_LOAD_R;
                end
            end
            else begin
                next = S_CNN_LOAD_R;
            end
        end

        S_CNN_IN: begin
            n_cnn_in_valid = 1;
            n_cnn_mode = inst_reg[0];

            if(cnn_feed_cnt < 7'd36) begin
                n_cnn_in_data_ch1 = imgA_ch1_buf[cnn_feed_cnt];
                n_cnn_in_data_ch2 = imgA_ch2_buf[cnn_feed_cnt];
            end
            else begin
                n_cnn_in_data_ch1 = imgB_ch1_buf[cnn_feed_cnt - 7'd36];
                n_cnn_in_data_ch2 = imgB_ch2_buf[cnn_feed_cnt - 7'd36];
            end

            if(cnn_feed_cnt < 7'd9) begin
                n_cnn_kernel_ch1 = ker_ch1_buf[cnn_feed_cnt];
                n_cnn_kernel_ch2 = ker_ch2_buf[cnn_feed_cnt];
            end
            else begin
                n_cnn_kernel_ch1 = 0;
                n_cnn_kernel_ch2 = 0;
            end

            if(cnn_feed_cnt < 7'd32)
                n_cnn_weight = weight_buf[cnn_feed_cnt];
            else
                n_cnn_weight = 0;

            if(cnn_feed_cnt == 71) begin
                n_cnn_busy    = 1'b1;
                n_cnn_pending = 1'b1;
                n_cnn_rd      = inst_reg[6:3];
                n_cnn_out_cnt = 2'd0;
                n_cnn_out_data_form[0] = 20'd0;
                n_cnn_out_data_form[1] = 20'd0;
                n_cnn_out_data_form[2] = 20'd0;
                n_cnn_out_data_form[3] = 20'd0;

                next = S_DONE;
            end
            else begin
                n_cnn_feed_cnt = cnn_feed_cnt + 1;
                next = S_CNN_IN;
            end
        end

        S_CNN_OUT: begin
            if(cnn_out_valid) begin
                n_cnn_out_data_form[cnn_cnt] = cnn_out_data;

                if(cnn_cnt == 2'd3) begin
                    if(cnn_out_data_form[0] >= cnn_out_data_form[1] &&
                       cnn_out_data_form[0] >= cnn_out_data_form[2] &&
                       cnn_out_data_form[0] >= cnn_out_data) begin
                        n_reg_file[inst_reg[6:3]] = 0;
                    end
                    else if(cnn_out_data_form[1] >= cnn_out_data_form[2] &&
                            cnn_out_data_form[1] >= cnn_out_data) begin
                        n_reg_file[inst_reg[6:3]] = 1;
                    end
                    else if(cnn_out_data_form[2] >= cnn_out_data) begin
                        n_reg_file[inst_reg[6:3]] = 2;
                    end
                    else begin
                        n_reg_file[inst_reg[6:3]] = 3;
                    end

                    n_cnn_cnt = 0;
                    next = S_DONE;
                end
                else begin
                    n_cnn_cnt = cnn_cnt + 1;
                    next = S_CNN_OUT;
                end
            end
            else begin
                next = S_CNN_OUT;
            end
        end

        S_DONE: begin
            n_IO_stall = 0;

            if(opcode == 3'b100 || opcode == 3'b101) begin
                n_inst_buf_cnt  = 0;
                n_inst_exec_cnt = 0;
                next = S_IF_AR;
            end
            else begin
                n_pc = pc + 1;

                if(inst_exec_cnt == 3'd7) begin
                    n_inst_exec_cnt = 0;
                    n_inst_buf_cnt  = 0;
                    next = S_IF_AR;
                end
                else begin
                    n_inst_exec_cnt = inst_exec_cnt + 1'b1;
                    n_inst_reg = inst_buf[inst_exec_cnt + 1'b1];
                    next = S_DECODE;
                end
            end
        end

        default: begin
            next = S_IDLE;
        end
    endcase
end

always @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
        state <= S_IDLE;
        IO_stall <= 1;

        pc <= 0;
        inst_reg <= 0;
        inst_buf_cnt <= 0;
        inst_exec_cnt <= 0;

        awaddr_m_inf_data  <= 0;
        awvalid_m_inf_data <= 0;
        awlen_m_inf_data   <= 0;

        wdata_m_inf_data   <= 0;
        wvalid_m_inf_data  <= 0;
        wlast_m_inf_data   <= 0;

        bready_m_inf_data  <= 0;

        araddr_m_inf_data  <= 0;
        arvalid_m_inf_data <= 0;
        arlen_m_inf_data   <= 0;

        rready_m_inf_data  <= 0;

        araddr_m_inf_inst  <= 0;
        arvalid_m_inf_inst <= 0;
        arlen_m_inf_inst   <= 0;

        rready_m_inf_inst  <= 0;

        cnn_in_valid <= 0;
        cnn_mode <= 0;
        cnn_in_data_ch1 <= 0;
        cnn_in_data_ch2 <= 0;
        cnn_kernel_ch1 <= 0;
        cnn_kernel_ch2 <= 0;
        cnn_weight <= 0;

        cnn_cnt <= 0;
        cnn_load_cnt <= 0;
        cnn_feed_cnt <= 0;
        cnn_burst_id  <= 0;
        cnn_burst_cnt <= 0;
        cnn_busy      <= 0;
        cnn_pending   <= 0;
        cnn_rd        <= 0;
        cnn_out_cnt   <= 0;
        for(i = 0; i < 16; i = i + 1)
            reg_file[i] <= 0;

        for(i = 0; i < 8; i = i + 1)
            inst_buf[i] <= 0;

        for(i = 0; i < 4; i = i + 1)
            cnn_out_data_form[i] <= 0;

        for(i = 0; i < 36; i = i + 1) begin
            imgA_ch1_buf[i] <= 0;
            imgA_ch2_buf[i] <= 0;
            imgB_ch1_buf[i] <= 0;
            imgB_ch2_buf[i] <= 0;
        end

        for(i = 0; i < 9; i = i + 1) begin
            ker_ch1_buf[i] <= 0;
            ker_ch2_buf[i] <= 0;
        end

        for(i = 0; i < 32; i = i + 1)
            weight_buf[i] <= 0;
    end
    else begin
        state <= next;
        IO_stall <= n_IO_stall;

        pc <= n_pc;
        inst_reg <= n_inst_reg;
        inst_buf_cnt <= n_inst_buf_cnt;
        inst_exec_cnt <= n_inst_exec_cnt;

        awaddr_m_inf_data  <= n_awaddr_m_inf_data;
        awvalid_m_inf_data <= n_awvalid_m_inf_data;
        awlen_m_inf_data   <= n_awlen_m_inf_data;

        wdata_m_inf_data   <= n_wdata_m_inf_data;
        wvalid_m_inf_data  <= n_wvalid_m_inf_data;
        wlast_m_inf_data   <= n_wlast_m_inf_data;

        bready_m_inf_data  <= n_bready_m_inf_data;

        araddr_m_inf_data  <= n_araddr_m_inf_data;
        arvalid_m_inf_data <= n_arvalid_m_inf_data;
        arlen_m_inf_data   <= n_arlen_m_inf_data;

        rready_m_inf_data  <= n_rready_m_inf_data;

        araddr_m_inf_inst  <= n_araddr_m_inf_inst;
        arvalid_m_inf_inst <= n_arvalid_m_inf_inst;
        arlen_m_inf_inst   <= n_arlen_m_inf_inst;

        rready_m_inf_inst  <= n_rready_m_inf_inst;

        cnn_in_valid <= n_cnn_in_valid;
        cnn_mode <= n_cnn_mode;
        cnn_in_data_ch1 <= n_cnn_in_data_ch1;
        cnn_in_data_ch2 <= n_cnn_in_data_ch2;
        cnn_kernel_ch1 <= n_cnn_kernel_ch1;
        cnn_kernel_ch2 <= n_cnn_kernel_ch2;
        cnn_weight <= n_cnn_weight;

        cnn_cnt <= n_cnn_cnt;
        cnn_load_cnt <= n_cnn_load_cnt;
        cnn_feed_cnt <= n_cnn_feed_cnt;
        cnn_burst_id  <= n_cnn_burst_id;
        cnn_burst_cnt <= n_cnn_burst_cnt;
        cnn_busy      <= n_cnn_busy;
        cnn_pending   <= n_cnn_pending;
        cnn_rd        <= n_cnn_rd;
        cnn_out_cnt   <= n_cnn_out_cnt;
        for(i = 0; i < 16; i = i + 1)
            reg_file[i] <= n_reg_file[i];

        for(i = 0; i < 8; i = i + 1)
            inst_buf[i] <= n_inst_buf[i];

        for(i = 0; i < 4; i = i + 1)
            cnn_out_data_form[i] <= n_cnn_out_data_form[i];

        for(i = 0; i < 36; i = i + 1) begin
            imgA_ch1_buf[i] <= n_imgA_ch1_buf[i];
            imgA_ch2_buf[i] <= n_imgA_ch2_buf[i];
            imgB_ch1_buf[i] <= n_imgB_ch1_buf[i];
            imgB_ch2_buf[i] <= n_imgB_ch2_buf[i];
        end

        for(i = 0; i < 9; i = i + 1) begin
            ker_ch1_buf[i] <= n_ker_ch1_buf[i];
            ker_ch2_buf[i] <= n_ker_ch2_buf[i];
        end

        for(i = 0; i < 32; i = i + 1)
            weight_buf[i] <= n_weight_buf[i];
    end
end

endmodule