//############################################################################
//   2025 Digital Circuit and System Lab
//   Final Project : MCU System with CNN Instruction Acceleration
//   Author      : Ceres Lab 2025 MS1
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//   Date        : 2025/05/24
//   Version     : v1.0
//   File Name   : TOP.v
//   Module Name : TOP
//############################################################################
//==============================================//
//           TOP Module Declaration             //
//==============================================//
module TOP(
	// System IO 
	clk            	,	
	rst_n          	,	
	IO_stall        ,	

	// AXI4 IO for Data DRAM
        awaddr_m_inf_data,
        awvalid_m_inf_data,
        awready_m_inf_data,
        awlen_m_inf_data,     

        wdata_m_inf_data,
        wvalid_m_inf_data,
        wlast_m_inf_data,
        wready_m_inf_data,
                    
        
        bresp_m_inf_data,
        bvalid_m_inf_data,
        bready_m_inf_data,
                    
        araddr_m_inf_data,
        arvalid_m_inf_data,         
        arready_m_inf_data, 
        arlen_m_inf_data,

        rdata_m_inf_data,
        rvalid_m_inf_data,
        rlast_m_inf_data,
        rready_m_inf_data,
    // AXI4 IO for Instruction DRAM
        araddr_m_inf_inst,
        arvalid_m_inf_inst,         
        arready_m_inf_inst, 
        arlen_m_inf_inst,
        
        rdata_m_inf_inst,
        rvalid_m_inf_inst,
        rlast_m_inf_inst,
        rready_m_inf_inst   
);
// ===============================================================
//  			   		Parameters
// ===============================================================
parameter ADDR_WIDTH = 32;           // Do not modify
parameter DATA_WIDTH_inst = 16;      // Do not modify
parameter DATA_WIDTH_data = 8;       // Do not modify

// ===============================================================
//  					Input / Output 
// ===============================================================
// << System io port >>
input wire			  	clk,rst_n;
output reg 			    IO_stall;   
 
// << AXI Interface wire connecttion for pseudo Data DRAM read/write >>
// (1) 	axi write address channel 
// 		src master
output reg [ADDR_WIDTH-1:0]     awaddr_m_inf_data;
output reg [7:0]                awlen_m_inf_data;      // burst length 0~127
output reg                      awvalid_m_inf_data;
// 		src slave   
input wire                     awready_m_inf_data;
// -----------------------------
// (2)	axi write data channel 
// 		src master
output reg [DATA_WIDTH_data-1:0]  wdata_m_inf_data;
output reg                   wlast_m_inf_data;
output reg                   wvalid_m_inf_data;
// 		src slave
input wire                  wready_m_inf_data;
// -----------------------------
// (3)	axi write response channel 
// 		src slave
input wire  [1:0]           bresp_m_inf_data;
input wire                  bvalid_m_inf_data;
// 		src master 
output reg                   bready_m_inf_data;
// -----------------------------
// (4)	axi read address channel 
// 		src master
output reg [ADDR_WIDTH-1:0]     araddr_m_inf_data;
output reg [7:0]                arlen_m_inf_data;     // burst length 0~127
output reg                      arvalid_m_inf_data;
// 		src slave
input wire                     arready_m_inf_data;
// -----------------------------
// (5)	axi read data channel 
// 		src slave
input wire [DATA_WIDTH_data-1:0]  rdata_m_inf_data;
input wire                   rlast_m_inf_data;
input wire                   rvalid_m_inf_data;
// 		src master
output reg                    rready_m_inf_data;

// << AXI Interface wire connecttion for pseudo Instruction DRAM read >>
// -----------------------------
// (1)	axi read address channel 
// 		src master
output reg [ADDR_WIDTH-1:0]     araddr_m_inf_inst;
output reg [7:0]                arlen_m_inf_inst;     // burst length 0~127
output reg                      arvalid_m_inf_inst;
// 		src slave
input wire                     arready_m_inf_inst;
// -----------------------------
// (2)	axi read data channel 
// 		src slave
input wire [DATA_WIDTH_inst-1:0]  rdata_m_inf_inst;
input wire                   rlast_m_inf_inst;
input wire                   rvalid_m_inf_inst;
// 		src master
output reg                    rready_m_inf_inst;


// ===============================================================
//  					Signal Declaration 
// ===============================================================
localparam S_IDLE      = 5'd0;
localparam S_FETCH     = 5'd1;
localparam S_IF_AR     = 5'd2;
localparam S_IF_R      = 5'd3;
localparam S_DEC       = 5'd4;
localparam S_EXE       = 5'd5;
localparam S_LD_AR     = 5'd6;
localparam S_LD_R      = 5'd7;
localparam S_ST_AW     = 5'd8;
localparam S_ST_W      = 5'd9;
localparam S_ST_B      = 5'd10;
localparam S_CNN_AR    = 5'd11;
localparam S_CNN_R     = 5'd12;
localparam S_CNN_FEED  = 5'd13;
localparam S_CNN_COL   = 5'd14;
localparam S_DONE      = 5'd15;

reg [4:0] state;

reg signed [7:0] reg_file [0:15];

reg [12:0] pc;
reg [15:0] inst;

reg [15:0] inst_mem [0:127];
reg [6:0]  inst_cnt;
reg [12:0] inst_base;
reg        inst_valid;

wire cache_hit;
wire [6:0] inst_idx;

assign cache_hit = inst_valid && (pc >= inst_base) && (pc < inst_base + 13'd128);
assign inst_idx  = pc[6:0] - inst_base[6:0];

wire [2:0] op;
wire [3:0] rs, rt, rd_rtype;
wire r_func;
wire signed [4:0] imm5;
wire [12:0] jaddr;

assign op       = inst[15:13];
assign rs       = inst[12:9];
assign rt       = inst[8:5];
assign rd_rtype = inst[4:1];
assign r_func   = inst[0];
assign imm5     = inst[4:0];
assign jaddr    = inst[12:0];

wire [2:0] cnn_img_a;
wire [2:0] cnn_img_b;
wire [3:0] cnn_rd;
wire cnn_k, cnn_w, cnn_mode_bit;

assign cnn_img_a    = inst[12:10];
assign cnn_img_b    = inst[9:7];
assign cnn_rd       = inst[6:3];
assign cnn_k        = inst[2];
assign cnn_w        = inst[1];
assign cnn_mode_bit = inst[0];

reg signed [15:0] eff_addr;

reg [2:0]  burst_id;
reg [5:0]  burst_cnt;
reg [5:0]  burst_len;
reg [31:0] burst_addr;

reg [6:0] cnn_feed_cnt;
reg [1:0] cnn_out_cnt;

reg signed [7:0] imgA_ch1_mem [0:35];
reg signed [7:0] imgA_ch2_mem [0:35];
reg signed [7:0] imgB_ch1_mem [0:35];
reg signed [7:0] imgB_ch2_mem [0:35];
reg signed [7:0] ker1_mem [0:8];
reg signed [7:0] ker2_mem [0:8];
reg signed [7:0] w_mem [0:31];

reg cnn_in_valid;
reg cnn_mode;
reg signed [7:0] cnn_in_data_ch1;
reg signed [7:0] cnn_in_data_ch2;
reg signed [7:0] cnn_kernel_ch1;
reg signed [7:0] cnn_kernel_ch2;
reg signed [7:0] cnn_weight;

wire cnn_out_valid;
wire signed [19:0] cnn_out_data;

reg signed [19:0] cnn_res0, cnn_res1, cnn_res2, cnn_res3;

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
    eff_addr = $signed(reg_file[rs]) * $signed(imm5) + $signed(imm5);
end

always @(*) begin
    burst_addr = 32'd0;
    burst_len  = 6'd1;

    case (burst_id)
        3'd0: begin
            burst_addr = 32'h00001000 + cnn_img_a * 32'd72;
            burst_len  = 6'd36;
        end
        3'd1: begin
            burst_addr = 32'h00001024 + cnn_img_a * 32'd72;
            burst_len  = 6'd36;
        end
        3'd2: begin
            burst_addr = 32'h00001000 + cnn_img_b * 32'd72;
            burst_len  = 6'd36;
        end
        3'd3: begin
            burst_addr = 32'h00001024 + cnn_img_b * 32'd72;
            burst_len  = 6'd36;
        end
        3'd4: begin
            burst_addr = 32'h00001240 + cnn_k * 32'd18;
            burst_len  = 6'd9;
        end
        3'd5: begin
            burst_addr = 32'h00001249 + cnn_k * 32'd18;
            burst_len  = 6'd9;
        end
        3'd6: begin
            burst_addr = 32'h00001264 + cnn_w * 32'd32;
            burst_len  = 6'd32;
        end
        default: begin
            burst_addr = 32'd0;
            burst_len  = 6'd1;
        end
    endcase
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= S_IDLE;
        pc <= 13'd0;
        inst <= 16'd0;
        inst_cnt <= 7'd0;
        inst_base <= 13'd0;
        inst_valid <= 1'b0;
        IO_stall <= 1'b1;

        awaddr_m_inf_data <= 32'd0;
        awlen_m_inf_data <= 8'd0;
        awvalid_m_inf_data <= 1'b0;
        wdata_m_inf_data <= 8'd0;
        wvalid_m_inf_data <= 1'b0;
        wlast_m_inf_data <= 1'b0;
        bready_m_inf_data <= 1'b0;

        araddr_m_inf_data <= 32'd0;
        arvalid_m_inf_data <= 1'b0;
        arlen_m_inf_data <= 8'd0;
        rready_m_inf_data <= 1'b0;

        araddr_m_inf_inst <= 32'd0;
        arvalid_m_inf_inst <= 1'b0;
        arlen_m_inf_inst <= 8'd0;
        rready_m_inf_inst <= 1'b0;

        burst_id <= 3'd0;
        burst_cnt <= 6'd0;
        cnn_feed_cnt <= 7'd0;
        cnn_out_cnt <= 2'd0;

        cnn_in_valid <= 1'b0;
        cnn_mode <= 1'b0;
        cnn_in_data_ch1 <= 8'sd0;
        cnn_in_data_ch2 <= 8'sd0;
        cnn_kernel_ch1 <= 8'sd0;
        cnn_kernel_ch2 <= 8'sd0;
        cnn_weight <= 8'sd0;

        cnn_res0 <= 20'sd0;
        cnn_res1 <= 20'sd0;
        cnn_res2 <= 20'sd0;
        cnn_res3 <= 20'sd0;

        for (i = 0; i < 16; i = i + 1) reg_file[i] <= 8'sd0;
        for (i = 0; i < 128; i = i + 1) inst_mem[i] <= 16'd0;
        for (i = 0; i < 36; i = i + 1) begin
            imgA_ch1_mem[i] <= 8'sd0;
            imgA_ch2_mem[i] <= 8'sd0;
            imgB_ch1_mem[i] <= 8'sd0;
            imgB_ch2_mem[i] <= 8'sd0;
        end
        for (i = 0; i < 9; i = i + 1) begin
            ker1_mem[i] <= 8'sd0;
            ker2_mem[i] <= 8'sd0;
        end
        for (i = 0; i < 32; i = i + 1) w_mem[i] <= 8'sd0;
    end
    else begin
        IO_stall <= 1'b1;

        awvalid_m_inf_data <= 1'b0;
        wvalid_m_inf_data <= 1'b0;
        wlast_m_inf_data <= 1'b0;
        bready_m_inf_data <= 1'b0;

        arvalid_m_inf_data <= 1'b0;
        rready_m_inf_data <= 1'b0;

        arvalid_m_inf_inst <= 1'b0;
        rready_m_inf_inst <= 1'b0;

        cnn_in_valid <= 1'b0;
        cnn_in_data_ch1 <= 8'sd0;
        cnn_in_data_ch2 <= 8'sd0;
        cnn_kernel_ch1 <= 8'sd0;
        cnn_kernel_ch2 <= 8'sd0;
        cnn_weight <= 8'sd0;

        case (state)
            S_IDLE: begin
                state <= S_FETCH;
            end

            S_FETCH: begin
                if (cache_hit) begin
                    inst <= inst_mem[inst_idx];
                    state <= S_DEC;
                end
                else begin
                    inst_base <= {pc[12:7], 7'd0};
                    inst_cnt <= 7'd0;
                    inst_valid <= 1'b0;
                    state <= S_IF_AR;
                end
            end

            S_IF_AR: begin
                araddr_m_inf_inst <= {19'd0, inst_base};
                arlen_m_inf_inst <= 8'd127;
                arvalid_m_inf_inst <= 1'b1;

                if (arready_m_inf_inst) begin
                    inst_cnt <= 7'd0;
                    state <= S_IF_R;
                end
            end

            S_IF_R: begin
                rready_m_inf_inst <= 1'b1;

                if (rvalid_m_inf_inst) begin
                    inst_mem[inst_cnt] <= rdata_m_inf_inst;

                    if (rlast_m_inf_inst || inst_cnt == 7'd127) begin
                        inst_valid <= 1'b1;
                        state <= S_FETCH;
                    end
                    else begin
                        inst_cnt <= inst_cnt + 7'd1;
                    end
                end
            end

            S_DEC: begin
                if (op == 3'b010) begin
                    state <= S_LD_AR;
                end
                else if (op == 3'b011) begin
                    state <= S_ST_AW;
                end
                else if (op == 3'b111) begin
                    burst_id <= 3'd0;
                    burst_cnt <= 6'd0;
                    cnn_feed_cnt <= 7'd0;
                    cnn_out_cnt <= 2'd0;
                    cnn_mode <= cnn_mode_bit;
                    state <= S_CNN_AR;
                end
                else begin
                    state <= S_EXE;
                end
            end

            S_EXE: begin
                case (op)
                    3'b000: begin
                        if (!r_func)
                            reg_file[rd_rtype] <= reg_file[rs] + reg_file[rt];
                        else
                            reg_file[rd_rtype] <= reg_file[rs] - reg_file[rt];
                        pc <= pc + 13'd1;
                    end

                    3'b001: begin
                        reg_file[rd_rtype] <= reg_file[rs] * reg_file[rt];
                        pc <= pc + 13'd1;
                    end

                    3'b100: begin
                        if (reg_file[rs] == reg_file[rt])
                            pc <= pc + 13'd1 + {{8{imm5[4]}}, imm5};
                        else
                            pc <= pc + 13'd1;
                    end

                    3'b101: begin
                        pc <= jaddr;
                    end

                    default: begin
                        pc <= pc + 13'd1;
                    end
                endcase

                state <= S_DONE;
            end

            S_LD_AR: begin
                araddr_m_inf_data <= 32'h00001000 + {{16{eff_addr[15]}}, eff_addr};
                arlen_m_inf_data <= 8'd0;
                arvalid_m_inf_data <= 1'b1;

                if (arready_m_inf_data)
                    state <= S_LD_R;
            end

            S_LD_R: begin
                rready_m_inf_data <= 1'b1;

                if (rvalid_m_inf_data) begin
                    reg_file[rt] <= rdata_m_inf_data;
                    pc <= pc + 13'd1;
                    state <= S_DONE;
                end
            end

            S_ST_AW: begin
                awaddr_m_inf_data <= 32'h00001000 + {{16{eff_addr[15]}}, eff_addr};
                awlen_m_inf_data <= 8'd0;
                awvalid_m_inf_data <= 1'b1;

                if (awready_m_inf_data)
                    state <= S_ST_W;
            end

            S_ST_W: begin
                wdata_m_inf_data <= reg_file[rt];
                wvalid_m_inf_data <= 1'b1;
                wlast_m_inf_data <= 1'b1;

                if (wready_m_inf_data)
                    state <= S_ST_B;
            end

            S_ST_B: begin
                bready_m_inf_data <= 1'b1;

                if (bvalid_m_inf_data) begin
                    pc <= pc + 13'd1;
                    state <= S_DONE;
                end
            end

            S_CNN_AR: begin
                araddr_m_inf_data <= burst_addr;
                arlen_m_inf_data <= burst_len - 6'd1;
                arvalid_m_inf_data <= 1'b1;

                if (arready_m_inf_data) begin
                    burst_cnt <= 6'd0;
                    state <= S_CNN_R;
                end
            end

            S_CNN_R: begin
                rready_m_inf_data <= 1'b1;

                if (rvalid_m_inf_data) begin
                    case (burst_id)
                        3'd0: imgA_ch1_mem[burst_cnt] <= rdata_m_inf_data;
                        3'd1: imgA_ch2_mem[burst_cnt] <= rdata_m_inf_data;
                        3'd2: imgB_ch1_mem[burst_cnt] <= rdata_m_inf_data;
                        3'd3: imgB_ch2_mem[burst_cnt] <= rdata_m_inf_data;
                        3'd4: ker1_mem[burst_cnt] <= rdata_m_inf_data;
                        3'd5: ker2_mem[burst_cnt] <= rdata_m_inf_data;
                        3'd6: w_mem[burst_cnt] <= rdata_m_inf_data;
                        default: begin end
                    endcase

                    if (rlast_m_inf_data || burst_cnt == burst_len - 6'd1) begin
                        if (burst_id == 3'd6) begin
                            cnn_feed_cnt <= 7'd0;
                            cnn_out_cnt <= 2'd0;
                            state <= S_CNN_FEED;
                        end
                        else begin
                            burst_id <= burst_id + 3'd1;
                            burst_cnt <= 6'd0;
                            state <= S_CNN_AR;
                        end
                    end
                    else begin
                        burst_cnt <= burst_cnt + 6'd1;
                    end
                end
            end

            S_CNN_FEED: begin
                cnn_in_valid <= 1'b1;
                cnn_mode <= cnn_mode_bit;

                if (cnn_feed_cnt < 7'd36) begin
                    cnn_in_data_ch1 <= imgA_ch1_mem[cnn_feed_cnt[5:0]];
                    cnn_in_data_ch2 <= imgA_ch2_mem[cnn_feed_cnt[5:0]];
                end
                else begin
                    cnn_in_data_ch1 <= imgB_ch1_mem[cnn_feed_cnt - 7'd36];
                    cnn_in_data_ch2 <= imgB_ch2_mem[cnn_feed_cnt - 7'd36];
                end

                if (cnn_feed_cnt < 7'd9) begin
                    cnn_kernel_ch1 <= ker1_mem[cnn_feed_cnt[3:0]];
                    cnn_kernel_ch2 <= ker2_mem[cnn_feed_cnt[3:0]];
                end
                else begin
                    cnn_kernel_ch1 <= 8'sd0;
                    cnn_kernel_ch2 <= 8'sd0;
                end

                if (cnn_feed_cnt < 7'd32)
                    cnn_weight <= w_mem[cnn_feed_cnt[4:0]];
                else
                    cnn_weight <= 8'sd0;

                if (cnn_feed_cnt == 7'd71) begin
                    cnn_feed_cnt <= 7'd0;
                    cnn_out_cnt <= 2'd0;
                    state <= S_CNN_COL;
                end
                else begin
                    cnn_feed_cnt <= cnn_feed_cnt + 7'd1;
                end
            end

            S_CNN_COL: begin
                if (cnn_out_valid) begin
                    if (cnn_out_cnt == 2'd0) begin
                        cnn_res0 <= cnn_out_data;
                    end
                    else if (cnn_out_cnt == 2'd1) begin
                        cnn_res1 <= cnn_out_data;
                    end
                    else if (cnn_out_cnt == 2'd2) begin
                        cnn_res2 <= cnn_out_data;
                    end
                    else begin
                        cnn_res3 <= cnn_out_data;

                        if ((cnn_res1 > cnn_res0) &&
                            (cnn_res1 > cnn_res2) &&
                            (cnn_res1 > cnn_out_data))
                            reg_file[cnn_rd] <= 8'd1;
                        else if ((cnn_res2 > cnn_res0) &&
                                 (cnn_res2 > cnn_res1) &&
                                 (cnn_res2 > cnn_out_data))
                            reg_file[cnn_rd] <= 8'd2;
                        else if ((cnn_out_data > cnn_res0) &&
                                 (cnn_out_data > cnn_res1) &&
                                 (cnn_out_data > cnn_res2))
                            reg_file[cnn_rd] <= 8'd3;
                        else
                            reg_file[cnn_rd] <= 8'd0;

                        pc <= pc + 13'd1;
                        state <= S_DONE;
                    end

                    if (cnn_out_cnt != 2'd3)
                        cnn_out_cnt <= cnn_out_cnt + 2'd1;
                end
            end

            S_DONE: begin
                IO_stall <= 1'b0;
                state <= S_FETCH;
            end

            default: begin
                state <= S_IDLE;
            end
        endcase
    end
end

endmodule


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
    output  reg signed [19:0] out_data
);

localparam [2:0] ST_IDLE  = 3'd0;
localparam [2:0] ST_CONV  = 3'd1;
localparam [2:0] ST_QUANT = 3'd3;
localparam [2:0] ST_FC    = 3'd4;
localparam [2:0] ST_FC_WB = 3'd6;
localparam [2:0] ST_OUT   = 3'd5;

integer ii;

reg [2:0] st;

reg in_valid_d;
reg in_valid_p_d;
reg [6:0] load_cnt;
reg img0_ready;
reg mode_r;
reg pat_hold;

reg signed [7:0] img_ch1 [0:71];
reg signed [7:0] img_ch2 [0:71];
reg signed [7:0] ker1 [0:8];
reg signed [7:0] ker2 [0:8];
reg signed [7:0] w_mem [0:31];

reg conv_img;
reg [3:0] conv_pos;
reg [4:0] conv_mac;

reg signed [31:0] conv_acc0;
reg signed [31:0] conv_acc1;
reg signed [31:0] conv_acc2;
reg signed [31:0] conv_acc3;

reg signed [31:0] conv_sum0_r;
reg signed [31:0] conv_sum1_r;
reg signed [31:0] conv_sum2_r;
reg signed [31:0] conv_sum3_r;
reg conv_sum_valid;

reg signed [19:0] pool0 [0:3];
reg signed [19:0] pool1 [0:3];

reg signed [7:0] feat [0:7];

reg [1:0] fc_out_idx;
reg [2:0] fc_in_idx;
reg signed [31:0] fc_acc;
reg signed [19:0] fc_y [0:3];

reg [1:0] out_cnt;

reg [1:0] conv_oy;
reg [1:0] conv_ox0;
reg [1:0] conv_ox1;
reg [1:0] conv_ox2;
reg [1:0] conv_ox3;
reg [3:0] conv_rem9;
reg [1:0] conv_ky;
reg [1:0] conv_kx;
reg [6:0] conv_img_base;
reg [2:0] conv_row6;
reg [2:0] conv_col6_0;
reg [2:0] conv_col6_1;
reg [2:0] conv_col6_2;
reg [2:0] conv_col6_3;
reg [5:0] conv_pix6_0;
reg [5:0] conv_pix6_1;
reg [5:0] conv_pix6_2;
reg [5:0] conv_pix6_3;
reg [6:0] conv_idx72_0;
reg [6:0] conv_idx72_1;
reg [6:0] conv_idx72_2;
reg [6:0] conv_idx72_3;

reg signed [7:0] conv_img0_ch1;
reg signed [7:0] conv_img0_ch2;
reg signed [7:0] conv_img1_ch1;
reg signed [7:0] conv_img1_ch2;
reg signed [7:0] conv_img2_ch1;
reg signed [7:0] conv_img2_ch2;
reg signed [7:0] conv_img3_ch1;
reg signed [7:0] conv_img3_ch2;
reg signed [7:0] conv_ker_v1;
reg signed [7:0] conv_ker_v2;

reg signed [7:0] conv_op_img0_ch1_r;
reg signed [7:0] conv_op_img0_ch2_r;
reg signed [7:0] conv_op_img1_ch1_r;
reg signed [7:0] conv_op_img1_ch2_r;
reg signed [7:0] conv_op_img2_ch1_r;
reg signed [7:0] conv_op_img2_ch2_r;
reg signed [7:0] conv_op_img3_ch1_r;
reg signed [7:0] conv_op_img3_ch2_r;
reg signed [7:0] conv_op_ker1_r;
reg signed [7:0] conv_op_ker2_r;

reg signed [15:0] conv_prod0_ch1_r;
reg signed [15:0] conv_prod0_ch2_r;
reg signed [15:0] conv_prod1_ch1_r;
reg signed [15:0] conv_prod1_ch2_r;
reg signed [15:0] conv_prod2_ch1_r;
reg signed [15:0] conv_prod2_ch2_r;
reg signed [15:0] conv_prod3_ch1_r;
reg signed [15:0] conv_prod3_ch2_r;

reg signed [16:0] conv_prod_sum0;
reg signed [16:0] conv_prod_sum1;
reg signed [16:0] conv_prod_sum2;
reg signed [16:0] conv_prod_sum3;

reg signed [31:0] conv_prod_sum_ext0;
reg signed [31:0] conv_prod_sum_ext1;
reg signed [31:0] conv_prod_sum_ext2;
reg signed [31:0] conv_prod_sum_ext3;

reg conv_op_valid;
reg conv_prod_valid;

reg signed [19:0] conv_actv0;
reg signed [19:0] conv_actv1;
reg signed [19:0] conv_actv2;
reg signed [19:0] conv_actv3;

reg signed [19:0] prev0_r0;
reg signed [19:0] prev0_r1;
reg signed [19:0] prev0_r2;
reg signed [19:0] prev0_r3;
reg signed [19:0] prev1_r0;
reg signed [19:0] prev1_r1;
reg signed [19:0] prev1_r2;
reg signed [19:0] prev1_r3;

reg [1:0] pool_idx_base;

reg signed [7:0] fc_w_v0;
reg signed [7:0] fc_w_v1;
reg signed [7:0] fc_x_v0;
reg signed [7:0] fc_x_v1;
reg signed [15:0] fc_prod0;
reg signed [15:0] fc_prod1;
reg signed [16:0] fc_pair_sum;
reg signed [16:0] fc_pair_sum_r;
reg fc_pair_valid;
reg fc_pair_last;
reg fc_drain;

reg signed [7:0] fc_op_w0_r;
reg signed [7:0] fc_op_w1_r;
reg signed [7:0] fc_op_x0_r;
reg signed [7:0] fc_op_x1_r;
reg fc_op_valid;
reg fc_op_last;
reg signed [15:0] fc_prod0_r2;
reg signed [15:0] fc_prod1_r2;
reg fc_prod_valid2;
reg fc_prod_last2;
reg signed [16:0] fc_sum_r2;
reg fc_sum_valid2;
reg fc_sum_last2;
reg signed [31:0] fc_sum_ext;
reg signed [31:0] fc_acc_with_sum;

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

    pool_idx_base = {conv_oy[1], 1'b0};

    fc_w_v0 = w_mem[{fc_out_idx, fc_in_idx}];
    fc_w_v1 = w_mem[{fc_out_idx, (fc_in_idx + 3'd1)}];
    fc_x_v0 = feat[fc_in_idx];
    fc_x_v1 = feat[fc_in_idx + 3'd1];

    fc_prod0 = $signed(fc_op_w0_r) * $signed(fc_op_x0_r);
    fc_prod1 = $signed(fc_op_w1_r) * $signed(fc_op_x1_r);

    fc_pair_sum = $signed(fc_prod0_r2) + $signed(fc_prod1_r2);
    fc_sum_ext = {{15{fc_sum_r2[16]}}, fc_sum_r2};
    fc_acc_with_sum = fc_acc + fc_sum_ext;
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        in_valid_d <= 1'b0;
        load_cnt <= 7'd0;
        img0_ready <= 1'b0;
        mode_r <= 1'b0;

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
            load_cnt <= 7'd0;
            img0_ready <= 1'b0;
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

                if (load_cnt == 7'd17) begin
                    img0_ready <= 1'b1;
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

        conv_sum0_r <= 32'sd0;
        conv_sum1_r <= 32'sd0;
        conv_sum2_r <= 32'sd0;
        conv_sum3_r <= 32'sd0;
        conv_sum_valid <= 1'b0;

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

        fc_out_idx <= 2'd0;
        fc_in_idx <= 3'd0;
        fc_acc <= 32'sd0;
        fc_pair_sum_r <= 17'sd0;
        fc_pair_valid <= 1'b0;
        fc_pair_last <= 1'b0;
        fc_drain <= 1'b0;
        fc_op_w0_r <= 8'sd0;
        fc_op_w1_r <= 8'sd0;
        fc_op_x0_r <= 8'sd0;
        fc_op_x1_r <= 8'sd0;
        fc_op_valid <= 1'b0;
        fc_op_last <= 1'b0;
        fc_prod0_r2 <= 16'sd0;
        fc_prod1_r2 <= 16'sd0;
        fc_prod_valid2 <= 1'b0;
        fc_prod_last2 <= 1'b0;
        fc_sum_r2 <= 17'sd0;
        fc_sum_valid2 <= 1'b0;
        fc_sum_last2 <= 1'b0;

        out_cnt <= 2'd0;

        for (ii = 0; ii < 4; ii = ii + 1) begin
            pool0[ii] <= 20'sd0;
            pool1[ii] <= 20'sd0;
            fc_y[ii] <= 20'sd0;
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
    end else begin
        in_valid_p_d <= in_valid;

        if (!in_valid_p_d && in_valid) begin
            pat_hold <= 1'b0;
        end

        out_valid <= 1'b0;
        out_data <= 20'sd0;

        case (st)
            ST_IDLE: begin
                if (img0_ready && !pat_hold) begin
                    conv_img <= 1'b0;
                    conv_pos <= 4'd0;
                    conv_mac <= 5'd0;

                    conv_acc0 <= 32'sd0;
                    conv_acc1 <= 32'sd0;
                    conv_acc2 <= 32'sd0;
                    conv_acc3 <= 32'sd0;

                    conv_sum0_r <= 32'sd0;
                    conv_sum1_r <= 32'sd0;
                    conv_sum2_r <= 32'sd0;
                    conv_sum3_r <= 32'sd0;
                    conv_sum_valid <= 1'b0;

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
                if (conv_sum_valid) begin
                    conv_acc0 <= conv_acc0 + conv_sum0_r;
                    conv_acc1 <= conv_acc1 + conv_sum1_r;
                    conv_acc2 <= conv_acc2 + conv_sum2_r;
                    conv_acc3 <= conv_acc3 + conv_sum3_r;
                end

                if (conv_mac <= 5'd8) begin
                    conv_sum0_r <= conv_prod_sum_ext0;
                    conv_sum1_r <= conv_prod_sum_ext1;
                    conv_sum2_r <= conv_prod_sum_ext2;
                    conv_sum3_r <= conv_prod_sum_ext3;
                    conv_sum_valid <= conv_prod_valid;

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
                    conv_sum0_r <= conv_prod_sum_ext0;
                    conv_sum1_r <= conv_prod_sum_ext1;
                    conv_sum2_r <= conv_prod_sum_ext2;
                    conv_sum3_r <= conv_prod_sum_ext3;
                    conv_sum_valid <= conv_prod_valid;

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
                    conv_sum0_r <= conv_prod_sum_ext0;
                    conv_sum1_r <= conv_prod_sum_ext1;
                    conv_sum2_r <= conv_prod_sum_ext2;
                    conv_sum3_r <= conv_prod_sum_ext3;
                    conv_sum_valid <= conv_prod_valid;

                    conv_prod_valid <= 1'b0;
                    conv_op_valid <= 1'b0;
                    conv_mac <= 5'd11;
                end else if (conv_mac == 5'd11) begin
                    conv_sum_valid <= 1'b0;
                    conv_prod_valid <= 1'b0;
                    conv_op_valid <= 1'b0;
                    conv_mac <= 5'd12;
                end else begin
                    if (!conv_oy[0]) begin
                        if (!conv_img) begin
                            prev0_r0 <= conv_actv0;
                            prev0_r1 <= conv_actv1;
                            prev0_r2 <= conv_actv2;
                            prev0_r3 <= conv_actv3;
                        end else begin
                            prev1_r0 <= conv_actv0;
                            prev1_r1 <= conv_actv1;
                            prev1_r2 <= conv_actv2;
                            prev1_r3 <= conv_actv3;
                        end
                    end else begin
                        if (!conv_img) begin
                            pool0[pool_idx_base] <= fn_max4_20(prev0_r0, prev0_r1, conv_actv0, conv_actv1);
                            pool0[pool_idx_base + 2'd1] <= fn_max4_20(prev0_r2, prev0_r3, conv_actv2, conv_actv3);
                        end else begin
                            pool1[pool_idx_base] <= fn_max4_20(prev1_r0, prev1_r1, conv_actv0, conv_actv1);
                            pool1[pool_idx_base + 2'd1] <= fn_max4_20(prev1_r2, prev1_r3, conv_actv2, conv_actv3);
                        end
                    end

                    conv_mac <= 5'd0;

                    conv_acc0 <= 32'sd0;
                    conv_acc1 <= 32'sd0;
                    conv_acc2 <= 32'sd0;
                    conv_acc3 <= 32'sd0;

                    conv_sum0_r <= 32'sd0;
                    conv_sum1_r <= 32'sd0;
                    conv_sum2_r <= 32'sd0;
                    conv_sum3_r <= 32'sd0;
                    conv_sum_valid <= 1'b0;

                    conv_op_valid <= 1'b0;
                    conv_prod_valid <= 1'b0;

                    if (conv_pos == 4'd3) begin
                        if (!conv_img) begin
                            conv_img <= 1'b1;
                            conv_pos <= 4'd0;
                        end else begin
                            st <= ST_QUANT;
                        end
                    end else begin
                        conv_pos <= conv_pos + 4'd1;
                    end
                end
            end

            ST_QUANT: begin
                feat[0] <= fn_quant8(pool0[0]);
                feat[1] <= fn_quant8(pool0[1]);
                feat[2] <= fn_quant8(pool0[2]);
                feat[3] <= fn_quant8(pool0[3]);
                feat[4] <= fn_quant8(pool1[0]);
                feat[5] <= fn_quant8(pool1[1]);
                feat[6] <= fn_quant8(pool1[2]);
                feat[7] <= fn_quant8(pool1[3]);

                fc_out_idx <= 2'd0;
                fc_in_idx <= 3'd0;
                fc_acc <= 32'sd0;
                fc_pair_sum_r <= 17'sd0;
                fc_pair_valid <= 1'b0;
                fc_pair_last <= 1'b0;
                fc_drain <= 1'b0;
                fc_op_w0_r <= 8'sd0;
                fc_op_w1_r <= 8'sd0;
                fc_op_x0_r <= 8'sd0;
                fc_op_x1_r <= 8'sd0;
                fc_op_valid <= 1'b0;
                fc_op_last <= 1'b0;
                fc_prod0_r2 <= 16'sd0;
                fc_prod1_r2 <= 16'sd0;
                fc_prod_valid2 <= 1'b0;
                fc_prod_last2 <= 1'b0;
                fc_sum_r2 <= 17'sd0;
                fc_sum_valid2 <= 1'b0;
                fc_sum_last2 <= 1'b0;

                st <= ST_FC;
            end

            ST_FC: begin
                if (fc_sum_valid2 && fc_sum_last2) begin
                    fc_y[fc_out_idx] <= fc_acc_with_sum[19:0];

                    fc_acc <= 32'sd0;
                    fc_in_idx <= 3'd0;
                    fc_pair_sum_r <= 17'sd0;
                    fc_pair_valid <= 1'b0;
                    fc_pair_last <= 1'b0;
                    fc_drain <= 1'b0;
                    fc_op_w0_r <= 8'sd0;
                    fc_op_w1_r <= 8'sd0;
                    fc_op_x0_r <= 8'sd0;
                    fc_op_x1_r <= 8'sd0;
                    fc_op_valid <= 1'b0;
                    fc_op_last <= 1'b0;
                    fc_prod0_r2 <= 16'sd0;
                    fc_prod1_r2 <= 16'sd0;
                    fc_prod_valid2 <= 1'b0;
                    fc_prod_last2 <= 1'b0;
                    fc_sum_r2 <= 17'sd0;
                    fc_sum_valid2 <= 1'b0;
                    fc_sum_last2 <= 1'b0;

                    if (fc_out_idx == 2'd3) begin
                        out_cnt <= 2'd0;
                        st <= ST_OUT;
                    end else begin
                        fc_out_idx <= fc_out_idx + 2'd1;
                        st <= ST_FC;
                    end
                end else begin
                    if (fc_sum_valid2) begin
                        fc_acc <= fc_acc_with_sum;
                    end

                    fc_sum_r2 <= fc_pair_sum;
                    fc_sum_valid2 <= fc_prod_valid2;
                    fc_sum_last2 <= fc_prod_last2;

                    fc_prod0_r2 <= fc_prod0;
                    fc_prod1_r2 <= fc_prod1;
                    fc_prod_valid2 <= fc_op_valid;
                    fc_prod_last2 <= fc_op_last;

                    if (!fc_drain) begin
                        fc_op_w0_r <= fc_w_v0;
                        fc_op_w1_r <= fc_w_v1;
                        fc_op_x0_r <= fc_x_v0;
                        fc_op_x1_r <= fc_x_v1;
                        fc_op_valid <= 1'b1;
                        fc_op_last <= (fc_in_idx == 3'd6);

                        if (fc_in_idx == 3'd6) begin
                            fc_drain <= 1'b1;
                            fc_in_idx <= 3'd0;
                        end else begin
                            fc_in_idx <= fc_in_idx + 3'd2;
                        end
                    end else begin
                        fc_op_w0_r <= 8'sd0;
                        fc_op_w1_r <= 8'sd0;
                        fc_op_x0_r <= 8'sd0;
                        fc_op_x1_r <= 8'sd0;
                        fc_op_valid <= 1'b0;
                        fc_op_last <= 1'b0;
                    end
                end
            end

            ST_OUT: begin
                out_valid <= 1'b1;
                out_data <= fc_y[out_cnt];

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