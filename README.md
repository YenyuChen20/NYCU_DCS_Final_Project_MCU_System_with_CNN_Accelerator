# MCU System with CNN Accelerator

## Introduction

This project implements a Microcontroller Unit (MCU) integrated with a CNN accelerator using Verilog HDL. The system communicates with instruction and data memory through the AXI4 protocol and supports DMA-based data transfer for efficient CNN execution.

<img width="685" height="380" alt="image" src="https://github.com/user-attachments/assets/bbe476cb-0b1c-48f5-9dec-ab3d3e8238cc" />
<img width="632" height="380" alt="image" src="https://github.com/user-attachments/assets/7850776b-20b2-48b5-81e9-afbebf093a78" />



## Features

- Microcontroller (MCU)
- Instruction Decoder
- Register File
- ALU
- AXI4 Master Interface
- DMA Controller
- CNN Accelerator
- Burst Read / Write
- Instruction and Data DRAM Access

## Development Environment

- Verilog HDL
- Synopsys Design Compiler
- VCS
- nWave
- Linux

  ## CNN Accelerator

The CNN accelerator is integrated into the MCU as a dedicated hardware module for accelerating convolutional neural network inference. The accelerator communicates with the microcontroller through a custom CNN instruction and accesses image data and kernel weights from the Data DRAM via the AXI4 interface.

When a CNN instruction is executed, the MCU configures the accelerator by selecting the target image, kernel, and operating mode. The CNN module performs the convolution and activation operations in hardware, generates four output values, and returns the index of the maximum output to the destination register. This hardware implementation significantly reduces the computation workload of the processor and improves the overall execution efficiency compared with software-based execution.

<img width="765" height="380" alt="image" src="https://github.com/user-attachments/assets/4fe4e959-a79b-4e24-a2a5-140e2231e054" />


  ## Highlights

- Designed a complete MCU architecture in Verilog
- Implemented AXI4 burst transactions
- Integrated a CNN hardware accelerator with the processor
- Optimized memory access through DMA
- Passed RTL, Synthesis, and Gate-Level simulations

## Course

NYCU Digital Circuit and System Laboratory Final Project

