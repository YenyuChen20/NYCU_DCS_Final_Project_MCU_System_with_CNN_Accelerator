# MCU System with CNN Accelerator

## Introduction

This project implements a Microcontroller Unit (MCU) integrated with a CNN accelerator using Verilog HDL. The system communicates with instruction and data memory through the AXI4 protocol and supports DMA-based data transfer for efficient CNN execution.

<p align="center">
<img width="320" height="230" alt="image" src="https://github.com/user-attachments/assets/bbe476cb-0b1c-48f5-9dec-ab3d3e8238cc" />
<img width="320" height="230" alt="image" src="https://github.com/user-attachments/assets/7850776b-20b2-48b5-81e9-afbebf093a78" />
</p>


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

 ## AXI4 Interface

The communication between the MCU and external memory is implemented using the AXI4 protocol with separate instruction and data channels.

For instruction fetching, an **8-beat burst read (ARLEN = 7)** is adopted. Eight consecutive 16-bit instructions are fetched in a single AXI transaction and stored in the instruction prefetch buffer. This reduces the number of AXI address handshakes while avoiding the extra hardware cost of a larger buffer, providing a good balance between memory bandwidth and hardware area.

Burst transfers are also applied to data memory accesses for continuous image, kernel, and weight data, improving overall memory access efficiency.

<p align="center">
  <img src="https://github.com/user-attachments/assets/ef26e9ab-0af0-47b5-978d-b59dc4d9db56" width="300">
  <img src="https://github.com/user-attachments/assets/22b0b649-9df4-4e09-a1be-1cef43a67262" width="300">
</p>


## Instruction Prefetch Buffer

To reduce the latency caused by frequent instruction memory accesses, an instruction prefetch buffer is implemented. Instead of requesting a new instruction from DRAM every cycle, the processor fetches eight consecutive instructions through a single AXI burst transaction and stores them in an internal buffer.

The buffer contains eight 16-bit instructions, which exactly matches one AXI burst transfer (ARLEN = 7). The CPU sequentially executes instructions from the buffer until all entries have been consumed, after which another burst read is issued to refill the buffer.

The buffer size is selected to match the burst length rather than being unnecessarily large. A smaller buffer would require more AXI transactions and increase address handshake overhead, while a larger buffer would consume additional hardware resources without providing significant performance improvement. Therefore, an 8-entry instruction prefetch buffer provides a good balance between execution efficiency and hardware cost.


  ## CNN Accelerator

The CNN accelerator is integrated into the MCU as a dedicated hardware module for accelerating convolutional neural network inference. The accelerator communicates with the microcontroller through a custom CNN instruction and accesses image data and kernel weights from the Data DRAM via the AXI4 interface.

When a CNN instruction is executed, the MCU configures the accelerator by selecting the target image, kernel, and operating mode. The CNN module performs the convolution and activation operations in hardware, generates four output values, and returns the index of the maximum output to the destination register. This hardware implementation significantly reduces the computation workload of the processor and improves the overall execution efficiency compared with software-based execution.

<img width="565" height="280" alt="image" src="https://github.com/user-attachments/assets/4fe4e959-a79b-4e24-a2a5-140e2231e054" />


  ## Highlights

- Designed a complete MCU architecture in Verilog
- Implemented AXI4 burst transactions
- Integrated a CNN hardware accelerator with the processor
- Optimized memory access through DMA
- Passed RTL, Synthesis, and Gate-Level simulations

## Course

NYCU Digital Circuit and System Laboratory Final Project

