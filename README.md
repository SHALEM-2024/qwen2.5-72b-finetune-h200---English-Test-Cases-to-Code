# Bridging the gap between Test Plans and Test Benches.

A QLoRA fine-tuning pipeline for Qwen-2.5-72B on H200 hardware, specialized for generating proprietary XML structures for Automotive Hardware-in-Loop systems.

In the automotive industry, verifying hardware involves a massive manual bottleneck: translating thousands of English test cases into proprietary code for dSPACE Automation Desk. This repository documents the engineering pipeline used to fine-tune a 72-Billion parameter LLM on an NVIDIA H200. The result is an AI agent capable of autonomously writing syntactically perfect .blkx (XML) test scripts, eliminating manual coding labor and significantly accelerating the vehicle validation Time-to-Market.

Details about the project:

This project tackles the complexity of generating strict, proprietary XML structures (.blkx) from unstructured natural language. Utilizing the massive compute of NVIDIA H200s, we fine-tuned Qwen-2.5-72B to understand the nuance of automotive verification. This pipeline moves beyond simple code completion, creating a domain-specific model that automates the rigorous workflow of dSPACE Hardware-in-Loop (HIL) testing, turning human intent into machine execution.
