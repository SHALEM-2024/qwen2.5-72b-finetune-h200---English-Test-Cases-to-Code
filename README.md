# Bridging the gap between Test Plans and Test Benches.

A QLoRA fine-tuning pipeline for Qwen-2.5-72B on H200 hardware, specialized for generating proprietary XML structures for Automotive Hardware-in-Loop systems.

In the automotive industry, verifying hardware involves a massive manual bottleneck: translating thousands of English test cases into proprietary code for dSPACE Automation Desk. This repository documents the engineering pipeline used to fine-tune a 72-Billion parameter LLM on an NVIDIA H200. The result is an AI agent capable of autonomously writing syntactically perfect .blkx (XML) test scripts, eliminating manual coding labor and significantly accelerating the vehicle validation Time-to-Market.

Details about the project:

This project tackles the complexity of generating strict, proprietary XML structures (.blkx) from unstructured natural language. Utilizing the massive compute of NVIDIA H200s, we fine-tuned Qwen-2.5-72B to understand the nuance of automotive verification. This pipeline moves beyond simple code completion, creating a domain-specific model that automates the rigorous workflow of dSPACE Hardware-in-Loop (HIL) testing, turning human intent into machine execution.

Here is a clean, structured README/Instruction document based on your terminal logs. It filters out the trial-and-error steps to provide a clear "Happy Path" for you and new team members to reproduce the training run.

Axolotl Training Workflow: Qwen 72B Fine-Tuning
This document outlines the standard operating procedure for initializing the environment and launching a fine-tuning job using axolotl on the workspace container.

1. Prerequisites
Environment: Access to the GPU workspace container.

Hardware: Sufficient VRAM for Qwen 72B (Logs indicate ~40GB+ utilized during training).

Files: Ensure config.yaml and fine_tuning_data.jsonl are present in the project root.

2. Setup & Installation
Important: Do not attempt to run commands from the root / directory. You must navigate to the workspace directory where the axolotl repository is cloned.

Step 2.1: Navigate to Project Directory
The setup.py and project files are located in /workspace/axolotl.

Bash

cd /workspace/axolotl
Step 2.2: Lock Version (Optional but Recommended)
To ensure reproducibility, checkout the specific commit version used in previous successful runs (e.g., prior to Nov 2, 2024).

Bash

git checkout $(git rev-list -n 1 --before="2024-11-02 12:00" main)
> Note: This will put git into a 'detached HEAD' state. This is normal and safe for training purposes.

Step 2.3: Install Dependencies
Install the package in editable mode without reinstalling dependencies (saves time if container is pre-built).

Bash

pip install -e . --no-deps
3. Pre-Flight Checks
Before launching, verify your configuration and dataset files exist in the current directory.

Bash

ls -lh fine_tuning_data.jsonl config.yaml
Expected Output: You should see both files listed with non-zero file sizes (e.g., 1.0M for the jsonl file).

4. Launching Training
Run the training module using accelerate.

Bash

accelerate launch -m axolotl.cli.train config.yaml
Understanding the Launch Log
Once the command is running, watch for the following stages in the logs to confirm success:

Configuration Warning: You may see warnings about defaults (--num_processes=1, mixed_precision='no'). These can usually be ignored unless you specifically need multi-node setup.

Dataset Processing:

Tokenizing Prompts: Progress bar showing tokenization.

Sample Packing: "explicitly setting eval_sample_packing to match sample_packing".

Model Loading:

Loading Qwen/Qwen2.5-72B-Instruct.

VRAM Check: GPU memory usage will jump (approx 38GB-40GB).

Training Loop:

Look for the loss output to verify the model is learning.

Example: {'loss': 0.4131, ... 'epoch': 0.05}
