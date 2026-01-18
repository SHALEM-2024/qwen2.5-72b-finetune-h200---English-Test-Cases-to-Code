# Bridging the gap between Test Plans and Test Benches

**A QLoRA fine-tuning pipeline for Qwen-2.5-72B on H200 hardware, specialized for generating proprietary XML structures for Automotive Hardware-in-Loop systems.**

In the automotive industry, verifying hardware involves a massive manual bottleneck: translating thousands of English test cases into proprietary code for dSPACE Automation Desk. This repository documents the engineering pipeline used to fine-tune a 72-Billion parameter LLM on an NVIDIA H200. The result is an AI agent capable of autonomously writing syntactically perfect `.blkx` (XML) test scripts, eliminating manual coding labor and significantly accelerating the vehicle validation Time-to-Market.

### Details about the project

This project tackles the complexity of generating strict, proprietary XML structures (`.blkx`) from unstructured natural language. Utilizing the massive compute of NVIDIA H200s, we fine-tuned Qwen-2.5-72B to understand the nuance of automotive verification. This pipeline moves beyond simple code completion, creating a domain-specific model that automates the rigorous workflow of dSPACE Hardware-in-Loop (HIL) testing, turning human intent into machine execution.

---

## Axolotl Training Workflow: Qwen 72B Fine-Tuning

This section outlines the standard operating procedure for initializing the environment and launching a fine-tuning job using axolotl on the workspace container.

### 1. Prerequisites

* **Environment:** Access to the GPU workspace container.
* **Hardware:** Sufficient VRAM for Qwen 72B (Logs indicate ~40GB+ utilized during training).
* **Files:** Ensure `config.yaml` and `fine_tuning_data.jsonl` are present in the project root.

### 2. Setup & Installation

**Important:** Do not attempt to run commands from the root `/` directory. You must navigate to the workspace directory where the axolotl repository is cloned.

**Step 2.1: Navigate to Project Directory**
The `setup.py` and project files are located in `/workspace/axolotl`.

```bash
cd /workspace/axolotl

```

**Step 2.2: Lock Version (Optional but Recommended)**
To ensure reproducibility, checkout the specific commit version used in previous successful runs (e.g., prior to Nov 2, 2024).

```bash
git checkout $(git rev-list -n 1 --before="2024-11-02 12:00" main)

```

> *Note: This will put git into a 'detached HEAD' state. This is normal and safe for training purposes.*

**Step 2.3: Install Dependencies**
Install the package in editable mode without reinstalling dependencies (saves time if container is pre-built).

```bash
pip install -e . --no-deps

```

### 3. Pre-Flight Checks

Before launching, verify your configuration and dataset files exist in the current directory.

```bash
ls -lh fine_tuning_data.jsonl config.yaml

```

*Expected Output: You should see both files listed with non-zero file sizes (e.g., 1.0M for the jsonl file).*

### 4. Launching Training

Run the training module using `accelerate`.

```bash
accelerate launch -m axolotl.cli.train config.yaml

```

**Understanding the Launch Log**
Once the command is running, watch for the following stages in the logs to confirm success:

* **Configuration Warning:** You may see warnings about defaults (`--num_processes=1`, `mixed_precision='no'`). These can usually be ignored unless you specifically need multi-node setup.
* **Dataset Processing:**
* **Tokenizing Prompts:** Progress bar showing tokenization.
* **Sample Packing:** "explicitly setting eval_sample_packing to match sample_packing".


* **Model Loading:**
* Loading `Qwen/Qwen2.5-72B-Instruct`.
* **VRAM Check:** GPU memory usage will jump (approx 38GB-40GB).


* **Training Loop:** Look for the loss output to verify the model is learning.

---

## Model Upload Workflow

Follow these steps to upload your Qwen model adapter and tokenizer to Hugging Face.

### 🛑 Pre-flight Check

Before starting, ensure you are logged into Hugging Face on this machine.

```bash
huggingface-cli login

```

*(Paste your Write Token when asked).*

### Step 1: Navigate to the Correct Folder

Move into the folder where your output directory (`model-out`) is visible.

```bash
cd /workspace/axolotl

```

### Step 2: Run the Upload Script

Ensure the script `upload_direct.py` is present in your directory (refer to the repository files). This script handles creating the repo, and uploading the adapter files and the tokenizer.

```bash
python upload_direct.py

```

### Step 3: Verify

Watch the screen. Once you see `=== UPLOAD COMPLETE ===`, you can click the link it provides to see your model on the Hugging Face website.

---

## Phase 5: The Stress Test (Production Simulation)

This phase simulates a real-world scenario by processing large dictionary contexts and complex test steps. We utilize external text files to keep the Python execution clean.

### 1. Create Input Files

**A. Create the Context File**
Paste your full JSON Library Dictionary into `context.txt`.

```bash
nano context.txt
# Paste your long JSON dictionary here
# Save & Exit: Ctrl+O, Enter, Ctrl+X

```

**B. Create the Test Case File (Formatted)**
**CRITICAL:** You must structure your test case with clear headers: **Precondition**, **Action**, and **Postcondition**.

```bash
nano input.txt

```

**Paste your test case in this exact format:**

```text
Precondition:
1. Set Battery Voltage to 13.5V
... [Your Preconditions] ...

Action:
... [Your Actions] ...

Postcondition:
... [Your Postconditions] ...

```

### 2. Run the Test

Ensure `test_inference.py` is updated to read the files created above (refer to repository code).

```bash
python test_inference.py

```

### 3. Validate the Output

View the result or compare it to your expected XML.

```bash
cat output.xml

```

---

## Phase 6: Batch Processing & Context Filtering (The "V14" Engine)

This is the advanced automation script ("Version 14"). It upgrades your setup from a simple test to a production engine that can:

1. **Filter the Context:** It intelligently selects only the relevant dictionary items for your specific test case, allowing you to use massive libraries without hitting token limits.
2. **Batch Process:** It reads **all** `.txt` files in an `inputs/` folder and processes them one by one.
3. **Auto-Format:** It wraps the raw AI output in the correct `Standard.Sequence` XML structure automatically.

### 1. Setup Directories

Create a place to drop your test cases (`inputs`) and a place to collect the results (`outputs`).

```bash
mkdir -p inputs outputs

```

### 2. Add Your Test Cases

Create distinct text files for each test case inside the `inputs/` folder.

* **Format:** Use the same `Precondition:` / `Action:` / `Postcondition:` format from Phase 5.
* **Example:** `nano inputs/TC_001_Dark_Mode.txt`

### 3. Run Batch Inference

Ensure `batch_inference.py` is present in your directory. Execute the script to automatically iterate through every file in `inputs/` and save the corresponding XML in `outputs/`.

```bash
python batch_inference.py

```

### 4. Check Results

Your fully formatted, ready-to-import XML files will be in the output folder.

```bash
ls -l outputs/

```
