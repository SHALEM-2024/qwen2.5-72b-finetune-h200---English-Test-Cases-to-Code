# 🚗 Bridging the Gap: Test Plans ➡️ Test Benches
### A Generative AI Pipeline for Automotive HIL Validation

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Model](https://img.shields.io/badge/Model-Qwen_2.5_72B-violet?style=for-the-badge&logo=huggingface)
![Hardware](https://img.shields.io/badge/Hardware-NVIDIA_H200-green?style=for-the-badge&logo=nvidia)
![System](https://img.shields.io/badge/Target-dSPACE_Automation_Desk-orange?style=for-the-badge)
---

## 📖 Project Overview

In the automotive industry, verifying hardware involves a massive manual bottleneck: translating thousands of English test cases into proprietary code for **dSPACE Automation Desk**. We faced a backlog of **almost 2,500 test cases**.

This repository documents the engineering pipeline used to fine-tune a **72-Billion parameter LLM** on an NVIDIA H200. The result is an AI agent capable of autonomously writing syntactically perfect `.blkx` (XML) test scripts.

| ❌ The Bottleneck | ✅ The Solution |
| :--- | :--- |
| **Manual Labor:** Engineers manually coding using dSPACE Automation Desk GUI or XML structures. | **Autonomous Generation:** AI converts Natural Language to XML. |
| **Slow TTM:** Weeks spent on 2500+ test cases. | **Accelerated Validation:** Instant script generation. |
| **Human Error:** Syntax errors in proprietary code. | **Syntactically Perfect:** Validated against dSPACE libraries. |

### 🛠️ Technical Implementation
This project tackles the complexity of generating strict, proprietary XML structures (`.blkx`) from unstructured natural language. To understand the nuance of automotive verification, I fine-tuned the **Qwen-2.5-72B** model specifically for this domain.

#### AI Generated Illustration
> **<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/c17faf67-df71-4e21-a17b-bd7a043ff375" />**
#### Actual Illustration
> <img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/7df5f1b4-7906-422c-a757-08ab118f577d" />

---

## 🚀 Replication Guide
If you want to replicate this engineering process, follow the workflow below.

## 1. Axolotl Training Workflow
*Standard operating procedure for initializing the environment and launching a fine-tuning job.*

### 📋 Prerequisites
* **Environment:** Access to the GPU workspace container.
* **Hardware:** Sufficient VRAM for Qwen 72B (Logs indicate **~40GB+** utilized).
* **Files:** `config.yaml` and `fine_tuning_data.jsonl` must be in the project root.

### ⚙️ Setup & Installation

**Step 2.1: Navigate to Project Directory**
> ⚠️ **Critical:** Do not run commands from root.
```bash
cd /workspace/axolotl

```

**Step 2.2: Lock Version (Recommended)**
To ensure reproducibility, checkout the specific commit version used in previous runs.

```bash
git checkout $(git rev-list -n 1 --before="2024-11-02 12:00" main)

```

*(Note: 'Detached HEAD' state is normal and safe for training.)*

**Step 2.3: Install Dependencies**

```bash
pip install -e . --no-deps

```

### ✈️ Launching Training

**Pre-Flight Check:** Verify files exist (`ls -lh fine_tuning_data.jsonl config.yaml`).

**Ignition:**

```bash
accelerate launch -m axolotl.cli.train config.yaml

```
**<img width="1719" height="872" alt="image" src="https://github.com/user-attachments/assets/9f839771-0622-48d0-9e35-28711bb3b9e0" />**
---

## ☁️ Model Upload Workflow

*Pushing the adapter and tokenizer to Hugging Face.*

### 🛑 Pre-flight

Ensure you are logged in.

```bash
huggingface-cli login
# Enter your Write Token when prompted

```

### 📤 Upload Steps

**1. Navigate:**

```bash
cd /workspace/axolotl

```

**2. Run Upload Script:**
This script handles repo creation and file uploading.

```bash
python upload_direct.py

```

**3. Verify:**
Wait for `=== UPLOAD COMPLETE ===` and check your Hugging Face repo link.

---

## 🧪 Phase 5: Production Simulation

*Simulating real-world processing with large context dictionaries.*

### 1. Create Input Files

**A. Context File (`context.txt`)**
Paste your full JSON Library Dictionary here.

```bash
nano context.txt

```

**B. Test Case File (`input.txt`)**

> ❗ **CRITICAL FORMATTING:** You must use clear headers: **Precondition**, **Action**, and **Postcondition**.

```text
Precondition:
1. Set Battery Voltage to 13.5V
...

Action:
...

Postcondition:
...

```

### 2. Run & Validate

**Run the Inference:**

```bash
python test_inference.py

```

**Check Output:**

```bash
cat output.xml

```
**<img width="1719" height="872" alt="image" src="https://github.com/user-attachments/assets/7df5f1b4-7906-422c-a757-08ab118f577d" />**

---

## ⚡ Phase 6: Batch Processing ("The V14 Engine")

This is the advanced automation script (**Version 14**). It upgrades the setup to a production engine with three key capabilities:

* 🎯 **Context Filtering:** Intelligently selects only relevant dictionary items to save tokens.
* 📦 **Batch Processing:** Iterates through **all** `.txt` files in `inputs/`.
* ✨ **Auto-Formatting:** Wraps output in `Standard.Sequence` XML automatically.

### Execution Steps

**1. Setup Directories:**

```bash
mkdir -p inputs outputs

```

**2. Add Test Cases:**
Place your formatted `.txt` files inside the `inputs/` folder.

**3. Run Batch Inference:**

```bash
python batch_inference.py

```

**4. View Results:**

```bash
ls -l outputs/

```

> **![PLACEHOLDER: Insert a screenshot of the batch script running in the terminal with the progress logs]**

