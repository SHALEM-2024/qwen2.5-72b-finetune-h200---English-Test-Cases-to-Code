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

## 🧪 Phase 5: Production Infrastructure & Inference Engine

This phase covers the end-to-end production workflow: from provisioning the GPU infrastructure to running the "V14" Batch Inference Engine.

### 1. Infrastructure Setup (RunPod)
*Goal: Deploy a GPU instance with enough persistent storage to handle large model weights.*

* **Select GPU:** Navigate to **RunPod Secure Cloud** and select **1x A100 80GB PCIe** (Essential for 72B models) or **H100**.
* **Select Template:** Use `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`.
* **Configure Storage (CRITICAL):**
    * Click "Edit Pod Settings".
    * **Container Disk:** Default (20GB).
    * **Volume Disk:** `100 GB` (Do not skip this. We need this space in `/workspace` for the model).

### 2. Environment Initialization
Once your pod is **Running**, click **Connect > Start Web Terminal**.

**A. Install Dependencies**
Update Python tools and install the inference engine (`vllm`) and Hugging Face tools.

```bash
pip install --upgrade pip
pip install vllm peft huggingface_hub

```

**B. Login to Hugging Face**
Required to download the gated Qwen model.

```bash
huggingface-cli login
# Paste your HF Token when prompted (Right-click to paste, then Enter)

```

### 3. Model Setup (The "No-Crash" Method)

We manually download models to the `/workspace` volume to avoid filling up the root drive.

```bash
# 1. Create a safe directory on the Volume drive
mkdir -p /workspace/manual_models

# 2. Download Base Model (Qwen 72B AWQ) - Approx 40GB
echo "--> Downloading Base Model..."
huggingface-cli download Qwen/Qwen2.5-72B-Instruct-AWQ \
  --local-dir /workspace/manual_models/base \
  --local-dir-use-symlinks False

# 3. Download Adapter (Fine-Tuned Layers) - Approx 1GB
echo "--> Downloading Adapter..."
huggingface-cli download shalem-2024/Qwen2.5-72B-dSPACE-XML-Adapter \
  --local-dir /workspace/manual_models/adapter \
  --local-dir-use-symlinks False

```

### 4. Production Inference Engine (Batch Processing)

We use a custom Python script (Version 14) that features **Context Filtering** (to fit large dictionaries) and **Batch Processing** (automating multiple files).

**A. Create Directory Structure**

```bash
mkdir -p inputs outputs

```

**B. Create the Context File (`context.txt`)**
Paste your entire JSON library dictionary into this file.

```bash
nano context.txt
# Paste JSON content -> Ctrl+O -> Enter -> Ctrl+X

```

**C. Prepare the Inference Script**
Ensure the `batch_inference.py` script (Version 14) is present in your directory.
*(Note: This script handles the VLLM initialization, LoRA loading, and smart context filtering.)*

**D. Create Input Files**
Create text files inside the `inputs/` folder (e.g., `inputs/Test_01.txt`).

> ❗ **CRITICAL:** Use the format `Precondition:` / `Action:` / `Postcondition:`.

**E. Run the Engine**

```bash
python batch_inference.py

```

**F. Retrieve Output**
The generated XML files will be saved in the `outputs/` folder.

```bash
ls -l outputs/
cat outputs/Test_01.xml

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
