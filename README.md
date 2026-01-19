# Bridging the Gap: Test Plans ➡️ Test Benches
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
> <img width="400" alt="image" src="https://github.com/user-attachments/assets/c17faf67-df71-4e21-a17b-bd7a043ff375" />

#### Actual Illustration
> <img width="400" alt="image" src="https://github.com/user-attachments/assets/7df5f1b4-7906-422c-a757-08ab118f577d" />

---

## 🚀 Replication Guide
This guide is split into two main parts: **Training the Model** (Fine-tuning) and **Deploying for Production**.

## 🧠 Part 1: Axolotl Training Workflow
*Standard operating procedure for initializing the environment and launching a fine-tuning job.*

### 📋 Prerequisites
* **Environment:** Access to the GPU workspace container.
* **Hardware:** Sufficient VRAM for Qwen 72B (**~40GB+**).
* **Files:** `config.yaml` and `fine_tuning_data.jsonl` must be in the project root.

### ⚙️ Setup & Installation

**Step 1: Navigate to Project Directory**
> ⚠️ **Critical:** Do not run commands from root.
```bash
cd /workspace/axolotl

```

**Step 2: Lock Version (Recommended)**
To ensure reproducibility, checkout the specific commit version used in previous runs.

```bash
git checkout $(git rev-list -n 1 --before="2024-11-02 12:00" main)

```

**Step 3: Install Dependencies**

```bash
pip install -e . --no-deps

```

### ✈️ Launching Training

**Pre-Flight Check:** Verify files exist (`ls -lh fine_tuning_data.jsonl config.yaml`).

**Start Fine Tuning:**

```bash
accelerate launch -m axolotl.cli.train config.yaml

```
Logs should look something like this,

<img width="1919" height="870" alt="image" src="https://github.com/user-attachments/assets/9f839771-0622-48d0-9e35-28711bb3b9e0" />

---

## ☁️ Part 2: Model Upload Workflow

*Pushing the adapter and tokenizer to Hugging Face. - Used LoRA adapter*

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

## 🏭 Part 3: Production Infrastructure & Inference

*The end-to-end guide to setting up the production inference engine on RunPod.*

### 🏗️ Phase 1: Infrastructure Setup (RunPod)


1. **Select GPU:** 
* Choose **1x A100 80GB PCIe** (Essential for 72B parameter model).


2. **Select Template:**
* Search for: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`.


3. **Configure Storage (CRITICAL):**
* Click "Edit Pod Settings".
* **Container Disk:** Default (200GB).
* **Volume Disk:** `100 GB`.



### 🛠️ Phase 2: Environment Initialization

Once your pod is **Running**.

**1. Install Dependencies**
Update Python tools and install the inference engine (`vllm`) and Hugging Face tools.

```bash
pip install --upgrade pip
pip install vllm peft huggingface_hub

```

**2. Login to Hugging Face**
Required to download the gated Qwen model.

```bash
huggingface-cli login
# Paste your HF Token when prompted

```

### 📥 Phase 3: Model Setup

Manually download models to the `/workspace` volume to avoid filling up the root drive. - This caused a lot of trouble for me.

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

### 🤖 Phase 4: Production Inference Engine

We use a custom Python script (Version 14) that features **Context Filtering** (to fit large dictionaries) and **Batch Processing** (automating multiple files).

**1. Create Directory Structure**

```bash
mkdir -p inputs outputs

```

**2. Create the Context File (`context.txt`)**
Paste your entire JSON library dictionary into this file.

```bash
nano context.txt

```

**3. Prepare the Inference Script**
Ensure the `run_batch_tests_v14.py` (V-14) script is present in your directory. This script handles the VLLM initialization, LoRA loading, and smart context filtering.

**4. Create Input Files**
Create text files inside the `inputs/` folder (e.g., `inputs/Test_01.txt`).

> ❗ **CRITICAL:** Use the format `Precondition:` / `Action:` / `Postcondition:`.

**5. Run the Engine**

```bash
python run_batch_tests_v14.py

```

**6. Retrieve Output**
The generated XML files will be saved in the `outputs/` folder.

```bash
ls -l outputs/
cat outputs/Test_01.xml

```

<img width="1919" height="870" alt="image" src="https://github.com/user-attachments/assets/08e766e4-1962-4a39-89f2-89f93bc4fccf" />

