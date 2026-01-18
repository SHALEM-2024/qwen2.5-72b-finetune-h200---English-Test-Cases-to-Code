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





🚀 Phase 1: Infrastructure Setup (RunPod)
Goal: Deploy a GPU instance with enough persistent storage to handle large model weights without crashing.

Select GPU:

Go to RunPod Secure Cloud.

Choose 1x A100 80GB PCIe (Essential for 72B models) or H100.

Select Template:

Search for and select: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04.

Configure Storage (CRITICAL):

Click "Edit Pod Settings".

Container Disk: Default (20GB).

Volume Disk: 100 GB (Do not skip this. We need this space in /workspace for the model).

Deploy: Start the pod.

🛠 Phase 2: Environment Initialization
Once your pod is Running, click Connect > Start Web Terminal.

1. Install Dependencies
Copy and paste this block to update Python tools and install the inference engine (vllm):

Bash

# Update pip and install vLLM + Hugging Face tools
pip install --upgrade pip
pip install vllm peft huggingface_hub
2. Login to Hugging Face
You need this to download the gated Qwen model.

Bash

huggingface-cli login
# Paste your HF Token when prompted (Right-click to paste, then Enter)
📥 Phase 3: Model Setup (The "No-Crash" Method)
We manually download models to the /workspace volume to avoid filling up the root drive.

Bash

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
🤖 Phase 4: Production Inference Engine
We use a custom Python script (Version 14) that features:

Context Filtering: Automatically selects only relevant library items to fit large dictionaries into memory.

Batch Processing: Processes all files in the inputs/ folder automatically.

XML Formatting: Wraps the output in valid Standard.Sequence tags.

1. Create Directory Structure
Bash

mkdir -p inputs outputs
2. Create the Context File (context.txt)
Paste your entire JSON library dictionary into this file.

Bash

nano context.txt
# Paste JSON content -> Ctrl+O -> Enter -> Ctrl+X
3. Create the Inference Script (batch_inference.py)
Create the file:

Bash

nano batch_inference.py
Paste the following code:

Python

import os
import sys
import re
import json
import glob
import time

# --- 0. CRITICAL OVERRIDES ---
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# --- 1. CONFIGURATION ---
base_model_path = "/workspace/manual_models/base"
adapter_path = "/workspace/manual_models/adapter"
adapter_name = "dspace_adapter"

INPUT_DIR = "inputs"
OUTPUT_DIR = "outputs"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Initialize vLLM
try:
    print("--> Initializing vLLM Engine...")
    llm = LLM(
        model=base_model_path,
        enable_lora=True,
        max_lora_rank=64,
        gpu_memory_utilization=0.92,
        max_model_len=32768, 
        kv_cache_dtype="auto", 
        enforce_eager=True,            
        enable_chunked_prefill=False, 
        max_num_seqs=1
    )
except Exception as e:
    print(f"\nINITIALIZATION ERROR: {e}")
    sys.exit(1)

# --- 2. RANKED FILTER (SMART CONTEXT SELECTION) ---
def filter_context(context_text, user_input):
    try:
        library_data = json.loads(context_text)
        if not isinstance(library_data, list): library_data = [library_data]
    except json.JSONDecodeError:
        print("       [ERROR] Context file is not valid JSON.")
        return "[]"

    # Tokenize Input
    user_words = re.findall(r'[a-zA-Z0-9]+', user_input.lower())
    stop_words = {"the", "and", "or", "to", "of", "in", "is", "a", "step", "measure", "that", "value"} 
    base_keywords = set([w for w in user_words if w not in stop_words and len(w) > 2]) 

    # Synonym Mapping for Automotive Terms
    synonym_map = {
        "fault":  ["fiu", "short", "circuit", "failure", "scg"],
        "remove": ["deactivate", "release", "clear", "reset"], 
        "can":    ["fiu", "scg", "bus"],
        "gear":   ["write_read_gear", "gear_position"], 
        "pedal":  ["write_read_aps", "acc_pedal"],
        "acc":    ["write_read_aps", "acc_pedal"],
        "aps":    ["write_read_aps", "acc_pedal"],
        "create": ["set", "activate", "trigger"],
        "check":  ["read", "verify", "validate", "camera", "vision", "pattern"],
        "mil":    ["telltale", "indicator", "warning", "lamp"],
        "screen": ["cluster", "display", "hmi"],
        "simulate": ["set", "force", "write"],
        "ignition": ["ign", "key", "switch", "simulating"],
        "battery":  ["batt", "voltage"],
        "crank":    ["start", "engine"]
    }

    final_keywords = set(base_keywords)
    for word in base_keywords:
        if word in synonym_map:
            final_keywords.update(synonym_map[word])
            
    # Score Library Items
    scored_items = []
    for item in library_data:
        actual_data = item.get("json_snippet", item)
        item_str = json.dumps(actual_data).lower()
        
        match_count = 0
        for key in final_keywords:
            if key in item_str:
                match_count += 1
        
        # Priority Boosting for exact technical matches
        if "deactivate" in item_str and "remove" in final_keywords: match_count += 10
        if "gear" in item_str and "gear" in final_keywords: match_count += 5
        if "battery" in item_str and "battery" in final_keywords: match_count += 5
        if "write_read_aps" in item_str and "aps" in final_keywords: match_count += 20 
        
        if match_count > 0:
             scored_items.append((match_count, actual_data))
    
    # Sort and Keep Top 100
    scored_items.sort(key=lambda x: x[0], reverse=True)
    MAX_ITEMS = 100
    relevant_items = [x[1] for x in scored_items[:MAX_ITEMS]]
    
    print(f"       [DEBUG] Filtered Context: Kept {len(relevant_items)} relevant items.")
    return json.dumps(relevant_items, indent=2)

def read_file(filename):
    with open(filename, 'r') as f: return f.read().strip()

# --- 3. SYSTEM PROMPT ---
system_block = """### System:
You are an expert Automotive Test Automation Engineer.
Convert Natural Language Test Steps into dSPACE XML.

### CRITICAL RULES:
1. **Block Type:** Use `<Standard.LibraryLinkBlock>`.
2. **Parameters:** YOU MUST NEST DATA INSIDE A `<value>` TAG.
3. **Logic:** - "Remove Fault" -> Use `DEACTIVATE_RELEASE_ERROR`.
   - "Check Telltale" -> Use `CHECK_CLUSTER_THROUGH_CAMERA`.
   - "Simulate Gear" -> Use `WRITE_READ_GEAR`.
   - "Simulate APS" -> Use `WRITE_READ_APS`.

### Required Output Format:
<FrameworkBuilder.ActualOperationSlot name="Initialization">
    <subsystems> ... </subsystems>
</FrameworkBuilder.ActualOperationSlot>
<FrameworkBuilder.ActualOperationSlot name="StepsAndEvaluation">
    <subsystems> ... </subsystems>
</FrameworkBuilder.ActualOperationSlot>
<FrameworkBuilder.ActualOperationSlot name="Cleanup">
    <subsystems> ... </subsystems>
</FrameworkBuilder.ActualOperationSlot>
"""

# --- 4. EXECUTION LOOP ---
print("--> Loading Library Context...")
full_context_path = "context.txt"
if not os.path.exists(full_context_path):
    print("CRITICAL: context.txt missing.")
    sys.exit(1)
full_context_raw = read_file(full_context_path)

input_files = glob.glob(os.path.join(INPUT_DIR, "*.txt"))
print(f"--> Found {len(input_files)} test cases.")

for i, input_file in enumerate(input_files):
    print(f"\n[{i+1}/{len(input_files)}] Processing: {input_file}")
    
    start_t = time.time()
    user_content = read_file(input_file)
    
    # Apply Filter
    filtered_context = filter_context(full_context_raw, user_content)
    
    # Construct Prompt
    full_prompt = f"{system_block}\n\n### Library Dictionary (JSON):\n{filtered_context}\n\n### User Input:\n{user_content}\n\n### Response (XML):\n"
    
    sampling_params = SamplingParams(
        temperature=0.1, 
        repetition_penalty=1.15,
        max_tokens=8192,
        stop=["</FrameworkBuilder.ActualDataSlot>"]
    )
    
    # Generate
    outputs = llm.generate(
        [full_prompt], 
        sampling_params=sampling_params,
        lora_request=LoRARequest(adapter_name, 1, adapter_path)
    )
    
    generated_text = outputs[0].outputs[0].text.strip()
    
    # Wrap in XML Header/Footer
    full_xml_output = f"""<?xml version="1.0" encoding="utf-8"?>
<Standard.Sequence name="Test_Sequence_Generated">
    <library-description>Generated by AI Model</library-description>
    <subsystems>
        <FrameworkBuilder.Frame name="Test_Frame_Main">
            <library-description>To execute subsystems sequentially.</library-description>
            <subsystems>
                <FrameworkBuilder.ActualDataSlot name="Data">
                    <subsystems>
{generated_text}
                    </subsystems>
                </FrameworkBuilder.ActualDataSlot>
            </subsystems>
        </FrameworkBuilder.Frame>
    </subsystems>
</Standard.Sequence>"""

    output_path = os.path.join(OUTPUT_DIR, os.path.basename(input_file).replace(".txt", ".xml"))
    with open(output_path, "w") as f:
        f.write(full_xml_output)
    
    print(f"    [STATS] Duration: {time.time() - start_t:.2f}s")
    print(f"    [SUCCESS] Saved to: {output_path}")

print("\n--> All tests completed.")
📝 Usage Guide
1. Prepare Input File
Create a text file inside the inputs/ folder (e.g., inputs/Test_01.txt). You MUST use the following format:

Plaintext

Precondition:
1. Set Battery Voltage to 13.5V
2. Turn Ignition ON
...

Action:
7. Navigate to Menu -> Display Setup
8. Check that Display is set to Dark mode
...

Postcondition:
19. Reset Trip details
20. Turn Ignition OFF
2. Run the Script
Bash

python batch_inference.py
3. Retrieve Output
The generated XML file will be saved in the outputs/ folder with the same name as your input file.

Bash

cat outputs/Test_01.xml

