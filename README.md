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



Here are the clear, corrected steps to upload your Qwen model.

I have cleaned up the Python code (fixed a small import error in the snippet you provided) to ensure it runs without issues.

🛑 Pre-flight Check
Before starting, ensure you are logged into Hugging Face on this machine. If you haven't done this yet, run this first:

Bash

huggingface-cli login
(Paste your Write Token when asked).

Step 1: Navigate to the Correct Folder
We need to move into the folder where your output directory (model-out) is visible.

Run this command:

Bash

cd /workspace/axolotl
Step 2: Create the Upload Script
Now, create the Python script that handles the upload logic.

Open the text editor:

Bash

nano upload_direct.py
Copy and Paste the code below exactly. (I have separated the imports so it is valid Python):

Python

import os
from huggingface_hub import HfApi, create_repo
from transformers import AutoTokenizer

# 1. SETUP
hf_username = "shalem-2024"
model_name = "Qwen2.5-72B-dSPACE-XML-Adapter"
repo_id = f"{hf_username}/{model_name}"
local_folder = "./model-out"  # Relative path inside /workspace/axolotl

api = HfApi()

print(f"--> preparing to upload to: {repo_id}")

# 2. CREATE REPO
try:
    create_repo(repo_id, repo_type="model", exist_ok=True)
    print("--> Repo created/verified.")
except Exception as e:
    print(f"Error creating repo: {e}")

# 3. UPLOAD ADAPTER
print("--> Uploading adapter files...")
try:
    api.upload_folder(
        folder_path=local_folder,
        repo_id=repo_id,
        repo_type="model",
    )
    print("--> Adapter files uploaded successfully!")
except Exception as e:
    print(f"Error uploading folder: {e}")

# 4. UPLOAD TOKENIZER
print("--> Processing Tokenizer...")
try:
    base_model_id = "Qwen/Qwen2.5-72B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer.push_to_hub(repo_id)
    print("--> Tokenizer pushed successfully!")
except Exception as e:
    print(f"Warning: Tokenizer upload failed: {e}")

print("\n=== UPLOAD COMPLETE ===")
print(f"Your model is ready at: https://huggingface.co/{repo_id}")
Step 3: Save and Exit
Once the code is pasted into the black window:

Press Ctrl + O (Write Out).

Press Enter (To confirm the filename).

Press Ctrl + X (To exit).

Step 4: Run the Upload
Now, execute the script. It will read the model files from ./model-out and push them to your account.

Run this command:

Bash

python upload_direct.py
Step 5: Verify
Watch the screen. Once you see "=== UPLOAD COMPLETE ===", you can click the link it provides to see your model on the Hugging Face website.

Example: {'loss': 0.4131, ... 'epoch': 0.05}



Here is the updated **Phase 5** section. I have modified **Step 1B** to explicitly enforce the "Precondition / Action / Postcondition" formatting structure in the input file.

---

## Phase 5: The Stress Test (Production Simulation)

This phase simulates a real-world scenario by processing large dictionary contexts and complex test steps. We will move inputs to external text files to keep the Python script clean.

### 1. Create Input Files

**A. Create the Context File**
Paste your full JSON Library Dictionary here.

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
2. Turn Ignition ON [Ignition_SW_IP= 1]
3. Crank the vehicle [ES_SW_IP=1]
4. Check that No CAN failure & MIL failure is present
5. Simulate Gear_Position_Sensor_IP = Neutral [GEAR_POSITION[0x150] = 0x2]
6. Simulate ACC_PEDAL_IP = 0% to achieve DISPLAY_SPEED[0x120] = 0kmph

Action:
7. Navigate to Menu -> Display Setup -> Light and Dark Mode and Select Dark mode
8. Check that Display is set to Dark mode
9. Navigate to Menu -> Ride Mode and Select Urban mode
10. Check that Ride Mode is set to Urban mode
11. Simulate ACC_PEDAL_IP 30% to achieve DISPLAY_SPEED[0x120] = 50kmph and Simulate Gear_Position_Sensor_IP = 3rd Gear [GEAR_POSITION[0x150] = 0x7]
12. Check that Vehicle shall be moving at 50 kmph in 3rd gear: ENGINE_DATA_1.GEAR_POSITION[0x150] = 7, ABS_DATA_3(0x450): WHEEL_SPEED_FRONT = 50, ABS_DATA_3(0x450: WHEEL_SPEED_REAR = 50
13. Set conditions such that no EMS malfunction exists using below CAN signal: ENGINE_DATA_2 (0X350).MIL_STATUS = 0
14. Check that EMS MIL Telltale shall be OFF
15. Simulate EMS malfunction through below CAN Signal: ENGINE_DATA_2 (0X350).MIL_STATUS = 1
16. Check that EMS MIL Telltale shall be ON (Glowing in Amber color)
17. Stop the EMS Malfunction through below CAN Signal: ENGINE_DATA_2 (0X350).MIL_STATUS = 0
18. Check that EMS MIL Telltale shall be OFF

postcondition:
19. Click on Menu->Trip details->Select "Reset"-> select "Yes"
20. Turn Ignition ON [Ignition_SW_IP= 1]
21. Set Battery Voltage to 13.5V

```

*(Save & Exit: Ctrl+O, Enter, Ctrl+X)*

### 2. Update the Python Script

Update `test_inference.py` to read these files and handle the larger context window.

```bash
nano test_inference.py

```

**Paste this code:**

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import os

# --- 1. SETUP ---
base_model_path = "/workspace/manual_models/base"
adapter_path = "/workspace/manual_models/adapter"
adapter_name = "dspace_adapter"

# Load Model
llm = LLM(
    model=base_model_path,
    enable_lora=True,
    max_lora_rank=64,
    gpu_memory_utilization=0.95,
    max_model_len=8192  # Increased to 8192 for long contexts
)

# --- 2. READ FILES ---
def read_file(filename):
    with open(filename, 'r') as f:
        return f.read().strip()

print("--> Reading context.txt...")
context_content = read_file("context.txt")

print("--> Reading input.txt...")
user_content = read_file("input.txt")

# --- 3. CONSTRUCT PROMPT ---
system_block = """### System:
You are an expert Automotive Test Automation Engineer.
Convert Natural Language Test Steps into dSPACE XML.

Rules:
1. Use the Library Dictionary provided in the Context.
2. Use exact xml_tag, library_link, and id.
3. Output ONLY operational XML blocks."""

full_prompt = f"{system_block}\n\n### Context (Library Dictionary):\n{context_content}\n\n### User Input (Test Case):\n{user_content}\n\n### Response (XML):\n"

# --- 4. RUN INFERENCE ---
print(f"--> Sending request (Input Length: {len(full_prompt)} chars)...")

sampling_params = SamplingParams(
    temperature=0.1, 
    max_tokens=4096,
    stop=["### User Input"]
)

outputs = llm.generate(
    [full_prompt], 
    sampling_params=sampling_params,
    lora_request=LoRARequest(adapter_name, 1, adapter_path)
)

generated_xml = outputs[0].outputs[0].text

# --- 5. SAVE OUTPUT ---
output_file = "output.xml"
with open(output_file, "w") as f:
    f.write(generated_xml)

print("-" * 50)
print(f"SUCCESS! Output saved to: {output_file}")
print("-" * 50)

```

### 3. Run the Test

```bash
python test_inference.py

```

### 4. Validate the Output

View the result or compare it to your expected XML.

```bash
cat output.xml

```




Here is **Phase 6** of your Golden Path guide. This introduces the "Production-Grade" script (Version 14), which adds critical features like Context Filtering (to fit large dictionaries into memory) and Batch Processing.

---

## Phase 6: Batch Processing & Context Filtering (The "V14" Engine)

This is the advanced automation script ("Version 14"). It upgrades your setup from a simple test to a production engine that can:

1. **Filter the Context:** It intelligently selects only the relevant dictionary items for your specific test case, allowing you to use massive libraries without hitting token limits.
2. **Batch Process:** It reads **all** `.txt` files in an `inputs/` folder and processes them one by one.
3. **Auto-Format:** It wraps the raw AI output in the correct `Standard.Sequence` XML structure automatically.

### 1. Setup Directories

We need a place to drop your test cases (`inputs`) and a place to collect the results (`outputs`).

```bash
mkdir -p inputs outputs

```

### 2. Add Your Test Cases

Create distinct text files for each test case inside the `inputs/` folder.

* **Format:** Use the same `Precondition:` / `Action:` / `Postcondition:` format from Phase 5.
* **Example:** `nano inputs/TC_001_Dark_Mode.txt`

### 3. Create the Batch Script

Create the master python script.

```bash
nano batch_inference.py

```

**Paste this exact code (Version 14):**

```python
import os
import sys
import re
import json
import glob
import time

# --- 0. CRITICAL OVERRIDES ---
# Allows vLLM to process contexts slightly larger than the model's native config if needed
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

# --- 2. RANKED FILTER (FIXED TOKENIZATION) ---
def filter_context(context_text, user_input):
    try:
        library_data = json.loads(context_text)
        if not isinstance(library_data, list): library_data = [library_data]
    except json.JSONDecodeError:
        print("       [ERROR] Context file is not valid JSON.")
        return "[]"

    # --- THE FIX: SPLIT BY UNDERSCORES TOO ---
    # Old: re.findall(r'\w+', ...) -> kept "HIL_Mdl_Cons_APS" together
    # New: re.findall(r'[a-zA-Z0-9]+', ...) -> splits into "HIL", "Mdl", "Cons", "APS"
    user_words = re.findall(r'[a-zA-Z0-9]+', user_input.lower())
    
    stop_words = {"the", "and", "or", "to", "of", "in", "is", "a", "step", "measure", "that", "value"} 
    base_keywords = set([w for w in user_words if w not in stop_words and len(w) > 2]) 

    # --- SYNONYM MAP ---
    synonym_map = {
        # Faults & Safety
        "fault":  ["fiu", "short", "circuit", "failure", "scg"],
        "remove": ["deactivate", "release", "clear", "reset"], 
        "can":    ["fiu", "scg", "bus"],
        
        # Specific Simulations
        "gear":   ["write_read_gear", "gear_position"], 
        "pedal":  ["write_read_aps", "acc_pedal"],
        "acc":    ["write_read_aps", "acc_pedal"],
        "aps":    ["write_read_aps", "acc_pedal"], # Now this will definitely trigger
        
        # Standard Mappings
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
            
    # Score Items
    scored_items = []
    for item in library_data:
        actual_data = item.get("json_snippet", item)
        item_str = json.dumps(actual_data).lower()
        
        match_count = 0
        for key in final_keywords:
            if key in item_str:
                match_count += 1
        
        # Priority Boosting
        if "deactivate" in item_str and "remove" in final_keywords: match_count += 10
        if "gear" in item_str and "gear" in final_keywords: match_count += 5
        if "battery" in item_str and "battery" in final_keywords: match_count += 5
        
        # APS BOOST (Now guaranteed to trigger because "aps" is in final_keywords)
        if "write_read_aps" in item_str and "aps" in final_keywords: match_count += 20 
        
        if match_count > 0:
             scored_items.append((match_count, actual_data))
    
    # Sort and Trim
    scored_items.sort(key=lambda x: x[0], reverse=True)
    MAX_ITEMS = 100
    relevant_items = [x[1] for x in scored_items[:MAX_ITEMS]]
    
    # Debug Output (Check if WRITE_READ_APS is at the top now)
    print(f"       [DEBUG] Found {len(scored_items)} matches. Keeping top {len(relevant_items)}.")
    if len(relevant_items) > 0:
        print("       [DEBUG] Top 10 Selected Items:")
        for idx, item in enumerate(relevant_items[:10]):
            name = item.get("library_link") or item.get("concept") or "Unknown"
            print(f"          {idx+1}. {name}")
        
    return json.dumps(relevant_items, indent=2)

def read_file(filename):
    with open(filename, 'r') as f: return f.read().strip()

# --- 3. PROMPT ---
system_block = """### System:
You are an expert Automotive Test Automation Engineer.
Convert Natural Language Test Steps into dSPACE XML.

### CRITICAL RULES:
1. **Block Type:** Use `<Standard.LibraryLinkBlock>`.
2. **Parameters:** YOU MUST NEST DATA INSIDE A `<value>` TAG.
   - **WRONG:** `<MainLibrary.Int name="x">10</MainLibrary.Int>`
   - **CORRECT:** <MainLibrary.Int name="x">
          <value>10</value>
      </MainLibrary.Int>
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
    
    filtered_context = filter_context(full_context_raw, user_content)
    
    full_prompt = f"{system_block}\n\n### Library Dictionary (JSON):\n{filtered_context}\n\n### User Input:\n{user_content}\n\n### Response (XML):\n"
    
    sampling_params = SamplingParams(
        temperature=0.1, 
        repetition_penalty=1.15,
        max_tokens=8192,
        stop=["</FrameworkBuilder.ActualDataSlot>"]
    )
    
    outputs = llm.generate(
        [full_prompt], 
        sampling_params=sampling_params,
        lora_request=LoRARequest(adapter_name, 1, adapter_path)
    )
    
    generated_text = outputs[0].outputs[0].text.strip()
    
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

```

### 4. Run Batch Inference

Execute the script. It will automatically iterate through every file in `inputs/` and save the corresponding XML in `outputs/`.

```bash
python batch_inference.py

```

### 5. Check Results

Your fully formatted, ready-to-import XML files will be in the output folder.

```bash
ls -l outputs/

```

