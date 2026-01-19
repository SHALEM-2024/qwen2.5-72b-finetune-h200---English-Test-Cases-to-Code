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

# --- 2. RANKED FILTER (The Fix) ---
def filter_context(context_text, user_input):
    try:
        library_data = json.loads(context_text)
        if not isinstance(library_data, list): library_data = [library_data]
    except json.JSONDecodeError:
        print("       [ERROR] Context file is not valid JSON.")
        return "[]"

    # A. Keywords
    user_words = re.findall(r'\w+', user_input.lower())
    stop_words = {"the", "and", "or", "to", "of", "in", "is", "a", "step", "measure", "that", "value"} 
    base_keywords = set([w for w in user_words if w not in stop_words and len(w) > 2]) 

    # B. Synonyms (UPDATED)
    synonym_map = {
        # Faults & Cleanup
        "fault":  ["fiu", "short", "circuit", "failure", "scg"],
        "remove": ["deactivate", "release", "clear", "reset"], # <--- FIXES "REMOVE FAULT"
        "can":    ["fiu", "scg", "bus"],
        
        # Specific Simulations (Fixes Gear/Pedal)
        "gear":   ["write_read_gear", "gear_position"],
        "pedal":  ["write_read_aps", "acc_pedal"],
        "acc":    ["write_read_aps", "acc_pedal"],
        
        # Standard mappings
        "create": ["set", "activate", "trigger"],
        "check":  ["read", "verify", "validate", "camera", "vision", "pattern"],
        "mil":    ["telltale", "indicator", "warning", "lamp"],
        "screen": ["cluster", "display", "hmi"],
        "simulate": ["set", "force", "write"],
        "ignition": ["ign", "key", "switch", "simulating"],
        "battery":  ["batt", "voltage"]
    }

    final_keywords = set(base_keywords)
    for word in base_keywords:
        if word in synonym_map:
            final_keywords.update(synonym_map[word])
            
    # C. Scoring Logic (New!)
    scored_items = []
    for item in library_data:
        actual_data = item.get("json_snippet", item)
        item_str = json.dumps(actual_data).lower()
        
        # Calculate Score: How many UNIQUE keywords match this block?
        match_count = 0
        for key in final_keywords:
            if key in item_str:
                match_count += 1
        
        # Boost specific terms to ensure they survive
        if "battery" in item_str and "battery" in final_keywords: match_count += 5
        if "ignition" in item_str and "ignition" in final_keywords: match_count += 5
        if "fiu" in item_str and "fault" in final_keywords: match_count += 5
        
        if match_count > 0:
             scored_items.append((match_count, actual_data))
    
    # D. Sort and Trim
    # Sort by score (highest first)
    scored_items.sort(key=lambda x: x[0], reverse=True)
    
    # Take top 100
    MAX_ITEMS = 100
    relevant_items = [x[1] for x in scored_items[:MAX_ITEMS]]
    
    print(f"       [DEBUG] Found {len(scored_items)} matches. Keeping top {len(relevant_items)}.")
    
    # E. Print Debug Sample (To verify fix)
    if len(relevant_items) > 0:
        print("       [DEBUG] Top 5 Selected Items:")
        for idx, item in enumerate(relevant_items[:100]):
            name = item.get("library_link") or item.get("concept") or "Unknown"
            print(f"          {idx+1}. {name}")

    return json.dumps(relevant_items, indent=2)

def read_file(filename):
    with open(filename, 'r') as f: return f.read().strip()

# --- 3. PROMPT ---
system_block = """### System:
You are an expert Automotive Test Automation Engineer.
Convert Natural Language Test Steps into dSPACE XML.

### CRITICAL SCHEMA RULES:
1. **Block Type:** You MUST use `<Standard.LibraryLinkBlock>`. 
   - DO NOT use `<Standard.LibraryLink>`.
   - DO NOT use `<MainLibrary.Serial>`.
2. **Parameters:** You MUST use specific types like `<MainLibrary.Int>` or `<MainLibrary.String>`.
   - **WRONG:** `<MainLibrary.Int name="Val" value="1"/>`
   - **RIGHT:** `<MainLibrary.Int name="Val"><value>1</value></MainLibrary.Int>`
3. **Logic:** - If the user says "Create Fault", use an **FIU** block (e.g., `SET_FIU...`).
   - If the user says "Check Telltale", use a **Camera** block (e.g., `CHECK_CLUSTER...`).
   - Avoid `CHECK_ANIMATION` blocks unless explicitly requested.

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
    
    # 1. Filter
    filtered_context = filter_context(full_context_raw, user_content)
    
    # 2. Prompt
    full_prompt = f"{system_block}\n\n### Library Dictionary (JSON):\n{filtered_context}\n\n### User Input:\n{user_content}\n\n### Response (XML):\n"
    
    # 3. Inference
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
    
    # 4. Wrapper
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
