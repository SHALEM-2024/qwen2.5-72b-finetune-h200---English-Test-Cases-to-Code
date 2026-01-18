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

# --- 2. RANKED FILTER (Logic: Hybrid Search + Scoring) ---
def filter_context(context_text, user_input):
    try:
        library_data = json.loads(context_text)
        if not isinstance(library_data, list): library_data = [library_data]
    except json.JSONDecodeError:
        print("       [ERROR] Context file is not valid JSON.")
        return "[]"

    # Extract User Keywords
    user_words = re.findall(r'\w+', user_input.lower())
    stop_words = {"the", "and", "or", "to", "of", "in", "is", "a", "step", "measure", "that", "value"} 
    base_keywords = set([w for w in user_words if w not in stop_words and len(w) > 2]) 

    # --- SYNONYM MAP (The Logic Engine) ---
    synonym_map = {
        # Faults & Safety
        "fault":  ["fiu", "short", "circuit", "failure", "scg"],
        "remove": ["deactivate", "release", "clear", "reset"], # Critical: Maps "Remove" to "Deactivate"
        "can":    ["fiu", "scg", "bus"],
        
        # Specific Simulations (Prioritizes Specific Blocks)
        "gear":   ["write_read_gear", "gear_position"], 
        "pedal":  ["write_read_aps", "acc_pedal"],
        "acc":    ["write_read_aps", "acc_pedal"],
        
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
            
    # Score Items based on Keyword Matches
    scored_items = []
    for item in library_data:
        # Handle Nested JSON format automatically
        actual_data = item.get("json_snippet", item)
        item_str = json.dumps(actual_data).lower()
        
        match_count = 0
        for key in final_keywords:
            if key in item_str:
                match_count += 1
        
        # Priority Boosting (Force critical blocks to top)
        if "deactivate" in item_str and "remove" in final_keywords: match_count += 10
        if "gear" in item_str and "gear" in final_keywords: match_count += 5
        if "battery" in item_str and "battery" in final_keywords: match_count += 5
        
        if match_count > 0:
             scored_items.append((match_count, actual_data))
    
    # Sort and Trim
    scored_items.sort(key=lambda x: x[0], reverse=True)
    
    # MAX_ITEMS = 100 (The "Sweet Spot" to prevent 32k context overflow)
    MAX_ITEMS = 100
    relevant_items = [x[1] for x in scored_items[:MAX_ITEMS]]
    
    # Debug Output
    print(f"       [DEBUG] Found {len(scored_items)} matches. Keeping top {len(relevant_items)}.")
    if len(relevant_items) > 0:
        print("       [DEBUG] Top 10 Selected Items:")
        for idx, item in enumerate(relevant_items[:10]):
            name = item.get("library_link") or item.get("concept") or "Unknown"
            print(f"          {idx+1}. {name}")
        
    return json.dumps(relevant_items, indent=2)

def read_file(filename):
    with open(filename, 'r') as f: return f.read().strip()

# --- 3. PROMPT (With Strict Syntax Enforcement) ---
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
   - "Simulate Gear" -> Use `WRITE_READ_GEAR` (if available).

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
    
    # 1. Filter Context
    filtered_context = filter_context(full_context_raw, user_content)
    
    # 2. Build Prompt
    full_prompt = f"{system_block}\n\n### Library Dictionary (JSON):\n{filtered_context}\n\n### User Input:\n{user_content}\n\n### Response (XML):\n"
    
    # 3. Sampling
    sampling_params = SamplingParams(
        temperature=0.1, 
        repetition_penalty=1.15,
        max_tokens=8192,
        stop=["</FrameworkBuilder.ActualDataSlot>"]
    )
    
    # 4. Generate
    outputs = llm.generate(
        [full_prompt], 
        sampling_params=sampling_params,
        lora_request=LoRARequest(adapter_name, 1, adapter_path)
    )
    
    generated_text = outputs[0].outputs[0].text.strip()
    
    # 5. Wrap XML
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