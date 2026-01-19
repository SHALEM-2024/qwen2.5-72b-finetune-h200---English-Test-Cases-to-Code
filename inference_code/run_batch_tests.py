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

# --- 2. INTELLIGENT FILTER ---
def filter_context(context_text, user_input):
    try:
        library_data = json.loads(context_text)
        if not isinstance(library_data, list): library_data = [library_data]
    except json.JSONDecodeError:
        return context_text

    user_words = re.findall(r'\w+', user_input.lower())
    stop_words = {"the", "and", "or", "to", "of", "in", "is", "a", "step", "measure", "set", "turn"}
    base_keywords = set([w for w in user_words if w not in stop_words and len(w) > 3])

    # SYNONYM MAP (Updated for "Preconditions")
    synonym_map = {
        "fault":  ["fiu", "short", "circuit", "failure", "scg"],
        "create": ["set", "activate", "trigger"],
        "check":  ["read", "verify", "validate", "camera", "vision", "pattern"],
        "mil":    ["telltale", "indicator", "warning", "lamp"],
        "precondition": ["step_ems", "setup", "init"], # Look for the Macro Block
        "postcondition": ["step_ems", "cleanup"]
    }

    final_keywords = set(base_keywords)
    for word in base_keywords:
        if word in synonym_map:
            final_keywords.update(synonym_map[word])

    relevant_items = []
    for item in library_data:
        search_str = (str(item.get("concept", "")) + " " + 
                      str(item.get("library_link", "")) + " " + 
                      str(item.get("xml_tag", ""))).lower()
        if any(key in search_str for key in final_keywords):
             relevant_items.append(item)
    
    if len(relevant_items) == 0:
        relevant_items = library_data[:10]
        
    return json.dumps(relevant_items, indent=2)

def read_file(filename):
    with open(filename, 'r') as f: return f.read().strip()

# --- 3. STRICT SCHEMA PROMPT (The Fix) ---
system_block = """### System:
You are an expert Automotive Test Automation Engineer.
Convert Natural Language Test Steps into dSPACE XML.

### Rules:
1. **Tags:** You MUST use `<Standard.LibraryLinkBlock>`. Do NOT use `<Standard.LibraryLink>`.
2. **Parameters:** You MUST use `<MainLibrary.Int>` or `<MainLibrary.Float>` or `<MainLibrary.String>` inside a `<parameters>` block. 
   - DO NOT use attributes like `value="1"`. Use the text content: `<value>1</value>`.
3. **Macros:** If the user asks for "Preconditions", look for a Library Item starting with `STEP_` (e.g., `STEP_EMS_MIL_PRECONDITIONS`) instead of adding individual steps.

### Exact Output Schema Example:
<Standard.LibraryLinkBlock name="MyBlock" library-link="TVSM_Library.MyLink">
    <parameters>
        <MainLibrary.Int name="ParamName">
             <value>1</value>
        </MainLibrary.Int>
    </parameters>
</Standard.LibraryLinkBlock>

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
    
    process_start_time = time.time()
    user_content = read_file(input_file)
    filtered_context = filter_context(full_context_raw, user_content)
    
    full_prompt = f"{system_block}\n\n### Library Dictionary (JSON):\n{filtered_context}\n\n### User Input:\n{user_content}\n\n### Response (XML):\n"
    
    sampling_params = SamplingParams(
        temperature=0.1,  # Lower temp to force strict schema adherence
        repetition_penalty=1.15,
        max_tokens=8192,
        stop=["</FrameworkBuilder.ActualDataSlot>"]
    )
    
    inference_start_time = time.time()
    outputs = llm.generate(
        [full_prompt], 
        sampling_params=sampling_params,
        lora_request=LoRARequest(adapter_name, 1, adapter_path)
    )
    inference_duration = time.time() - inference_start_time
    
    generated_text = outputs[0].outputs[0].text.strip()
    
    # --- 5. WRAPPER ---
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
    
    print(f"    [STATS] Inference: {inference_duration:.2f}s")
    print(f"    [SUCCESS] Saved to: {output_path}")

print("\n--> All tests completed.")
