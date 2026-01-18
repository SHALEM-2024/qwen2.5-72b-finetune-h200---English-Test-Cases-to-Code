import os
import sys
import re
import json

# --- 0. CRITICAL OVERRIDES ---
# Force vLLM to allow the context length we need
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# --- 1. CONFIGURATION ---
base_model_path = "/workspace/manual_models/base"
adapter_path = "/workspace/manual_models/adapter"
adapter_name = "dspace_adapter"

# Initialize vLLM Engine (High Precision Mode)
# We keep the settings that worked for you previously
try:
    llm = LLM(
        model=base_model_path,
        enable_lora=True,
        max_lora_rank=64,
        gpu_memory_utilization=0.92,
        
        # High Precision Context
        max_model_len=32768, 
        kv_cache_dtype="auto", 
        
        enforce_eager=True,           
        enable_chunked_prefill=False, 
        max_num_seqs=1
    )
except Exception as e:
    print(f"\nINITIALIZATION ERROR: {e}")
    sys.exit(1)

# --- 2. SMART READ FUNCTIONS ---
def read_file(filename):
    if not os.path.exists(filename):
        print(f"ERROR: File '{filename}' not found!")
        sys.exit(1)
    with open(filename, 'r') as f:
        return f.read().strip()

# --- THE FIX: JSON-NATIVE FILTER ---
def filter_context(context_text, user_input):
    print("\n--> STARTING SMART FILTERING (JSON MODE)...")
    
    # 1. PARSE THE JSON LIBRARY
    try:
        library_data = json.loads(context_text)
        if not isinstance(library_data, list):
            # Ensure it's a list even if a single object was passed
            library_data = [library_data]
        print(f"    [SUCCESS] Loaded JSON dictionary with {len(library_data)} items.")
    except json.JSONDecodeError as e:
        print(f"    [CRITICAL ERROR] Context file is not valid JSON.\n    Error: {e}")
        sys.exit(1)

    # 2. EXTRACT KEYWORDS FROM USER INPUT
    user_words = re.findall(r'\w+', user_input.lower())
    
    # Define "Stop Words" (words to ignore to prevent bad matches)
    stop_words = {"the", "and", "or", "to", "of", "in", "is", "a", "check", "verify", "set", "turn", "measure", "wait", "step"}
    keywords = set([w for w in user_words if w not in stop_words and len(w) > 3])
    
    print(f"    [Keywords Identified]: {list(keywords)[:10]}...") 
    
    # 3. FILTERING LOGIC
    relevant_items = []
    
    for item in library_data:
        # We construct a search string from the 'concept' and 'library_link' fields
        # This allows the filter to match "Battery" even if it's only in the link name
        search_str = (str(item.get("concept", "")) + " " + str(item.get("library_link", ""))).lower()
        
        if any(key in search_str for key in keywords):
             relevant_items.append(item)
    
    # Safety Net: If filter removes everything (0 items), keep the first 5 so the model doesn't crash
    if len(relevant_items) == 0:
        print("    [WARNING] Filter resulted in 0 items. Adding fallback items to prevent crash.")
        relevant_items = library_data[:5]
        
    print(f"    [RESULT] Kept {len(relevant_items)} JSON items relevant to your specific test.")
    
    # Return the filtered list as a formatted JSON string
    return json.dumps(relevant_items, indent=2)

print("--> Reading files...")
full_context = read_file("context.txt") # This MUST be your JSON file
user_content = read_file("input.txt")   # This is your test steps

# Filter the context
filtered_context = filter_context(full_context, user_content)

# --- 3. CONSTRUCT PROMPT WITH JSON INSTRUCTIONS ---
# I have updated the Instructions so the model knows how to read the JSON fields.
system_block = """### System:
You are an expert dSPACE AutomationDesk Engineer.
Your task is to insert the correct XML Action Blocks into the provided Framework Skeleton.

### Framework Skeleton:
<Standard.Sequence name="Test_Sequence">
    <subsystems>
        <FrameworkBuilder.Frame name="TestBlock">
            <subsystems>
                <FrameworkBuilder.ActualDataSlot name="Data">
                    <subsystems>
                        <FrameworkBuilder.ActualOperationSlot name="Initialization">
                            <subsystems>
                                <MainLibrary.Serial name="PreConditions">
                                    <subsystems>
                                        [INSERT PRE-CONDITION BLOCKS HERE]
                                    </subsystems>
                                </MainLibrary.Serial>
                            </subsystems>
                        </FrameworkBuilder.ActualOperationSlot>

                        <FrameworkBuilder.ActualOperationSlot name="StepsAndEvaluation">
                            <subsystems>
                                [INSERT TEST STEP BLOCKS HERE]
                            </subsystems>
                        </FrameworkBuilder.ActualOperationSlot>

                        <FrameworkBuilder.ActualOperationSlot name="Cleanup">
                            <subsystems>
                                [INSERT POST-CONDITION BLOCKS HERE]
                            </subsystems>
                        </FrameworkBuilder.ActualOperationSlot>
                    </subsystems>
                </FrameworkBuilder.ActualDataSlot>
            </subsystems>
        </FrameworkBuilder.Frame>
    </subsystems>
</Standard.Sequence>

### Instructions:
1.  **Analyze the User Input** to understand the test steps required.
2.  **Search the Relevant Library Dictionary (JSON)** provided below to find the matching library item.
    - Match the user's intent to the `"concept"` field in the JSON.
3.  **Generate XML Code**:
    - Use the JSON `"xml_tag"` field to determine the XML element (e.g., `<Standard.LibraryLinkBlock>`).
    - Use the JSON `"library_link"` field as the `Path` or `Link` attribute in the XML.
    - If the JSON has `"required_params"`, ensure those values are set in the generated XML.
4.  **Final Output**:
    - Output ONLY the XML code required to fill the `[INSERT...]` placeholders in the Skeleton.
"""

full_prompt = f"{system_block}\n\n### Relevant Library Dictionary (JSON):\n{filtered_context}\n\n### User Input (Test Steps):\n{user_content}\n\n### Response (XML):\n"

# --- 4. RUN INFERENCE ---
print(f"--> Sending request (Input Length: {len(full_prompt)})...")

sampling_params = SamplingParams(
    temperature=0.1, 
    max_tokens=4096, 
    stop=None, 
    repetition_penalty=1.15
)

try:
    outputs = llm.generate(
        [full_prompt], 
        sampling_params=sampling_params,
        lora_request=LoRARequest(adapter_name, 1, adapter_path)
    )

    generated_xml = outputs[0].outputs[0].text

    output_file = "output_real_case.xml"
    with open(output_file, "w") as f:
        f.write(generated_xml)

    print("-" * 50)
    print(f"SUCCESS! Output saved to: {output_file}")
    print("-" * 50)

except ValueError as e:
    print(f"\nCRITICAL RUNTIME ERROR: {e}")