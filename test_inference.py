from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import os

# --- 1. CONFIGURATION (LOCAL PATHS) ---
# We point directly to the folders we just downloaded
base_model_path = "/workspace/manual_models/base"
adapter_path = "/workspace/manual_models/adapter"
adapter_name = "dspace_adapter"

print(f"--> Loading Local Base Model from: {base_model_path}")
print(f"--> Loading Local Adapter from: {adapter_path}")

# Initialize vLLM Engine
# enable_lora=True allows us to inject your fine-tune
llm = LLM(
    model=base_model_path,
    enable_lora=True,
    max_lora_rank=64,             # Matches your training config
    gpu_memory_utilization=0.95,  # Use 95% of the A100's VRAM
    max_model_len=4096            # Context Window
)

# --- 2. DEFINE THE TEST PROMPT ---
# This mimics the Alpaca format used in training:
# ### System ... ### Context ... ### User Input ...

system_block = """### System:
You are an expert Automotive Test Automation Engineer.
Convert Natural Language Test Steps into dSPACE XML.

Rules:
1. Use the Library Dictionary provided in the Context.
2. Use exact xml_tag, library_link, and id.
3. Output ONLY operational XML blocks."""

# Fake context to test hallucination (Model should use THESE IDs, not training memory)
context_block = """
### Context (Library Dictionary):
[
  {
    "concept": "Set Battery Voltage (Requires: Value)",
    "library_link": "TVSM_Library.SET_BATT_VOLTAGE",
    "xml_tag": "MainLibrary.Serial",
    "id": "{FAKE-ID-123}",
    "required_params": ["Value"]
  },
  {
    "concept": "Check Engine Speed (Requires: Expected_RPM)",
    "library_link": "TVSM_Library.CHECK_ENGINE_SPEED",
    "xml_tag": "MainLibrary.Serial",
    "id": "{FAKE-ID-456}",
    "required_params": ["Expected_RPM"]
  }
]"""

user_block = """
### User Input (Test Case):
1. Set the battery voltage to 12.5V.
2. Verify that the engine speed is approximately 3000 RPM.
"""

# Combine into full prompt
full_prompt = f"{system_block}\n{context_block}\n{user_block}\n### Response (XML):\n"

# --- 3. RUN INFERENCE ---
print("\n=== GENERATING OUTPUT ===\n")

sampling_params = SamplingParams(
    temperature=0.1, 
    max_tokens=2048,
    stop=["### User Input"] # Stop generation if it tries to continue the conversation
)

outputs = llm.generate(
    [full_prompt], 
    sampling_params=sampling_params,
    lora_request=LoRARequest(adapter_name, 1, adapter_path)
)

print("-" * 50)
print(outputs[0].outputs[0].text)
print("-" * 50)