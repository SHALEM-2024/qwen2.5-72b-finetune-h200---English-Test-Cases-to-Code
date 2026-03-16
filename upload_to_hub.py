import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# your hugging face username and model name
hf_username = "username"
new_model_name = "Qwen2.5-72B-dSPACE-XML-Adapter"
repo_id = f"{hf_username}/{new_model_name}"

adapter_path = "./model-out"
base_model = "Qwen/Qwen2.5-72B-Instruct"

print(f"--> Uploading adapter to {repo_id}")

# Load adapter config
config = PeftConfig.from_pretrained(adapter_path)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load adapter into base model
model = PeftModel.from_pretrained(
    model,
    adapter_path
)

# Push adapter to Hugging Face Hub
model.push_to_hub(repo_id, token=True)
print("--> Adapter Uploaded Successfully!")

# Upload tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.push_to_hub(repo_id)

print("--> Tokenizer Uploaded Successfully!")
