import pandas as pd
import json
import os
import re

# ==========================================
# 1. CONFIGURATION
# ==========================================
EXCEL_FILE = r"D:\Bookmarks\inputs\VHIL_H100_VehFunc_GearPositionSensor.xlsm"     # Your Excel Input
TARGET_FOLDER = r"D:\Bookmarks\targets"                # Folder with the weirdly named files
DICTIONARY_FILE = r"D:\Bookmarks\cleaned_dictionary_master.json"
OUTPUT_FILE = r"D:\Bookmarks\fine_tuning_data.jsonl"

# Column Mapping
COL_TITLE = "Test Case Title"
COL_PRE   = "Pre-Action"
COL_STEPS = "Test Steps.Action"
COL_POST  = "Post Condition"

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def load_json(filepath):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        return []

def find_target_file(title, folder):
    """
    Searches the folder for any file that STARTS with the title.
    Ignores the random UUID suffix.
    """
    # 1. List all files in the target folder
    try:
        all_files = os.listdir(folder)
    except FileNotFoundError:
        print(f"Error: Folder '{folder}' does not exist.")
        return None

    # 2. Filter for matches
    # We look for a file that starts with "Title" AND follows with a dot or underscore
    # to avoid partial matches (e.g. "Test1" matching "Test10")
    matches = [f for f in all_files if f.startswith(title)]
    
    if len(matches) == 0:
        print(f"  [MISSING] No file found for: {title}")
        return None
    elif len(matches) > 1:
        # If multiple matches, we take the first one but warn the user
        print(f"  [WARNING] Multiple files match '{title}'. Using: {matches[0]}")
        return os.path.join(folder, matches[0])
    else:
        # Perfect match
        return os.path.join(folder, matches[0])

def clean_cell(cell_value):
    if pd.isna(cell_value): return ""
    return str(cell_value).strip()

def construct_english_prompt(row):
    pre_action = clean_cell(row[COL_PRE])
    steps = clean_cell(row[COL_STEPS])
    post_cond = clean_cell(row[COL_POST])
    
    return f"""Pre-Action:
{pre_action}

Test Steps:
{steps}

Post Condition:
{post_cond}"""

def get_relevant_context(english_text, dictionary):
    relevant_entries = []
    text_lower = english_text.lower()
    
    for item in dictionary:
        hits = 0
        for kw in item['keywords']:
            if len(kw) > 2 and kw.lower() in text_lower:
                hits += 1
        if hits > 0:
            relevant_entries.append(item['json_snippet'])
            
    unique_entries = [dict(t) for t in {tuple(d.items()) for d in relevant_entries}]
    return json.dumps(unique_entries, indent=1)

def extract_xml_meat(full_xml_content):
    # Regex to capture everything inside the main Frame, skipping the wrapper if possible
    # We target the specific "StepsAndEvaluation" slot as the core logic
    pattern = r'(<FrameworkBuilder\.ActualOperationSlot name="StepsAndEvaluation".*?</FrameworkBuilder\.ActualOperationSlot>)'
    match = re.search(pattern, full_xml_content, re.DOTALL)
    
    if match:
        return match.group(1)
    else:
        # Fallback: Return the whole file if regex fails (better than nothing)
        return full_xml_content

def format_final_prompt(english_input, context_json):
    return f"""### System:
You are an expert Automotive Test Automation Engineer. Convert Natural Language Test Steps into dSPACE XML.
Rules:
1. Use the **Library Dictionary** provided in the Context.
2. Use exact `xml_tag`, `library_link`, and `id`.
3. Output ONLY the operational blocks.

### Context (Library Dictionary):
{context_json}

### User Input (Test Case):
{english_input}

### Response (XML):
"""

# ==========================================
# 3. MAIN LOGIC
# ==========================================
def main():
    dictionary = load_json(DICTIONARY_FILE)
    if not dictionary: return
    
    try:
        df = pd.read_excel(EXCEL_FILE)
        print(f"Loaded Excel with {len(df)} rows.")
    except Exception as e:
        print(f"Error loading Excel: {e}")
        return

    jsonl_data = []
    success_count = 0

    for index, row in df.iterrows():
        title = clean_cell(row[COL_TITLE])
        if not title: continue
        
        # 1. Find the weirdly named file
        xml_path = find_target_file(title, TARGET_FOLDER)
        
        if not xml_path:
            continue
            
        try:
            # 2. Read content (Try utf-8 first)
            with open(xml_path, 'r', encoding='utf-8') as f:
                xml_full = f.read()
        except UnicodeDecodeError:
            # Fallback for weird file encodings
            with open(xml_path, 'r', encoding='cp1252', errors='ignore') as f:
                xml_full = f.read()

        # 3. Process
        english_text = construct_english_prompt(row)
        xml_target = extract_xml_meat(xml_full)
        context = get_relevant_context(english_text, dictionary)
        full_prompt = format_final_prompt(english_text, context)
        
        # 4. Add to dataset
        entry = {
            "instruction": "Convert English Test Steps to dSPACE XML.",
            "input": full_prompt,
            "output": xml_target
        }
        jsonl_data.append(entry)
        success_count += 1

    # Save
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for line in jsonl_data:
            json.dump(line, f)
            f.write('\n')

    print(f"\nSuccess! Matched and compiled {success_count} training examples.")
    print(f"Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()