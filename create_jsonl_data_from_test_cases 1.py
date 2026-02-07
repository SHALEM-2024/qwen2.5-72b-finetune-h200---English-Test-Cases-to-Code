import pandas as pd 
import json
import os
import re

# ==========================================
# 1. CONFIGURATION
# ==========================================
CSV_FILE = r"C:\Users\VehicleHIL.pc\Downloads\inputs\VHIL_H100_VehFunc_GearPositionSensor.csv"
TARGET_FOLDER = r"C:\Users\VehicleHIL.pc\Downloads\targets"
DICTIONARY_FILE = r"C:\Users\VehicleHIL.pc\Downloads\cleaned_dictionary_master.json"
OUTPUT_FILE = r"C:\Users\VehicleHIL.pc\Downloads\fine_tuning_data.jsonl"

# Expected CSV Columns
COL_TITLE = "Test Case Title"
COL_PRE   = "Pre-Action"
COL_STEPS = "Test Steps.Action"
COL_POST  = "Post Condition"

REQUIRED_COLUMNS = [COL_TITLE, COL_PRE, COL_STEPS, COL_POST]

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def load_json(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load dictionary: {e}")
        return []

def clean_cell(value):
    if pd.isna(value):
        return ""
    return str(value).strip()

def load_csv(path):
    df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.strip()
    return df

def find_target_file(title, folder):
    try:
        files = os.listdir(folder)
    except FileNotFoundError:
        print(f"[ERROR] Target folder not found: {folder}")
        return None

    matches = [f for f in files if f.startswith(title)]
    if not matches:
        print(f"[MISSING] No target file found for: {title}")
        return None

    # Prefer .BLKX files if multiple matches
    blkx_files = [f for f in matches if f.lower().endswith(".blkx")]
    if blkx_files:
        if len(blkx_files) > 1:
            print(f"[WARNING] Multiple .BLKX files found for '{title}', using: {blkx_files[0]}")
        return os.path.join(folder, blkx_files[0])

    if len(matches) > 1:
        print(f"[WARNING] Multiple files found for '{title}', using: {matches[0]}")

    return os.path.join(folder, matches[0])

def construct_english_prompt(row):
    return f"""Pre-Action:
{clean_cell(row[COL_PRE])}

Test Steps:
{clean_cell(row[COL_STEPS])}

Post Condition:
{clean_cell(row[COL_POST])}
"""

# ==========================================
# REPLACEMENT FUNCTION
# Copy and Paste this over the existing 'get_relevant_context'
# ==========================================

def get_relevant_context(english_text, dictionary):
    text_lower = english_text.lower()
    entries = []

    # 1. Define "Stop Words" (Generic words to ignore)
    # If a keyword is in this list, it doesn't count as a match.
    STOP_WORDS = {
        "set", "check", "read", "write", "step", "get", "put", "call", 
        "precondition", "postcondition", "value", "variable", "status", 
        "enable", "disable", "on", "off", "true", "false", "result",
        "return", "output", "input", "expected", "actual"
    }

    for item in dictionary:
        hits = 0
        keywords = item.get("keywords", [])
        
        for kw in keywords:
            kw_lower = kw.lower()
            
            # Rule 1: Ignore short words (2 chars or less)
            if len(kw) <= 2: 
                continue
                
            # Rule 2: Ignore Stop Words
            if kw_lower in STOP_WORDS: 
                continue
                
            # Rule 3: Check for match
            # We look for the keyword surrounded by non-letters to avoid partial matches
            # (e.g. ensure "car" doesn't match "cart")
            if kw_lower in text_lower:
                hits += 1
        
        # Rule 4: Threshold
        # We require at least ONE *unique/specific* keyword match.
        if hits > 0:
            entries.append(item["json_snippet"])

    # Ensure uniqueness
    unique_entries = []
    seen = set()
    for e in entries:
        try:
            serialized = json.dumps(e, sort_keys=True)
            if serialized not in seen:
                seen.add(serialized)
                unique_entries.append(e)
        except TypeError:
            continue

    return json.dumps(unique_entries, indent=2)


def extract_xml_meat(xml_text):
    pattern = (
        r'(<FrameworkBuilder\.ActualOperationSlot '
        r'name="StepsAndEvaluation".*?</FrameworkBuilder\.ActualOperationSlot>)'
    )
    match = re.search(pattern, xml_text, re.DOTALL)
    return match.group(1) if match else xml_text

def format_final_prompt(english_input, context_json):
    return f"""### System:
You are an expert Automotive Test Automation Engineer.
Convert Natural Language Test Steps into dSPACE XML.

Rules:
1. Use the Library Dictionary provided in the Context.
2. Use exact xml_tag, library_link, and id.
3. Output ONLY operational XML blocks.

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

    # --- Load dictionary ---
    dictionary = load_json(DICTIONARY_FILE)
    if not dictionary:
        print("[ERROR] Dictionary is empty. Exiting.")
        return

    # --- Load CSV ---
    try:
        df = load_csv(CSV_FILE)
        print(f"Loaded CSV with {len(df)} rows.")
        print("Detected columns:", list(df.columns))
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    # --- Validate required columns ---
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    jsonl_data = []
    success_count = 0

    # --- Process rows ---
    for _, row in df.iterrows():

        title = clean_cell(row[COL_TITLE])
        if not title:
            continue

        xml_path = find_target_file(title, TARGET_FOLDER)
        if not xml_path:
            continue

        try:
            with open(xml_path, "r", encoding="utf-8") as f:
                xml_full = f.read()
        except UnicodeDecodeError:
            with open(xml_path, "r", encoding="cp1252", errors="ignore") as f:
                xml_full = f.read()

        english_text = construct_english_prompt(row)
        xml_target = extract_xml_meat(xml_full)
        context = get_relevant_context(english_text, dictionary)
        full_prompt = format_final_prompt(english_text, context)

        jsonl_data.append({
            "instruction": "Convert English Test Steps to dSPACE XML.",
            "input": full_prompt,
            "output": xml_target
        })

        success_count += 1

    # --- Save JSONL file ---
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in jsonl_data:
            json.dump(item, f)
            f.write("\n")

    print(f"\n✅ Success! Created {success_count} training examples.")
    print(f"📁 Output saved to: {OUTPUT_FILE}")

# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == "__main__":
    main()
