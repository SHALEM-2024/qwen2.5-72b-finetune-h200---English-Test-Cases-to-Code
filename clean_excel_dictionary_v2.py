import pandas as pd
import json
import re
import ast # Used to safely parse the string representation of lists "['a','b']"

# ==========================================
# 1. CONFIGURATION
# ==========================================
# CHANGE 1: Updated filename to .csv
FILE_PATH = r"C:\Users\Shalem.raju\Downloads\Dictionary_Inputs 1.csv" 

# Exact Column Headers from your text input
COL_LIB_NAME = "library_name"   # e.g., TVSM_Library
COL_REF_TYPE = "ref_type"       # e.g., MainLibrary.Serial
COL_REF_NAME = "ref_name"       # e.g., SET_CHECK_BATT_ON
COL_REF_ID   = "ref_id"         # e.g., {B379088E...}
COL_PARAMS   = "data-objects"   # e.g., ["SetVariable", "Value"]

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def generate_concept(name, params_list):
    """
    Creates a readable sentence for the AI.
    Input: SET_CHECK_BATT_ON, ["Value"]
    Output: "Set Check Batt On (Requires: Value)"
    """
    readable_name = str(name).replace("_", " ").title()
    if params_list:
        param_str = ", ".join(params_list)
        return f"{readable_name} (Requires: {param_str})"
    return readable_name

def parse_params(param_string):
    """
    Converts string "['A', 'B']" into python list ['A', 'B']
    """
    try:
        # Handle Pandas NaN or empty cells
        if pd.isna(param_string) or str(param_string).strip() == "[]":
            return []
        
        # Determine if it's a simple string or a list string
        clean_str = str(param_string).strip()
        
        # Use ast.literal_eval for safe parsing of stringified lists
        if clean_str.startswith("[") and clean_str.endswith("]"):
            return ast.literal_eval(clean_str)
        else:
            return [] # Fallback
    except:
        return []

# ==========================================
# 3. MAIN CONVERSION LOGIC
# ==========================================
def clean_csv_data():
    try:
        # CHANGE 2: Use read_csv instead of read_excel
        # We try 'utf-8' first, but if it fails (common with Excel CSVs), we try 'cp1252'
        try:
            df = pd.read_csv(FILE_PATH, encoding='utf-8')
        except UnicodeDecodeError:
            print("UTF-8 encoding failed, trying cp1252...")
            df = pd.read_csv(FILE_PATH, encoding='cp1252')
            
        print(f"Loaded CSV file with {len(df)} rows.")
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    clean_entries = []

    for index, row in df.iterrows():
        try:
            # 1. Extract Basic Fields
            ref_name = str(row[COL_REF_NAME]).strip()
            # Check for NaN/Empty strings effectively
            if pd.isna(row[COL_REF_NAME]) or ref_name.lower() == "nan" or ref_name == "": 
                continue

            lib_name = str(row[COL_LIB_NAME]).strip()
            ref_type = str(row[COL_REF_TYPE]).strip()
            ref_id   = str(row[COL_REF_ID]).strip()
            
            # 2. Parse Parameters
            # Input: "['SetVariable', 'Value']" -> Output: ['SetVariable', 'Value']
            params_list = parse_params(row[COL_PARAMS])

            # 3. Construct the 'Link' (What goes in library-link="")
            # Logic: Library.BlockName
            full_link = f"{lib_name}.{ref_name}"

            # 4. Generate Concept (Searchable Description)
            concept = generate_concept(ref_name, params_list)

            # 5. Generate Keywords (For retrieval)
            keywords = ref_name.split("_")
            keywords.extend(params_list) # Add parameter names to keywords

            # 6. Build the Knowledge Snippet
            # This is the "Brain" definition the AI will see
            entry = {
                "keywords": [k for k in keywords if k],
                "json_snippet": {
                    "concept": concept,
                    "library_link": full_link,  # The attribute for XML
                    "xml_tag": ref_type,        # The tag name (e.g. MainLibrary.Serial)
                    "id": ref_id,               # The GUID
                    "required_params": params_list # The parameters to fill
                }
            }
            clean_entries.append(entry)

        except Exception as e:
            print(f"Skipping Row {index} due to error: {e}")
            continue

    # ==========================================
    # 4. SAVE
    # ==========================================
    output_filename = "cleaned_dictionary_master.json"
    with open(output_filename, "w") as f:
        json.dump(clean_entries, f, indent=4)

    print(f"\nSuccess! Processed {len(clean_entries)} entries.")
    print(f"Saved to: {output_filename}")
    
    # Preview
    if len(clean_entries) > 0:
        print("\n--- SAMPLE ENTRY ---")
        print(json.dumps(clean_entries[0], indent=2))

if __name__ == "__main__":
    clean_csv_data()

