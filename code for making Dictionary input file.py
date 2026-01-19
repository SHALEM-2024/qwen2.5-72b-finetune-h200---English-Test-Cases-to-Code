"""
Parse a dSPACE AutomationDesk .aldx / .adlx-like XML file and extract all <blkx-reference> entries
with their full folder hierarchy.

What this script does:
- Reads the XML (your .aldx file)
- Walks through nested <Standard.LibraryFolder ...> blocks
- For every <blkx-reference ...>, it records:
    - Full folder path (e.g., "READ_CAN/..." )
    - Reference name, type, id, href
- Writes results to CSV (and optionally JSON)
- Prints a quick summary

How to use:
1) Edit the paths in the "USER SETTINGS" section below
2) Run:
   python parse_aldx_hardcoded.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from typing import List, Optional


# ============================ USER SETTINGS ============================
# Change these paths as needed

INPUT_ALDX_PATH = r"D:\Bookmarks\TVSM_Library.adlx"    # <-- your .aldx/.adlx XML file
OUTPUT_CSV_PATH = r"c:\Users\Shalem.raju\Downloads\Dictionary_Inputs 1.csv"             # <-- where you want the CSV
OUTPUT_JSON_PATH = None                               # e.g., r"C:\path\to\refs.json" or None

# Optional folder filter:
# - If None: keep everything
# - If "READ_CAN": keep only rows where folder_path contains "READ_CAN"
FOLDER_FILTER_CONTAINS: Optional[str] = None
# ======================================================================


@dataclass
class BlkxRef:
    library_name: str
    folder_path: str          # e.g., "READ_CAN" or "DID_Module/DID_Sub_Module/IO_Services"
    ref_type: str             # from attribute "type"
    ref_name: str             # from attribute "name"
    ref_id: str               # from attribute "id"
    href: str                 # from attribute "href"


def parse_aldx(input_path: str) -> List[BlkxRef]:
    """
    Streaming parse (iterparse) so it works even for big files.

    "Streaming parse" means: it reads the XML piece-by-piece, instead of loading the whole file
    into memory at once — useful for large .aldx files.

    We maintain a stack (a list used like a pile) of folder names as we enter/exit
    <Standard.LibraryFolder>.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    folder_stack: List[str] = []
    results: List[BlkxRef] = []

    # Root library name from <Standard.CustomLibrary name="...">
    library_name: str = ""

    # iterparse gives "start" and "end" events without loading the entire XML into memory.
    context = ET.iterparse(input_path, events=("start", "end"))

    for event, elem in context:
        tag = elem.tag

        # Capture library name when we see the root element start
        if event == "start" and tag == "Standard.CustomLibrary":
            library_name = (elem.attrib.get("name") or "").strip()

        # When we enter a folder, push its name on the stack
        if event == "start" and tag == "Standard.LibraryFolder":
            folder_name = (elem.attrib.get("name") or "").strip()
            folder_stack.append(folder_name)

        # When we encounter a blkx-reference, record it with current folder stack
        if event == "start" and tag == "blkx-reference":
            ref = BlkxRef(
                library_name=library_name,
                folder_path="/".join([f for f in folder_stack if f]),
                ref_type=(elem.attrib.get("type") or "").strip(),
                ref_name=(elem.attrib.get("name") or "").strip(),
                ref_id=(elem.attrib.get("id") or "").strip(),
                href=(elem.attrib.get("href") or "").strip(),
            )
            results.append(ref)

        # When we exit a folder, pop it
        if event == "end" and tag == "Standard.LibraryFolder":
            if folder_stack:
                folder_stack.pop()

        # Free memory as we finish elements (important for large XML files)
        if event == "end":
            elem.clear()

    return results


def write_csv(rows: List[BlkxRef], csv_path: str) -> None:
    # Ensure output folder exists
    out_dir = os.path.dirname(os.path.abspath(csv_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fieldnames = ["library_name", "folder_path", "ref_type", "ref_name", "ref_id", "href"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def write_json(rows: List[BlkxRef], json_path: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(json_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in rows], f, ensure_ascii=False, indent=2)


def main() -> None:
    refs = parse_aldx(INPUT_ALDX_PATH)

    # Optional filter by folder path substring
    if FOLDER_FILTER_CONTAINS:
        key = FOLDER_FILTER_CONTAINS.strip()
        refs = [r for r in refs if key in r.folder_path]

    write_csv(refs, OUTPUT_CSV_PATH)

    if OUTPUT_JSON_PATH:
        write_json(refs, OUTPUT_JSON_PATH)

    # Quick summary
    unique_folders = len(set(r.folder_path for r in refs))
    library_name = refs[0].library_name if refs else ""

    print("Done.")
    print(f"Input: {INPUT_ALDX_PATH}")
    print(f"Library name: {library_name}")
    print(f"Total blkx-reference entries: {len(refs)}")
    print(f"Unique folder paths: {unique_folders}")
    print(f"CSV written to: {OUTPUT_CSV_PATH}")
    if OUTPUT_JSON_PATH:
        print(f"JSON written to: {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    try:
        main()
    except ET.ParseError as e:
        print(f"[XML ParseError] The file is not well-formed XML: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"[Error] {e}", file=sys.stderr)
        sys.exit(1)
