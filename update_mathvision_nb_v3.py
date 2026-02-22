import json
import os

notebook_path = '/Users/smeh/Desktop/Thesis/ThesisCode/MathVisionEval.ipynb'

source_code = """import os
import pandas as pd
import time
import sys
from pathlib import Path

# --- Configuration & Discovery ---
def find_dataset_automatically(folder_name, file_name):
    current_dir = Path.cwd()
    print(f"[{time.strftime('%H:%M:%S')}] PROGRESS: Step 1/4 - Initializing file discovery...", flush=True)
    print(f"DEBUG: Working directory: {current_dir}", flush=True)
    
    # Search upwards for the 'Thesis' root
    print(f"DEBUG: Searching upwards for '{folder_name}'...", flush=True)
    for parent in [current_dir, *current_dir.parents]:
        if parent.name.lower() == "thesis":
            print(f"DEBUG: Root 'Thesis' found at {parent}", flush=True)
            target_path = parent / folder_name / file_name
            print(f"DEBUG: Checking {target_path}...", flush=True)
            if target_path.exists():
                return target_path
    
    # Fallback check
    fallback_path = current_dir.parent / folder_name / file_name
    print(f"DEBUG: Fallback check at {fallback_path}...", flush=True)
    if fallback_path.exists():
        return fallback_path
        
    return None

# --- Execution ---
if __name__ == "__main__":
    print("="*60)
    print("MATHVISION DATASET LOADER - PROGRESS TRACKER")
    print("="*60)
    
    # 1. Discovery
    dataset_path = find_dataset_automatically("DMathVision", "mathtestmini.parquet")
    
    if dataset_path:
        print(f"\\n[{time.strftime('%H:%M:%S')}] PROGRESS: Step 2/4 - Dataset located successfully.", flush=True)
        print(f"✅ Path: {dataset_path}", flush=True)
        
        try:
            # 2. Loading
            print(f"\\n[{time.strftime('%H:%M:%S')}] PROGRESS: Step 3/4 - Starting pandas.read_parquet...", flush=True)
            print("💡 This may take 5-30 seconds depending on file size and disk speed.", flush=True)
            
            start_load = time.time()
            df = pd.read_parquet(dataset_path)
            end_load = time.time()
            
            # 3. Success
            print(f"\\n[{time.strftime('%H:%M:%S')}] PROGRESS: Step 4/4 - Data loaded into memory.", flush=True)
            print(f"✅ Loaded {len(df)} rows in {end_load - start_load:.2f} seconds.", flush=True)
            
            print("-" * 40)
            print("📊 DATASET SNAPSHOT:")
            if 'display' in globals():
                display(df.head(4))
            else:
                print(df.head(4).to_string())
            print("-" * 40)
            
        except Exception as e:
            print(f"\\n❌ FATAL ERROR DURING LOADING: {e}", flush=True)
            import traceback
            traceback.print_exc()
            print("\\nTIP: If it says 'ModuleNotFoundError', run: !pip install pyarrow fastparquet")
    else:
        print(f"\\n[{time.strftime('%H:%M:%S')}] ❌ FATAL ERROR: Discovery failed.", flush=True)
        print("Dataset 'mathtestmini.parquet' not found in 'DMathVision' folder.")
"""

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

nb['cells'][0]['source'] = source_code.splitlines(True)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Updated MathVisionEval.ipynb with real-time progress markers.")
