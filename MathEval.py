import pandas as pd
import re
import requests
from pathlib import Path

# =====================================================================
# --- CONFIGURATION VARIABLES ---
# =====================================================================

# File Paths
TARGET_FOLDER_PATH = "MathVisionResults" # Folder containing all your raw CSVs
OUTPUT_CSV_SUFFIX = "_GRADED.csv"        # What to append to the saved file

# Ollama Settings
OLLAMA_URL = "http://localhost:11434/api/generate"
JUDGE_MODEL = "llama3"
OLLAMA_TIMEOUT = 180
TEMPERATURE = 0.0      # Keep at 0.0 for deterministic True/False grading

# =====================================================================


def check_ollama_status():
    """Ensure local Ollama is running."""
    try:
        # Pings the base URL just to check if the server is awake
        requests.get(OLLAMA_URL.replace("/api/generate", ""), timeout=2)
        return True
    except:
        return False

def llm_math_judge_ollama(question, model_answer, ground_truth):
    """Uses local Ollama to verify mathematical equivalence with question context."""
    prompt = f"""You are an expert math grader. Analyze if the Model Answer is mathematically equivalent to the Ground Truth based on the Question context.

Question: {question}
Model Answer: {model_answer}
Ground Truth: {ground_truth}

Answer ONLY with "True" or "False". Do not explain your reasoning.
"""
    
    payload = {
        "model": JUDGE_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": TEMPERATURE}
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        response.raise_for_status()
        result_text = response.json().get("response", "").strip().lower()
        return "true" in result_text
    except Exception as e:
        print(f"   ⚠️ Ollama Judge Error: {e}")
        return False

def evaluate_mathvision_row(question, ans, gt):
    """Layered evaluation: Fast programmatic checks first, Ollama fallback second."""
    ans_str = str(ans).strip().lower()
    gt_str = str(gt).strip().lower()

    # --- LAYER 1: FAST PROGRAMMATIC CHECKS ---
    if ans_str == gt_str:
        return True

    if gt_str in ['a', 'b', 'c', 'd', 'e']:
        if re.search(fr'\b{gt_str}\b', ans_str):
            return True
        return False 

    try:
        if float(ans_str) == float(gt_str):
            return True
    except ValueError:
        pass 
        
    ans_clean = ans_str.replace(" ", "").replace("$", "")
    gt_clean = gt_str.replace(" ", "").replace("$", "")
    
    if ans_clean == gt_clean:
        return True

    # --- LAYER 2: LOCAL OLLAMA JUDGE ---
    # Triggered if complex math symbols exist and programmatic checks fail
    if any(char in gt_str for char in ['\\', '/', '+', '-', '*', '^', '=', 'x', 'y', 'pi', 'sqrt']):
        return llm_math_judge_ollama(question, ans_str, gt_str)

    return False

def grade_results_csv(csv_path):
    print(f"\n🚀 Starting Post-Processing Evaluation on {csv_path.name} using {JUDGE_MODEL}...")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ Error reading file {csv_path.name}: {e}")
        return
        
    # Ensure columns exist
    if not all(col in df.columns for col in ['question', 'model_answer', 'ground_truth']):
        print(f"❌ CSV {csv_path.name} is missing required columns ('question', 'model_answer', 'ground_truth'). Skipping.")
        return

    judgments = []
    for index, row in df.iterrows():
        question = str(row['question'])
        ans = str(row['model_answer'])
        gt = str(row['ground_truth'])
        
        is_correct = evaluate_mathvision_row(question, ans, gt)
        judgments.append(is_correct)
        
        mark = "✅" if is_correct else "❌"
        # Print a short log so you can watch it work
        print(f"Row {index} | {mark} | Model: {ans[:20]}... | GT: {gt[:20]}...")

    # Attach the final verdicts back to the DataFrame under the new column name
    df['ComplexEval'] = judgments
    
    # Save a clean, graded version of the CSV
    graded_csv_path = csv_path.parent / f"{csv_path.stem}{OUTPUT_CSV_SUFFIX}"
    df.to_csv(graded_csv_path, index=False)
    
    # Calculate final accuracy for your thesis
    accuracy = (sum(judgments) / len(judgments)) * 100
    print(f"\n📊 Final Accuracy for {csv_path.name}: {accuracy:.2f}%")
    print(f"💾 Graded results saved to: {graded_csv_path.name}")

def process_folder():
    """Iterates through all CSV files in the target folder and processes them."""
    if not check_ollama_status():
        print("❌ Local Ollama is not running. Please start it first.")
        return

    folder = Path(TARGET_FOLDER_PATH)
    if not folder.exists() or not folder.is_dir():
        print(f"❌ Folder not found: {TARGET_FOLDER_PATH}")
        return

    # Find all CSV files, excluding ones that already have the graded suffix
    csv_files = [f for f in folder.glob("*.csv") if not f.name.endswith(OUTPUT_CSV_SUFFIX)]

    if not csv_files:
        print(f"⚠️ No ungraded CSV files found in '{TARGET_FOLDER_PATH}'.")
        return

    print(f"📂 Found {len(csv_files)} CSV file(s) to process in '{TARGET_FOLDER_PATH}'.")
    
    for csv_file in csv_files:
        grade_results_csv(csv_file)
        
    print("\n🎉 All files processed successfully!")

if __name__ == "__main__":
    process_folder()
