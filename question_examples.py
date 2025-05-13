import os
from datasets import load_dataset

# 1. Carrega l'split de train
train_ds = load_dataset("openlifescienceai/medmcqa", split="train")
train_ds.shuffle(seed=42)

# 2. Extreu tots els subjectes únics
subjects = sorted(set(ex["subject_name"] for ex in train_ds if ex["subject_name"] != "Unknown"))

# 3. Assegura't que la carpeta existeix
out_dir = "exemples_MedMCQA"
os.makedirs(out_dir, exist_ok=True)

# 4. Escriu un fitxer per subject amb límit de 1000 preguntes
MAX_PER_SUBJECT = 100
for subj in subjects:
    fname = subj.replace(" & ", "_").replace(" ", "_") + ".txt"
    path = os.path.join(out_dir, fname)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for ex in train_ds:
            if ex["subject_name"] == subj:
                f.write(f"Question: {ex['question']}\n")
                for i, opt in enumerate((ex["opa"], ex["opb"], ex["opc"], ex["opd"])):
                    f.write(f"{chr(65+i)}. {opt}\n")
                # Add the correct answer letter
                correct_letter = chr(65 + ex["cop"])
                f.write(f"Correct Answer: {correct_letter}\n")
                # Add the explanation
                if ex["exp"] and ex["exp"].strip():  # Check if explanation exists and is not empty
                    f.write(f"Explanation: {ex['exp']}\n")
                f.write("\n")
                count += 1
                if count >= MAX_PER_SUBJECT:
                    break
    print(f"Saved {count} examples for {subj} → {path}")


# 5. Escriu un fitxer per a les preguntes genèriques
fname = "Mixed.txt"
path = os.path.join(out_dir, fname)
count = 0
with open(path, "w", encoding="utf-8") as f:
    for ex in train_ds:
        f.write(f"Question: {ex['question']}\n")
        for i, opt in enumerate((ex["opa"], ex["opb"], ex["opc"], ex["opd"])):
            f.write(f"{chr(65+i)}. {opt}\n")
        # Add the correct answer letter
        correct_letter = chr(65 + ex["cop"])
        f.write(f"Correct Answer: {correct_letter}\n")
        # Add the explanation
        if ex["exp"] and ex["exp"].strip():  # Check if explanation exists and is not empty
            f.write(f"Explanation: {ex['exp']}\n")
        f.write("\n")
        count += 1
        if count >= MAX_PER_SUBJECT:
            break

print(f"Saved {count} examples for Mixed → {path}")
