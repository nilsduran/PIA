import os
import shutil
import json
from datasets import load_dataset

TEXTBOOK_DIR = "textbooks"  # Carpeta on tens els .txt dels llibres
MEDMCQA_DIR = "medmcqa_json"  # Carpeta on guardarem JSON per subject
OUTPUT_DIR = "agents_data"  # Carpeta on es crearà cada agent

# Definició dels 5 agents amb els llibres reals i els subjects de MedMCQA
agents = {
    "Medicina_General": {
        "textbooks": ["First_Aid_Step1.txt", "First_Aid_Step2.txt", "InternalMed_Harrison.txt"],
        "subjects": ["Medicine", "Social & Preventive Medicine", "Forensic Medicine"],
    },
    "Ciències_Bàsiques": {
        "textbooks": [
            "Anatomy_Gray.txt",
            "Physiology_Levy.txt",
            "Cell_Biology_Alberts.txt",
            "Histology_Ross.txt",
            "Biochemistry_Lippincott.txt",
            "Immunology_Janeway.txt",
        ],
        "subjects": ["Anatomy", "Physiology", "Biochemistry", "Microbiology"],
    },
    "Patologia_Farmacologia": {
        "textbooks": ["Pathology_Robbins.txt", "Pathoma_Husain.txt", "Pharmacology_Katzung.txt"],
        "subjects": ["Pathology", "Pharmacology"],
    },
    "Cirurgia": {
        "textbooks": ["Surgery_Schwartz.txt"],
        "subjects": ["Surgery", "Anaesthesia"],
    },
    "Pediatria_Ginecologia": {
        "textbooks": ["Pediatrics_Nelson.txt", "Gynecology_Novak.txt", "Obstentrics_Williams.txt"],
        "subjects": ["Pediatrics", "Gynaecology & Obstetrics"],
    },
}

# 1. Descarrega i desa MedMCQA en format JSON per subject
os.makedirs(MEDMCQA_DIR, exist_ok=True)
ds = load_dataset("openlifescienceai/medmcqa", split="train")
all_subjects = set(s for cfg in agents.values() for s in cfg["subjects"])
for subj in all_subjects:
    out_path = os.path.join(MEDMCQA_DIR, f"{subj.replace(' & ', '_').replace(' ', '_')}.json")
    filtered = []
    for ex in ds:
        if ex["subject_name"] == subj:
            filtered.append(
                {
                    "question": ex["question"],
                    "options": [ex["opa"], ex["opb"], ex["opc"], ex["opd"]],
                    "label": chr(65 + ex["cop"]),
                    "explanation": ex.get("exp", ""),
                }
            )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(filtered)} examples for '{subj}' → {out_path}")

# 2. Crea estructura d'agents
os.makedirs(OUTPUT_DIR, exist_ok=True)
for agent_name, cfg in agents.items():
    base_dir = os.path.join(OUTPUT_DIR, agent_name)
    tb_dir = os.path.join(base_dir, "textbooks")
    mq_dir = os.path.join(base_dir, "medmcqa_json")
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(mq_dir, exist_ok=True)

    # Copia els llibres reals
    for tb in cfg["textbooks"]:
        src = os.path.join(TEXTBOOK_DIR, tb)
        dst = os.path.join(tb_dir, tb)
        if os.path.isfile(src):
            shutil.copy(src, dst)
        else:
            print(f"⚠️ No trobat el llibre: {src}")

    # Copia els fitxers JSON MedMCQA filtrats per subject
    for subj in cfg["subjects"]:
        fname = subj.replace(" & ", "_").replace(" ", "_") + ".json"
        src = os.path.join(MEDMCQA_DIR, fname)
        dst = os.path.join(mq_dir, fname)
        if os.path.isfile(src):
            shutil.copy(src, dst)
        else:
            print(f"⚠️ No trobat JSON MedMCQA per: {subj} ({src})")

print(f"\n✅ Estructura generada correctament a: {OUTPUT_DIR}")
