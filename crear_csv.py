import json
import csv
import os
from collections import defaultdict
from random import shuffle

# Defineix la ruta del directori amb els fitxers JSON de MedMCQA
INPUT_DIR = "medmcqa_json"
OUTPUT_DIR = "csv_output"
NUM_PREGUNTES_PER_AGENT = 5000
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Defineix els agents i els seus temes
AGENT_TOPICS = {
    "Medicina_General": ["Medicine", "Social_Preventive_Medicine", "Forensic_Medicine"],
    "Ciències_Bàsiques": ["Anatomy", "Physiology", "Biochemistry", "Microbiology"],
    "Patologia_Farmacologia": ["Pathology", "Pharmacology"],
    "Cirurgia": ["Surgery", "Anaesthesia"],
    "Pediatria_Ginecologia": ["Pediatrics", "Gynaecology_Obstetrics"],
}

# Inicialitza diccionaris per guardar preguntes per agent i per tema
agents_data = {agent: defaultdict(list) for agent in AGENT_TOPICS}

# Assegura't que el directori d'entrada existeix
if not os.path.isdir(INPUT_DIR):
    print(f"El directori {INPUT_DIR} no existeix. Assegura't que el camí és correcte.")
    exit(1)


# Carrega les dades de cada fitxer JSON
for filename in os.listdir(INPUT_DIR):
    if filename.endswith(".json"):
        file_path = os.path.join(INPUT_DIR, filename)
        topic_key = os.path.splitext(filename)[0]

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                questions = json.load(f)

                # Assigna les preguntes als agents corresponents
                for agent, topics in AGENT_TOPICS.items():
                    if topic_key in topics:
                        print(f"Afegint {len(questions)} preguntes de {topic_key} a {agent}")
                        agents_data[agent][topic_key].extend(questions)
        except Exception as e:
            print(f"Error llegint {file_path}: {e}")

# Escriu els fitxers CSV
for agent, topic_dict in agents_data.items():
    rows = []
    for topic, questions in topic_dict.items():
        shuffle(questions)  # barreja per si hi ha més de 2000
        for q in questions:
            if len(rows) >= NUM_PREGUNTES_PER_AGENT:
                break
            # Assegura't que totes les claus necessàries estan presents
            if all(key in q for key in ["question", "options", "label"]):
                # Salta si la quarta opció està buida o és NaN
                if (
                    q["options"][3] == ""
                    or q["options"][3] is None
                    or q.get("explanation") is None
                    or len(q.get("explanation", "")) < 10
                    or len(q.get("explanation", "")) > 4990
                ):
                    continue

                rows.append(
                    {
                        "question": q["question"],
                        "option_A": q["options"][0],
                        "option_B": q["options"][1],
                        "option_C": q["options"][2],
                        "option_D": q["options"][3],
                        "explanation": q.get("explanation"),
                        "answer": q["label"],
                    }
                )

    if rows:
        out_path = os.path.join(OUTPUT_DIR, f"{agent}.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Creat: {out_path} ({len(rows)} exemples)")
    else:
        print(f"Cap pregunta trobada per a {agent}")
