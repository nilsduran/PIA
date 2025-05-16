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


# Defineix les columnes que tindran els CSV
CSV_FIELDNAMES = ["question", "option_A", "option_B", "option_C", "option_D", "explanation", "answer"]

# Carrega les dades de cada fitxer JSON
for filename in os.listdir(INPUT_DIR):
    if filename.endswith(".json"):
        file_path = os.path.join(INPUT_DIR, filename)
        topic_key = os.path.splitext(filename)[0]

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                questions = json.load(f)

                # Verifica que questions és una llista
                if not isinstance(questions, list):
                    print(f"Error: {file_path} no conté una llista. Format no esperat.")
                    continue

                # Assigna les preguntes als agents corresponents
                for agent, topics in AGENT_TOPICS.items():
                    if topic_key in topics:
                        print(f"Afegint {len(questions)} preguntes de {topic_key} a {agent}")
                        agents_data[agent][topic_key].extend(questions)
        except Exception as e:
            print(f"Error llegint {file_path}: {e}")

# Estadístiques per al seguiment
stats = {agent: {"total": 0, "valid": 0, "skipped": 0} for agent in AGENT_TOPICS}

# Escriu els fitxers CSV
for agent, topic_dict in agents_data.items():
    rows = []
    all_questions = []

    # Primer, agrupem totes les preguntes d'aquest agent
    for topic, questions in topic_dict.items():
        all_questions.extend([(q, topic) for q in questions])

    # Barregem totes les preguntes
    shuffle(all_questions)

    stats[agent]["total"] = len(all_questions)

    # Processa les preguntes
    for q, topic in all_questions:
        if len(rows) >= NUM_PREGUNTES_PER_AGENT:
            break

        stats[agent]["processed"] = stats[agent].get("processed", 0) + 1

        # Verifiquem que té els camps mínims
        if not all(key in q for key in ["question", "options", "label"]):
            stats[agent]["missing_fields"] = stats[agent].get("missing_fields", 0) + 1
            continue

        # Verifiquem que options és una llista amb 4 elements
        if not isinstance(q["options"], list) or len(q["options"]) != 4:
            stats[agent]["invalid_options"] = stats[agent].get("invalid_options", 0) + 1
            continue

        # Verifiquem que l'etiqueta és vàlida
        if not isinstance(q["label"], str) or q["label"] not in ["A", "B", "C", "D"]:
            stats[agent]["invalid_label"] = stats[agent].get("invalid_label", 0) + 1
            continue

        # Verifiquem les opcions buides i la qualitat de l'explicació
        if (
            q["options"][3] == ""
            or q["options"][3] is None
            or q.get("explanation") is None
            or len(q.get("explanation", "")) < 10
            or len(q.get("explanation", "")) > 4980
        ):
            stats[agent]["invalid_content"] = stats[agent].get("invalid_content", 0) + 1
            continue

        # Si arribem aquí, la pregunta és vàlida
        rows.append(
            {
                "question": q["question"].strip(),
                "option_A": q["options"][0].strip(),
                "option_B": q["options"][1].strip(),
                "option_C": q["options"][2].strip(),
                "option_D": q["options"][3].strip(),
                "explanation": q.get("explanation", "").strip(),
                "answer": q["label"],
            }
        )
        stats[agent]["valid"] += 1

    # Escriu el CSV si tenim files
    if rows:
        out_path = os.path.join(OUTPUT_DIR, f"{agent}.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        print(f"✅ Creat: {out_path} ({len(rows)} preguntes)")
    else:
        print(f"❌ Cap pregunta vàlida trobada per a {agent}")
