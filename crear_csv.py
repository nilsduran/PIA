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
CSV_FIELDNAMES = ["Question", "A", "B", "C", "D", "Explanation", "Answer"]

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

# Escriu els fitxers CSV
for agent, topic_dict in agents_data.items():
    rows = []
    all_questions = []

    # Agrupem totes les preguntes d'aquest agent
    for topic, questions in topic_dict.items():
        all_questions.extend([(q, topic) for q in questions])

    # Barregem totes les preguntes
    shuffle(all_questions)

    # Processa les preguntes
    for q, topic in all_questions:
        if len(rows) >= NUM_PREGUNTES_PER_AGENT:
            break

        # Si l'opció D és "None" --> "None." perquè la llegeixi bé
        if q["options"][3] == "None":
            q["options"][3] = "None."

        # Verifiquem les opcions buides i la qualitat de l'explicació
        if q.get("explanation") is None or len(q.get("explanation", "")) < 50 or len(q.get("explanation", "")) > 4980:
            continue

        q["question"] = q["question"].replace('"', "'")  # Substitueix cometes dobles per simples
        q["options"] = [opt.replace('"', "'") for opt in q["options"]]  # Substitueix cometes dobles per simples
        q["explanation"] = q["explanation"].replace('"', "'")  # Substitueix cometes dobles per simples

        # Si arribem aquí, la pregunta és vàlida
        rows.append(
            {
                CSV_FIELDNAMES[0]: q["question"].strip(),
                CSV_FIELDNAMES[1]: q["options"][0].strip(),
                CSV_FIELDNAMES[2]: q["options"][1].strip(),
                CSV_FIELDNAMES[3]: q["options"][2].strip(),
                CSV_FIELDNAMES[4]: q["options"][3].strip(),
                CSV_FIELDNAMES[5]: q.get("explanation", "").strip(),
                CSV_FIELDNAMES[6]: q["label"],
            }
        )

    # Escriu el CSV si tenim files
    out_path = os.path.join(OUTPUT_DIR, f"{agent}.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Creat: {out_path} ({len(rows)} preguntes)")
