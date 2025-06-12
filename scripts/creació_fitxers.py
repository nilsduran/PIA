import os
import shutil
import csv
from collections import defaultdict
from random import shuffle
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import sys

# --- Configuracions globals i rutes ---
TEXTBOOK_SOURCE_DIR = "data/textbooks"  # Entrada: Directori amb els fitxers .txt dels llibres
TRAINING_DATA_FINAL_DIR = "data/training_data_csv"  # Sortida final: CSVs per a agents
EMBEDDINGS_OUTPUT_FINAL_DIR = "agents_embeddings"  # Sortida final: Embeddings per a agents

# Definició dels agents amb els llibres reals i els temes de MedMCQA
AGENTS_CONFIG = {
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

# Configuració per a la generació de CSV
NUM_PREGUNTES_PER_AGENT = 5000
CSV_FIELDNAMES = ["Question", "A", "B", "C", "D", "Explanation", "Answer"]

# Configuració per a la generació d'Embeddings
CHUNK_SIZE = 256  # nombre de paraules per chunk
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# --- Bloc d'execució principal ---
def main():
    print("Iniciant pipeline de processament de dades...")

    # Crear directoris de sortida (i netejar si ja existeixen per a una execució neta)
    for output_dir in [TRAINING_DATA_FINAL_DIR, EMBEDDINGS_OUTPUT_FINAL_DIR]:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            print(f"Directori de sortida existent eliminat: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    print("Directoris de sortida preparats.")

    # --- Fase 1: Carregar i preparar MedMCQA en memòria ---
    print("\n--- Fase 1: Carregant i preparant dades de MedMCQA en memòria ---")

    # Recollir tots els noms de temes únics necessaris per a la generació de CSV
    all_subjects_needed = set()
    for agent_cfg in AGENTS_CONFIG.values():
        all_subjects_needed.update(agent_cfg["subjects"])

    try:
        print("Descarregant dataset 'openlifescienceai/medmcqa'...")
        ds = load_dataset("openlifescienceai/medmcqa", split="train")

        # Guardar les preguntes filtrades per subjecte en memòria
        all_medmcqa_data_by_subject = defaultdict(list)
        for ex in ds:
            if ex["subject_name"] in all_subjects_needed:
                subject_name = ex["subject_name"].replace(" & ", "_").replace(" ", "_")
                all_medmcqa_data_by_subject[subject_name].append(
                    {
                        "question": ex["question"],
                        "options": [ex["opa"], ex["opb"], ex["opc"], ex["opd"]],
                        "label": chr(65 + ex["cop"]),  # 'A', 'B', 'C', 'D'
                        "explanation": ex.get("exp", ""),
                    }
                )
        print("Dades de MedMCQA processades i carregades en memòria.")
    except Exception as e:
        print(f"❌ Error en carregar/processar MedMCQA: {e}")
        sys.exit(1)

    # --- Fase 2: Crear sortides CSV ---
    print("\n--- Fase 2: Creant sortides CSV ---")
    for agent_name, agent_cfg in AGENTS_CONFIG.items():
        rows = []
        all_questions_for_agent = []

        # Agrupem totes les preguntes d'aquest agent a partir de les dades en memòria
        for subj_original in agent_cfg["subjects"]:
            subject_name = subj_original.replace(" & ", "_").replace(" ", "_")
            questions_from_topic = all_medmcqa_data_by_subject.get(subject_name, [])
            print(f"  Afegint {len(questions_from_topic)} preguntes de '{subj_original}' a '{agent_name}'")
            all_questions_for_agent.extend(questions_from_topic)

        # Barregem totes les preguntes
        shuffle(all_questions_for_agent)

        # Processa les preguntes
        for q in all_questions_for_agent:
            if len(rows) >= NUM_PREGUNTES_PER_AGENT:
                break

            # Si l'opció D és "None" --> "None." perquè la llegeixi bé
            if q["options"][3] == "None":
                q["options"][3] = "None."

            # Verifiquem la qualitat de l'explicació
            explanation = q.get("explanation", "")
            if not explanation or len(explanation) < 50 or len(explanation) > 4980:
                continue

            # Substitueix cometes dobles per simples per evitar problemes amb CSV
            q["question"] = q["question"].replace('"', "'")
            q["options"] = [opt.replace('"', "'") for opt in q["options"]]
            q["explanation"] = explanation.replace('"', "'")

            rows.append(
                {
                    CSV_FIELDNAMES[0]: q["question"].strip(),
                    CSV_FIELDNAMES[1]: q["options"][0].strip(),
                    CSV_FIELDNAMES[2]: q["options"][1].strip(),
                    CSV_FIELDNAMES[3]: q["options"][2].strip(),
                    CSV_FIELDNAMES[4]: q["options"][3].strip(),
                    CSV_FIELDNAMES[5]: q["explanation"].strip(),
                    CSV_FIELDNAMES[6]: q["label"],
                }
            )

        # Escriu el CSV si tenim files
        out_path = os.path.join(TRAINING_DATA_FINAL_DIR, f"{agent_name}.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Creat: {out_path} ({len(rows)} preguntes)")

    print(f"\n✅ Fase 2 completada: Fitxers CSV generats a {TRAINING_DATA_FINAL_DIR}")

    # --- Fase 3: Crear Embeddings ---
    print("\n--- Fase 3: Creant Embeddings ---")

    if not os.path.isdir(TEXTBOOK_SOURCE_DIR):
        print(f"❌ Error: El directori '{TEXTBOOK_SOURCE_DIR}' no existeix. No es poden generar embeddings.")
        sys.exit(1)

    print(f"Carregant model d'embeddings {EMBEDDING_MODEL_NAME}...")
    try:
        embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as e:
        print(f"❌ Error en carregar el model d'embeddings '{EMBEDDING_MODEL_NAME}': {e}")
        sys.exit(1)

    for agent_name, agent_cfg in AGENTS_CONFIG.items():
        print(f"Processant agent: {agent_name}")
        # Reunir tot el text dels llibres directament de la carpeta d'origen
        full_text = []
        for tb in agent_cfg["textbooks"]:
            file_path = os.path.join(TEXTBOOK_SOURCE_DIR, tb)
            with open(file_path, "r", encoding="utf-8") as f:
                full_text.append(f.read())

        # Concatenar i fragmentar per chunks de CHUNK_SIZE paraules
        text = "\n".join(full_text)
        words = text.split()
        chunks = []
        for i in range(0, len(words), CHUNK_SIZE):
            chunk = " ".join(words[i : i + CHUNK_SIZE])
            chunks.append(chunk)

        print(f"  Creant embeddings per a {len(chunks)} fragments...")
        embeddings = embed_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
        # Fer la mitjana dels embeddings per obtenir un embedding per a tot l'agent
        agent_embedding = np.mean(embeddings, axis=0)

        # Guardar embedding
        out_path = os.path.join(EMBEDDINGS_OUTPUT_FINAL_DIR, f"{agent_name}.npy")
        np.save(out_path, agent_embedding)
        print(f"  Embedding desat a {out_path}\n")

    print("✅ Fase 3 completada: Generació d'embeddings finalitzada.")
    print("\n🎉 Tot el processament ha finalitzat correctament!")


if __name__ == "__main__":
    main()
