import os
import numpy as np
from sentence_transformers import SentenceTransformer


AGENTS_DIR = "agents_data"  # Carpeta amb subcarpetes per agent
EMBEDDINGS_DIR = "agents_embeddings"  # Carpeta on guardarem els embeddings
CHUNK_SIZE = 256  # nombre de paraules per chunk
MODEL_NAME = "all-MiniLM-L6-v2"

# Crear carpeta per a embeddings si no existeix
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# Carrega model d'embeddings
print(f"Loading embedding model {MODEL_NAME}...")
embed_model = SentenceTransformer(MODEL_NAME)

for agent_name in os.listdir(AGENTS_DIR):
    agent_path = os.path.join(AGENTS_DIR, agent_name)
    textbooks_dir = os.path.join(agent_path, "textbooks")
    if not os.path.isdir(textbooks_dir):
        continue

    print(f"Processing agent: {agent_name}")
    # Reunir tot el text dels llibres
    full_text = []
    for fname in os.listdir(textbooks_dir):
        if fname.endswith(".txt"):
            file_path = os.path.join(textbooks_dir, fname)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    full_text.append(f.read())
            except Exception as e:
                print(f"Warning: failed reading {file_path}: {e}")
    if not full_text:
        print(f"  No textbooks found for {agent_name}, skipping.")
        continue

    # Concatenar i fragmentar per chunks de CHUNK_SIZE paraules
    text = "\n".join(full_text)
    words = text.split()
    chunks = []
    for i in range(0, len(words), CHUNK_SIZE):
        chunk = " ".join(words[i : i + CHUNK_SIZE])
        chunks.append(chunk)

    # Generar embeddings per chunk i fer la mitjana
    print(f"  Creating embeddings for {len(chunks)} chunks...")
    embeddings = embed_model.encode(chunks, convert_to_numpy=True)
    agent_embedding = np.mean(embeddings, axis=0)

    # Guardar embedding
    out_path = os.path.join(EMBEDDINGS_DIR, f"{agent_name}.npy")
    np.save(out_path, agent_embedding)
    print(f"  Saved embedding to {out_path}\n")

print("Embeddings generation completed.")
