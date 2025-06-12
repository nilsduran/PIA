import os
import numpy as np

# Globals per emmagatzemar el model i embeddings un cop carregats
_sbert_model = None
_agents_textbook_embeddings_data = {}
_sbert_initialized = False

# Constants (originalment a langgraph_workflow.py)
AGENTS_EMBEDDINGS_DIR_PATH = "agents_embeddings"
AGENT_EMBEDDING_FILENAME_MAP = {
    "Ciències Bàsiques": "Ciències_Bàsiques",
    "Medicina General": "Medicina_General",
    "Patologia i Farmacologia": "Patologia_Farmacologia",
    "Cirurgia": "Cirurgia",
    "Pediatria i Ginecologia": "Pediatria_Ginecologia",
}


def _ensure_resources_loaded():
    """
    Funció interna per carregar SentenceTransformer i els embeddings.
    S'assegura que només es carrega una vegada.
    """
    global _sbert_model, _agents_textbook_embeddings_data, _sbert_initialized

    if not _sbert_initialized:
        try:
            from sentence_transformers import SentenceTransformer  # AQUÍ ES FA L'IMPORT PESAT

            _sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

            if os.path.isdir(AGENTS_EMBEDDINGS_DIR_PATH) and _sbert_model:
                for display_name, filename_stem in AGENT_EMBEDDING_FILENAME_MAP.items():
                    file_path = os.path.join(AGENTS_EMBEDDINGS_DIR_PATH, f"{filename_stem}.npy")
                    if os.path.exists(file_path):
                        emb = np.load(file_path)
                        _agents_textbook_embeddings_data[display_name] = emb.reshape(1, -1) if emb.ndim == 1 else emb
            else:
                # Error crític si el directori principal d'embeddings no existeix
                _sbert_initialized = True  # Marcar com inicialitzat per evitar bucles si es torna a cridar
                raise FileNotFoundError(
                    f"HEAVY_MODEL_LOADER: Error Crític - El directori d'embeddings de llibres '{AGENTS_EMBEDDINGS_DIR_PATH}' no s'ha trobat "
                    "o el model SBERT no s'ha pogut carregar. El ruteig d'experts no funcionarà."
                )

            _sbert_initialized = True

        except Exception as e:
            _sbert_initialized = True
            print(f"HEAVY_MODEL_LOADER: Error inesperat durant la càrrega de SBERT o embeddings: {e}")


def get_sbert_model():
    """Retorna la instància del model SentenceTransformer, carregant-la si cal."""
    if not _sbert_initialized:
        _ensure_resources_loaded()
    return _sbert_model


def get_agents_textbook_embeddings():
    """Retorna el diccionari d'embeddings dels llibres, carregant-los si cal."""
    if not _sbert_initialized:
        _ensure_resources_loaded()
    return _agents_textbook_embeddings_data


def preload_heavy_models():
    """Funció pública per ser cridada des de app.py per forçar la càrrega."""
    _ensure_resources_loaded()
