# SinergIA - La Unió en la Diversitat

## Estructura del Projecte

### Fitxers Principals
- `app.py` - Applicació d'usuari per a la interacció amb el sistema
- `langgraph_benchmark.py` - Avaluació del benchmark MedQA del sistema multi-agent de LangGraph
- `langgraph_workflow.py` - Implementació del flux de treball multi-agent amb LangGraph
- `xat_models_finetuned.py` - MVP: Implementa la funcionalitat de conversa amb models de llenguatge ajustats LoRA
- `benchmarking.py` - Avaluació de rendiment dels models de llenguatge ajustats LoRA

### Fitxers Addicionals
- `temperature_finder.py` - Eina per avaluar l'efecte de la temperatura en els models de llenguatge
- `elo.py` - Càlcul de les puntuacions ELO dels models
- `diversitat_LoRA.py` - Càlcul de la diversitat dels models ajustats LoRA
- `Versió1_vs_Versió2.py` - Comparació head-to-head entre dues versions de LoRA per triar la millor
- `xat_models_finetuned.py` - Conversa MVP amb models de llenguatge ajustats LoRA i temperatura com a paràmetre de diversitat
- `crear_csv.py` - Script per crear fitxers CSV, probablement a partir de dades processades
- `crear_embeddings.py` - Crea embeddings amb el contingut dels llibres de text
- `dades_agents.py` - Script per crear i gestionar dades dels agents amb què s'ajusten els models i s'entrenenen els embeddings

### Dades i Recursos

- `agents_data/` - Directori que conté dades especialitzades dels agents
- `agents_embeddings/` - Directori que conté els embeddings dels agents experts
- `csv_output/` - Directori que conté els fitxers CSV amb les dades d'entrenament del LoRA
- `textbooks/` - Directori que conté llibres de text mèdics i científics
- `medmcqa_json/` - Directori que conté les dades del benchmark MedMCQA en format JSON
- `battle_votes.csv` - Fitxer CSV que conté les votacions de la batalla entre models
- `elo_ratings_with_ci.csv` - Fitxer CSV que conté les puntuacions ELO amb intervals de confiança
- `dissimilarity_matrix.csv` - Fitxer CSV que conté les puntuacions de dissimilaritat entre models

### Resultats i Visualització

- `elo_ratings_with_ci.png` - Visualització de les puntuacions ELO amb intervals de confiança
- `model_benchmarks.png` - Visualització de la precisió dels models ajustats LoRA a MedQA
- `model_accuracy_vs_agents_zoomed.png` - Visualització de la precisió del LangGraph a MedQA, amb zoom en els agents
- `expert_ranks.png` - Visualització de les dades de classificació d'experts
- `dissimilarity_matrix_heatmap.png` - Visualització de les puntuacions de dissimilaritat de diversitat entre models
- `agentic_workflow_benchmark.png` - Proves de rendiment dels fluxos de treball dels agents
- `accuracy_comparison_all_models.png` - Visualització comparant la precisió dels models ajustats (v1 i v2)
- `distribution_per_temperature.png` - Anàlisi de distribució segons configuracions de temperatura
- `expert_scores.json` - Dades en brut de les puntuacions de la rellevància dels experts

### Notebooks i Configuració

- `plots_extres.ipynb` - Notebook per a la visualització de resultats misc.
- `LoRA_kaggle.ipynb` - Notebook de Jupyter per a l'entrenament amb LoRA (deprecated per falta de recursos)
- `PIA_RAG.ipynb` - Notebook per a la implementació de Retrieval-Augmented Generation (RAG) amb LangGraph (deprecated per falta de recursos)


## Instal·lació
Es recomana executar el projecte en un entorn virtual. 
```bash
python3.11 -m venv .venv-stable
.\.venv-stable\Scripts\activate
pip install -r requirements.txt
```
## Execució
Per executar el sistema de conversa, utilitzeu el següent comandament:
```bash
streamlit run app.py
```
Els benchmarks trigen un parell d'hores, segons el nombre de preguntes i quantes configuracions diferents de models es volen provar.
Per a l'avaluació del benchmark MedQA, utilitzeu:
```bash
python langgraph_benchmark.py
```
Per a l'avaluació de rendiment dels models de llenguatge ajustats LoRA, utilitzeu:
```bash
python benchmarking.py
```
