# La Unió en la Diversitat

## Estructura del Projecte

### Fitxers Principals
- `langgraph_conversa.py` - MVP: Implementa la funcionalitat de conversa utilitzant LangGraph
- `langgraph_benchmark.py` - Avaluació del benchmark MedQA del sistema multi-agent de LangGraph
- `benchmarking.py` - Avaluació de rendiment dels models de llenguatge ajustats LoRA

### Fitxers Addicionals
- `temperature_finder.py` - Eina per avaluar l'efecte de la temperatura en els models de llenguatge
- `Versió1_vs_Versió2.py` - Comparació head-to-head entre dues versions de LoRA
- `xat_models_finetuned.py` - Conversa pre-MVP amb models de llenguatge ajustats LoRA i temperatura com a paràmetre de diversitat
- `crear_csv.py` - Script per crear fitxers CSV, probablement a partir de dades processades
- `crear_embeddings.py` - Crea embeddings amb el contingut dels llibres de text
- `dades_agents.py` - Script per crear i gestionar dades dels agents amb què s'ajusten els models i s'entrenenen els embeddings

### Dades i Recursos

- `textbooks/` - Directori que conté llibres de text mèdics i científics
- `agents_data/` - Directori que conté dades especialitzades dels agents

### Resultats i Visualització

- `accuracy_comparison_all_models.png` - Visualització comparant la precisió de tots els models
- `model_benchmark_results.png` - Visualització dels resultats de les proves
- `model_benchmarks.png`/`model_benchmarks_0.png` - Visualitzacions addicionals de proves
- `agentic_workflow_benchmark.png` - Proves de rendiment dels fluxos de treball dels agents
- `distribution_per_temperature.png` - Anàlisi de distribució segons configuracions de temperatura
- `expert_ranks.png` - Visualització de les dades de classificació d'experts
- `expert_scores.json` - Dades en brut de les puntuacions de la rellevància dels experts

### Notebooks i Configuració

- `LoRA_kaggle.ipynb` - Notebook de Jupyter per a l'entrenament amb LoRA (deprecated per falta de recursos)
- `expert_triat.ipynb` - Notebook per a la visualització de la selecció d'experts

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
python langgraph_conversa.py
```
Per a l'avaluació del benchmark MedQA, utilitzeu:
```bash
python langgraph_benchmark.py
```
Per a l'avaluació de rendiment dels models de llenguatge ajustats LoRA, utilitzeu:
```bash
python benchmarking.py
```



## Idees
- Avaluar qualitativament el sistema de conversa divers amb el no-divers, tipu lmarena i l'usuari tria el millor.
- Elo rating dels models més o menys diversos.
- Calcular quantes votacions calen per a comparar un nombre X de models si és head-to-head.