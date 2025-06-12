# SinergIA - La Unió en la Diversitat

## Resum del Projecte
SinergIA és un sistema multi-agent dissenyat per millorar el rendiment dels models de llenguatge mitjançant la integració d'agents experts diversos. El projecte se centra en l'ajustament de models de llenguatge utilitzant LoRA (Low-Rank Adaptation) i l'avaluació del seu rendiment en el benchmark MedQA, que és un conjunt de dades de preguntes i respostes mèdiques. També inclou una interfície d'usuari per a la interacció i visualització dels resultats.

## Instal·lació
### Clona del repositori
Per començar, clona el repositori a la teva màquina local:
```bash
git clone https://github.com/nilsduran/PIA.git
```
### Requisits
Es recomana executar el projecte en un entorn virtual. 
```bash
python3.11 -m venv .venv-stable
.\.venv-stable\Scripts\activate
pip install -r requirements.txt
```
### Incloure API de AI Studio com a variable d'entorn
Per a utilitzar l'API d'AI Studio, cal afegir la clau d'API com a variable d'entorn. No es possible fer servir qualsevol clau API, ja que els models de llenguatge ajustats LoRA estan vinculats a un compte d'AI Studio específic. Per obtenir la teva clau d'API, escriu-nos un correu electrònic a <nils.duran@estudiantat.upc.edu>.
```bash
export GOOGLE_API_KEY="la_teva_clau_api"
```
O, si utilitzes Windows:
cmd:
```bash
set GOOGLE_API_KEY=la_teva_clau_api
```
PowerShell:
```bash
$env:GOOGLE_API_KEY="la_teva_clau_api"
```

## Execució
Per executar el sistema de conversa, utilitzeu el següent comandament:
```bash
streamlit run app.py
```
---
### Benchmarks
Els benchmarks trigen un parell d'hores, segons el nombre de preguntes i quantes configuracions diferents de models es volen provar.
Per a l'avaluació del benchmark MedQA, utilitzeu:
```bash
python langgraph_benchmark.py
```
Per a l'avaluació de rendiment dels models de llenguatge ajustats LoRA, utilitzeu:
```bash
python benchmarking.py
```

### Demo

https://github.com/user-attachments/assets/60f4a9ab-ef25-4228-a35d-0584da7ca7e2

### Fitxers Principals
- `app.py` - Applicació d'usuari per a la interacció amb el sistema
- `main/langgraph_workflow.py` - Implementació del flux de treball multi-agent amb LangGraph
- `main/langgraph_benchmark.py` - Avaluació del benchmark MedQA del sistema multi-agent de LangGraph
- `main/langgraph_conversa.py` - Implementació de la conversa entre agents experts
- `main/benchmarking.py` - Avaluació de rendiment dels models de llenguatge ajustats LoRA

### Fitxers Addicionals
- `scripts/creació_fitxers.py` - Creació de fitxers necessaris per a l'ajustament i l'avaluació
- `scripts/funcions_auxiliars.py` - Funcions auxiliars per fer crides a l'API d'AI Studio i extreure respostes
- `scripts/diversitat_LoRA.py` - Càlcul de la diversitat dels models ajustats LoRA
- `scripts/temperature_finder.py` - Efecte de la temperatura en els models de llenguatge
- `scripts/elo.py` - Càlcul de les puntuacions ELO dels models
- `scripts/Versió1_vs_Versió2.py` - Comparació head-to-head entre dues versions de LoRA per triar la millor
- `scripts/embeddings_loader.py` - Càrrega dels embeddings dels agents experts
- `scripts/plots_extres.ipynb` - Notebook per a la visualització de resultats misc.

### Dades i Recursos

- `agents_embeddings/` - Directori que conté els embeddings dels agents experts
- `data/training_data_csv/` - Directori que conté els fitxers CSV amb les dades d'entrenament del LoRA
- `data/textbooks/` - Directori que conté llibres de text mèdics i científics
- `data/battle_votes.csv` - Fitxer CSV que conté les votacions de la batalla entre models
- `data/elo_ratings_with_ci.csv` - Fitxer CSV que conté les puntuacions ELO amb intervals de confiança
- `data/dissimilarity_matrix.csv` - Fitxer CSV que conté les puntuacions de dissimilaritat entre models
- `data/expert_scores.json` - Fitxer JSON que conté les puntuacions de rellevància dels experts

### Visualització de Resultats

- `plots/elo_ratings_with_ci.png` - Puntuacions ELO amb intervals de confiança
- `plots/model_benchmarks.png` - Precisió dels models ajustats LoRA a MedQA
- `plots/model_accuracy_vs_agents_zoomed.png` - Precisió del LangGraph a MedQA, amb zoom en els agents
- `plots/expert_ranks.png` - Tria dels experts en cada pas
- `plots/dissimilarity_matrix_heatmap.png` - Puntuacions de dissimilaritat de diversitat entre models
- `plots/accuracy_comparison_all_models.png` - Precisió dels models ajustats (v1 i v2)
- `plots/distribution_per_temperature.png` - Precisió i taxa de refús segons la temperatura