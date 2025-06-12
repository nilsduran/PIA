# SinergIA - PIA + Telefónica

## Project Summary
SinergIA is a multi-agent system designed to enhance the performance of language models through the integration of diverse expert agents. The project focuses on fine-tuning language models using LoRA (Low-Rank Adaptation) and evaluating their performance on the MedQA benchmark, which is a medical question-answering dataset. It also includes a user interface for interaction and visualization of results.

## Installation
### Cloning the Repository
To get started, clone the repository to your local machine:
```bash
git clone https://github.com/nilsduran/PIA.git
```
### Requirements
It is recommended to run the project in a virtual environment. 
```bash
cd PIA
python3.11 -m venv .venv-stable
.\.venv-stable\Scripts\activate
pip install -r requirements.txt
```
### Include AI Studio API as an Environment Variable
To use the AI Studio API, you need to add the API key as an environment variable. The API key is specific to the LoRA fine-tuned language models and cannot be used with any other key. To obtain your API key, please email us at
<nils.duran@estudiantat.upc.edu>.
```bash
export GOOGLE_API_KEY="your_api_key"
```
Or, if you are using cmd:
```bash
set GOOGLE_API_KEY=your_api_key
```
Or, if you are using PowerShell:
```bash
$env:GOOGLE_API_KEY="your_api_key"
```

## Execution
To run the conversation web app, use the following command:
```bash
streamlit run app.py
```
---

### Benchmarks
Executing the benchmarks may take a couple of hours, depending on the number of questions and how many different model configurations you want to test.
For the MedQA LangGraph benchmark evaluation, use:
```bash
python langgraph_benchmark.py
```
For the performance evaluation of the fine-tuned LoRA language models, use:
```bash
python benchmarking.py
```

### Demo

https://github.com/user-attachments/assets/60f4a9ab-ef25-4228-a35d-0584da7ca7e2

### Main Files

- `app.py` - User application for interaction with the system
- `main/langgraph_workflow.py` - Implementation of the multi-agent workflow with LangGraph
- `main/langgraph_benchmark.py` - Evaluation of the MedQA benchmark for the LangGraph multi-agent system
- `main/langgraph_conversa.py` - Implementation of the conversation between expert agents
- `main/benchmarking.py` - Performance evaluation of the fine-tuned LoRA language models

### Additional Files

- `scripts/creació_fitxers.py` - Creation of necessary files for fine-tuning and evaluation
- `scripts/funcions_auxiliars.py` - Auxiliary functions for making API calls to AI Studio and extracting responses
- `scripts/diversitat_LoRA.py` - Calculation of diversity for the fine-tuned LoRA models
- `scripts/temperature_finder.py` - Effect of temperature on language models
- `scripts/elo.py` - Calculation of ELO scores for the models
- `scripts/Versió1_vs_Versió2.py` - Head-to-head comparison between two versions of LoRA to choose the best one
- `scripts/embeddings_loader.py` - Loading embeddings for expert agents
- `scripts/plots_extres.ipynb` - Notebook for miscellaneous result visualizations

### Data and Resources

- `agents_embeddings/` - Directory containing embeddings for expert agents
- `data/training_data_csv/` - Directory containing CSV files with training data for LoRA
- `data/textbooks/` - Directory containing medical and scientific textbooks
- `data/battle_votes.csv` - CSV file containing votes from the model battle
- `data/elo_ratings_with_ci.csv` - CSV file containing ELO scores with confidence intervals
- `data/dissimilarity_matrix.csv` - CSV file containing dissimilarity scores between models
- `data/expert_scores.json` - JSON file containing expert relevance scores

### Visualization of Results

- `plots/elo_ratings_with_ci.png` - Elo ratings with confidence intervals
- `plots/model_benchmarks.png` - Accuracy of fine-tuned LoRA models on MedQA
- `plots/model_accuracy_vs_agents_zoomed.png` - LangGraph accuracy on MedQA, zoomed in on agents
- `plots/expert_ranks.png` - Expert selection at each step
- `plots/dissimilarity_matrix_heatmap.png` - Diversity dissimilarity scores between models
- `plots/accuracy_comparison_all_models.png` - Accuracy of fine-tuned models (v1 and v2)
- `plots/distribution_per_temperature.png` - Accuracy and rejection rate based on temperature settings
