# SinergIA - La Unió en la Diversitat

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
python3.11 -m venv .venv-stable
.\.venv-stable\Scripts\activate
pip install -r requirements.txt
```
### Include AI Studio API as an Environment Variable
To use the AI Studio API, you need to add the API key as an environment variable.
```bash
export GOOGLE_API_KEY="your_api_key"
```

## Execution
Per executar el sistema de conversa, utilitzeu el següent comandament:
```bash
streamlit run app.py
```

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
![](Demo.mov)

### Main Files
- `app.py` - User application for interaction with the system
- `langgraph_benchmark.py` - Evaluation of the MedQA benchmark for the LangGraph multi-agent system
- `langgraph_workflow.py` - Implementation of the multi-agent workflow with LangGraph
- `benchmarking.py` - Performance evaluation of the fine-tuned LoRA language models
### Additional Files
- `temperature_finder.py` - Tool to evaluate the effect of temperature on language models
- `elo.py` - Calculation of ELO scores for the models
- `diversitat_LoRA.py` - Calculation of diversity for the fine-tuned LoRA models
- `Versió1_vs_Versió2.py` - Head-to-head comparison between two versions of LoRA to choose the best one
- `xat_models_finetuned.py` - MVP chat with fine-tuned language models and temperature as a diversity parameter
- `crear_csv.py` - Script to create CSV files, likely from processed data
- `crear_embeddings.py` - Creates embeddings with the content of textbooks
- `dades_agents.py` - Script to create and manage data for the agents used to fine-tune models and train embeddings
### Data and Resources
- `agents_data/` - Directory containing specialized data for the agents
- `agents_embeddings/` - Directory containing embeddings for expert agents
- `csv_output/` - Directory containing CSV files with training data for LoRA
- `textbooks/` - Directory containing medical and scientific textbooks
- `medmcqa_json/` - Directory containing MedMCQA benchmark data in JSON format
- `battle_votes.csv` - CSV file containing votes from the model battle
- `elo_ratings_with_ci.csv` - CSV file containing ELO scores with confidence intervals
- `dissimilarity_matrix.csv` - CSV file containing dissimilarity scores between models
### Results and Visualization
- `elo_ratings_with_ci.png` - Visualization of ELO scores with confidence intervals
- `model_benchmarks.png` - Visualization of the accuracy of fine-tuned LoRA models on MedQA
- `model_accuracy_vs_agents_zoomed.png` - Visualization of LangGraph accuracy on MedQA, zoomed in on agents
- `expert_ranks.png` - Visualization of expert ranking data
- `dissimilarity_matrix_heatmap.png` - Visualization of diversity dissimilarity scores between models
- `agentic_workflow_benchmark.png` - Performance tests of agent workflows
- `accuracy_comparison_all_models.png` - Visualization comparing the accuracy of fine-tuned models (v1 and v2)
- `distribution_per_temperature.png` - Analysis of distribution based on temperature settings
- `expert_scores.json` - Raw data of expert relevance scores
### Notebooks and Configuration
- `plots_extres.ipynb` - Notebook for miscellaneous result visualizations
- `LoRA_kaggle.ipynb` - Jupyter notebook for training with LoRA (deprecated due to lack of resources)
- `PIA_RAG.ipynb` - Notebook for implementing Retrieval-Augmented Generation (RAG) with LangGraph (deprecated due to lack of resources)
