import random
import json
from datasets import load_dataset
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
import os
from collections import Counter
import matplotlib.pyplot as plt
import time
import seaborn as sns

# ----------------------------
# Configure your Gemini API key
# ----------------------------
# python3.11 -m venv .venv-stable
# .\.venv-stable\Scripts\activate
# $env:GOOGLE_API_KEY = "YOUR_ACTUAL_API_KEY_HERE"
# "AIzaSyA8KwZ5wYVoaiLlRMOI_ZsS2PYXH0qq4ms"
# "AIzaSyDxk7cxcrDx3mcofYIosCggfkVbyHedO4w"

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    print("API KEY NOT SET. Exiting.")
    exit()

genai.configure(api_key=API_KEY)

MODEL_NAME = "gemini-2.0-flash-lite"
EXAMPLES_DIR = "exemples_MedMCQA"
KNOWLEDGE_FILES_DIR = "books"  # obsolet
MAX_OUTPUT_TOKENS_WITH_REASONING = 1000
NUM_SAMPLES_TO_TEST = 10
SUBJECTS = [
    "Medicine",
    "Skin",
    "Gynaecology_Obstetrics",
    "Anatomy",
    "Social_Preventive_Medicine",
    # "Medicine",
    # "Pathology",
    # "Pharmacology",
    # "Pediatrics",
    # "Surgery",
    # "Anaesthesia",
    # "Anatomy",
    # "Biochemistry",
    # "Dental",
    # "ENT",
    # "Forensic_Medicine",
    # "Gynaecology_Obstetrics",
    # "Microbiology",
    # "Ophthalmology",
    # "Orthopaedics",
    # "Psychiatry",
    # "Physiology",
    # "Psychiatry",
    # "Radiology",
    # "Skin",
    # "Social_Preventive_Medicine",
    "Mixed",
]

train_ds = load_dataset("openlifescienceai/medmcqa", split="train")


# --- Helper Function to Read Knowledge Files ---
def load_examples(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def parse_llm_response(raw_response: str) -> tuple[str | None, str]:
    """Extracts the final answer letter and reasoning from the LLM response."""
    if not isinstance(raw_response, str) or "Final Answer: " not in raw_response:
        return None, "Error: Invalid response format."

    parts = raw_response.rsplit("Final Answer: ", 1)
    reasoning = parts[0].strip() or "No reasoning provided."
    llm_letter = parts[1].strip() if parts[1].strip() in ["A", "B", "C", "D"] else None

    return llm_letter, reasoning


# --- Core Function to Query Gemini ---
agents_config = {}
for subj in SUBJECTS:
    fname = subj + ".txt"
    path = os.path.join(EXAMPLES_DIR, fname)
    agents_config[subj] = {
        "persona": f"You are a specialist in {subj}. Use the following training examples to answer precisely.",
        "knowledge_file": path,
    }

# Add generic assistant without subject-specific examples
agents_config["Generic Assistant"] = {"persona": "You are a professional medical assistant.", "knowledge_file": None}


# --- Core Query Function ---
def ask_gemini(full_prompt: str) -> str:
    model = genai.GenerativeModel(MODEL_NAME)
    try:
        # Generate content
        response = model.generate_content(
            contents=full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2, max_output_tokens=MAX_OUTPUT_TOKENS_WITH_REASONING
            ),
        )
    except ResourceExhausted as e:
        print(f"Unexpected Error: {e}")
        print(f"Prompt: {full_prompt}")
        time.sleep(5)
        return ask_gemini(full_prompt)

    # Extract text
    text = getattr(response, "text", None)
    print(text)
    return text.strip()


# --- Dataset ---
dataset = load_dataset("openlifescienceai/medmcqa", split="validation")
samples_to_process = random.sample(list(dataset), NUM_SAMPLES_TO_TEST)
counter = Counter()
for sample in samples_to_process:
    counter[sample["subject_name"]] += 1

print("\n--- Sample Distribution ---")
for subj, count in counter.items():
    print(f"{subj}: {count} samples")


OUTPUT_FORMAT = (
    "First, provide a brief step-by-step reasoning for your choice based on the question, options, and any provided knowledge. "
    "After your reasoning, you MUST conclude your entire response with the phrase 'Final Answer: ' "
    "followed by only the single uppercase letter corresponding to the correct option (e.g., A, B, C, or D). "
    "Do NOT include any other text or punctuation after this final letter. "
    "Example format:\n"
    "Reasoning: [Your brief reasoning here, detailing how you arrived at the answer.]\n"
    "Final Answer: C"
)

# --- Data Collection for Evaluation ---
# Initialize result storage
agent_results = {name: {"correct": 0, "total": 0, "responses": []} for name in agents_config}
agent_results["Mode Agent"] = {"correct": 0, "total": 0, "responses": []}

for sample in samples_to_process:
    question = sample["question"]
    opts = [sample["opa"], sample["opb"], sample["opc"], sample["opd"]]
    options_str = "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(opts))
    gold_letter = chr(65 + sample["cop"])

    current_answers = {}
    for name, cfg in agents_config.items():
        persona = cfg["persona"]
        knowledge = ""
        if cfg["knowledge_file"]:
            data = load_examples(cfg["knowledge_file"])
            if data:
                knowledge = f"\n\nUse these examples:\n---BEGIN---\n{data}---END---\n"
        prompt = f"{persona}{knowledge}{OUTPUT_FORMAT}\nQuestion: {question}\n{options_str}"
        raw = ask_gemini(prompt)

        # Parse letter
        llm_letter = None
        if "Final Answer: " in raw:
            parts = raw.rsplit("Final Answer: ", 1)
            if len(parts) == 2:
                candidate = parts[1].strip()
                if candidate in ["A", "B", "C", "D"]:
                    llm_letter = candidate
        else:
            llm_letter = random.choice(["A", "B", "C", "D"])

        # Record
        agent_results[name]["total"] += 1
        agent_results[name]["responses"].append((llm_letter, gold_letter))
        if llm_letter == gold_letter:
            agent_results[name]["correct"] += 1
        current_answers[name] = llm_letter

    # Mode Agent by majority vote
    votes = [l for l in current_answers.values() if l]
    mode_letter = max(votes, key=votes.count) if votes else None

    agent_results["Mode Agent"]["total"] += 1
    agent_results["Mode Agent"]["responses"].append((mode_letter, gold_letter))
    if mode_letter == gold_letter:
        agent_results["Mode Agent"]["correct"] += 1

print("\n\n--- Overall Evaluation Results ---")
accuracies = {}
for agent_name, results in agent_results.items():
    total_processed = results["total"]
    correct_answers = results["correct"]
    if total_processed > 0:
        accuracy = (correct_answers / total_processed) * 100
        accuracies[agent_name] = accuracy
        print(f"{agent_name}: Accuracy = {accuracy:.2f}% ({correct_answers}/{total_processed})")
    else:
        accuracies[agent_name] = 0
        print(f"{agent_name}: No samples processed or all resulted in errors.")


agent_names_for_plot = list(accuracies.keys())
agent_names_for_plot = [
    "Medicina general",
    "Dermatologia",
    "Nefrologia",
    "Infermeria",
    "Drets humans",
    "Barreja",
    "Genèric",
    "Moda (votació)",
]
accuracy_values = list(accuracies.values())
plt.figure(figsize=(12, 7))
# Use a color palette with over 20 different colors
palette = sns.color_palette("tab20", len(agent_names_for_plot))
bars = plt.bar(agent_names_for_plot, accuracy_values, color=palette)
plt.xlabel("Agent")
plt.ylabel("Accuracy (%)")
plt.title(f"Agent Performance on MedMCQA ({len(samples_to_process)} samples)")
plt.xticks(rotation=45, ha="right")
plt.ylim(0, 100)  # Accuracy is between 0 and 100
plt.grid(axis="y", linestyle="--", alpha=0.7)
# Add text labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 1, f"{yval:.1f}%", ha="center", va="bottom")
plt.tight_layout()  # Adjust layout to prevent labels from overlapping
plot_filename = "agent_performance_comparison.png"
plt.savefig(plot_filename)
print(f"\nPerformance plot saved as {plot_filename}")
plt.show()

print("\n--- End of Script ---")
