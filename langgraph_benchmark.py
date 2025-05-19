import json
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from typing import TypedDict, List, Dict, Optional, Any
from langgraph.graph import StateGraph, END
from functools import partial
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
from benchmarking import generate_content, extract_explanation_and_answer


# --- DEFINICIÓ DE L'ESTAT DE LANGGRAPH ---
class AgenticWorkflowState(TypedDict):
    # Entrada
    question_text: str
    options: Dict[str, str]
    correct_answer_idx: str  # Per a l'avaluació
    available_experts: Dict[str, str]  # {"expert_name": "tunedModelId"}
    num_experts_to_select: int  # Paràmetre per al router

    # Estat del Router
    selected_expert_names: List[str]
    routing_rationale: Optional[str]

    # Estat de l'Agregador d'Experts
    expert_responses: List[Dict[str, Any]]  # [{"model_name": str, "answer": str, "explanation": str}]

    # Estat del Supervisor (incloent auto-correcció)
    provisional_supervisor_explanation: Optional[str]
    provisional_supervisor_answer: Optional[str]
    supervisor_self_reflection_prompt: Optional[str]
    final_supervisor_explanation: Optional[str]
    final_supervisor_answer: Optional[str]

    # Logs i errors
    error_messages: List[str]
    node_trace: List[str]  # Per seguir quin node s'executa


# --- DEFINICIÓ DELS EXPERTS AMB PARAULES CLAU ---
EXPERT_DEFINITIONS = {
    "Medicina General": "tunedModels/medicinageneralcsv-q4i0ydc9l1uvxbzsxmii8",
    "Ciències Bàsiques": "tunedModels/ciencies-basiques-2-pfg4bpafqcay88df2kr8",
    "Patologia i Farmacologia": "tunedModels/patologia-farmacologia-2-8iy2ixmy5bluqzw",
    "Cirurgia": "tunedModels/cirurgia-2-2c1cy8nkr5ca5mui15tu4wtlpapp8",
    "Pediatria i Ginecologia": "tunedModels/pediatria-ginecologia-2-ss7f3iy509x7x43h",
}


# --- NODES DE LANGGRAPH ---
def dynamic_expert_router_node(state: AgenticWorkflowState) -> Dict[str, any]:
    """
    Router basat en embeddings de textbooks per cada agent.
    - Embedding de la pregunta amb el mateix SBERT.
    - Cosine similarity amb l'embedding precomputat de cada agent.
    - Retorna els top-k experts.
    """
    state["node_trace"].append("dynamic_expert_router_node")

    question = state["question_text"]
    num_to_select = state["num_experts_to_select"]

    # 1) Embedding de la pregunta
    q_emb = _embedding_model.encode([question], convert_to_numpy=True)

    # 2) Calcula similitud per cada agent disponible
    agent_embeddings_names = {
        "Ciències Bàsiques": "Ciències_Bàsiques",
        "Medicina General": "Medicina_General",
        "Patologia i Farmacologia": "Patologia_Farmacologia",
        "Cirurgia": "Cirurgia",
        "Pediatria i Ginecologia": "Pediatria_Ginecologia",
    }
    scores = []
    for agent_name in state["available_experts"].keys():
        emb_key = agent_embeddings_names.get(agent_name)
        emb = _agents_embeddings.get(emb_key)

        sim = float(cosine_similarity(q_emb, emb)[0][0])
        scores.append((sim, agent_name))

    # 3) Ordena descendent i selecciona top-k
    scores.sort(reverse=True, key=lambda x: x[0])
    # Append scores to JSON instead of rewriting
    try:
        with open("expert_scores.json", "r", encoding="utf-8") as f:
            existing_scores = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_scores = []

    existing_scores.append({"question": question, "scores": scores})

    with open("expert_scores.json", "w", encoding="utf-8") as f:
        json.dump(existing_scores, f, ensure_ascii=False, indent=2)
    selected = [name for _, name in scores[:num_to_select]]

    # 4) Fallback si no hi ha prou seleccionats
    if len(selected) < num_to_select:
        remaining = [n for n in state["available_experts"] if n not in selected]
        needed = num_to_select - len(selected)
        selected += remaining[:needed]

    # 5) Rationale amb puntuacions
    top_info = ", ".join(f"{n}({s:.2f})" for s, n in scores[:num_to_select])
    rationale = f"Semantic routing via textbook embeddings; top-{num_to_select}: {top_info}"

    return {"selected_expert_names": selected, "routing_rationale": rationale}


def expert_consultation_node(
    state: AgenticWorkflowState, system_prompt_template: str, shot_prompt_template: str, temperature: float
):
    state["node_trace"].append("expert_consultation_node")
    expert_responses = []
    selected_names = state.get("selected_expert_names", [])

    if not selected_names:
        state["error_messages"].append("ExpertConsultationNode: No experts were selected by the router.")
        return {"expert_responses": []}

    for expert_name in selected_names:
        model_id = state["available_experts"].get(expert_name)

        prompt = (
            system_prompt_template + "\n" + shot_prompt_template + f"Question: {state['question_text']}\n"
            f"A: {state['options'].get('A', 'N/A')}\nB: {state['options'].get('B', 'N/A')}\n"
            f"C: {state['options'].get('C', 'N/A')}\nD: {state['options'].get('D', 'N/A')}\n"
            f"Explanation:"
        )

        raw_response = generate_content(model_id, prompt, temperature=temperature, max_output_tokens=300)
        explanation, answer = extract_explanation_and_answer(raw_response)

        if answer is None:  # Simple retry
            raw_response = generate_content(
                model_id, prompt, temperature=max(0.1, temperature - 0.2), max_output_tokens=500
            )
            explanation, answer = extract_explanation_and_answer(raw_response)

        expert_responses.append(
            {
                "model_name": expert_name,
                "answer": answer,
                "explanation": explanation if explanation else "No explanation provided.",
            }
        )

    return {"expert_responses": expert_responses}


def synthesizing_supervisor_node(state: AgenticWorkflowState, supervisor_model_id: str):
    state["node_trace"].append("synthesizing_supervisor_node")
    expert_responses = state.get("expert_responses", [])
    question_text = state["question_text"]
    options = state["options"]

    if not expert_responses:
        state["error_messages"].append("SupervisorNode: No expert responses to synthesize.")
        return {
            "provisional_supervisor_explanation": "Error: No expert responses available.",
            "provisional_supervisor_answer": None,
            "final_supervisor_explanation": "Error: No expert responses available.",
            "final_supervisor_answer": None,
        }

    # --- Primera Passada del Supervisor (Síntesi Inicial) ---
    initial_prompt_parts = [
        "You are a supervising medical expert. Your task is to synthesize the explanations and answers from several specialist AI models for a given multiple-choice question.",
        "Review all proposed answers and explanations carefully. Then, provide a consolidated explanation and select the single best answer (A, B, C, or D).",
        "Focus on identifying the most coherent and medically sound reasoning among the experts.",
        f"\nQuestion: {question_text}",
        f"Options:\nA: {options.get('A', 'N/A')}\nB: {options.get('B', 'N/A')}\nC: {options.get('C', 'N/A')}\nD: {options.get('D', 'N/A')}",
        "\n--- Expert Responses ---",
    ]
    for resp in expert_responses:
        initial_prompt_parts.append(f"\nExpert: {resp['model_name']}")
        initial_prompt_parts.append(f"  Proposed Answer: {resp.get('answer', 'N/A')}")
        initial_prompt_parts.append(f"  Explanation: {resp.get('explanation', 'N/A')}")
    initial_prompt_parts.append("\n--- Your Task ---")
    initial_prompt_parts.append("Provide your synthesized decision in the following format EXACTLY:")
    initial_prompt_parts.append("Explanation: [Your consolidated medical reasoning and synthesis]")
    initial_prompt_parts.append("Answer: [A, B, C, or D]")

    initial_supervisor_prompt = "\n".join(initial_prompt_parts)
    raw_initial_response = generate_content(
        supervisor_model_id, initial_supervisor_prompt, temperature=0.3, max_output_tokens=500
    )
    prov_expl, prov_ans = extract_explanation_and_answer(raw_initial_response)

    if prov_ans is None:  # Fallback si el format falla
        state["error_messages"].append("SupervisorNode: Initial pass failed to produce formatted answer.")
        # Es podria intentar un fallback aquí o simplement passar a la reflexió amb None
        prov_expl = prov_expl or "Failed to generate initial explanation due to formatting issues."
        # No hi ha resposta provisional, la reflexió haurà de treballar amb això.

    # --- Segona Passada del Supervisor (Auto-Correcció/Reflexió) ---
    reflection_prompt_parts = [
        "You are a supervising medical expert performing a self-correction step.",
        "You previously analyzed expert responses and provided an initial synthesized answer and explanation for a multiple-choice question.",
        f"\nQuestion: {question_text}",
        f"Options:\nA: {options.get('A', 'N/A')}\nB: {options.get('B', 'N/A')}\nC: {options.get('C', 'N/A')}\nD: {options.get('D', 'N/A')}",
        "\n--- Original Expert Responses (for your review again) ---",
    ]
    for resp in expert_responses:  # Re-incloure per context
        reflection_prompt_parts.append(f"\nExpert: {resp['model_name']}")
        reflection_prompt_parts.append(f"  Proposed Answer: {resp.get('answer', 'N/A')}")
        reflection_prompt_parts.append(f"  Explanation: {resp.get('explanation', 'N/A')}")
    reflection_prompt_parts.append("\n--- Your Previous Synthesized Decision ---")
    reflection_prompt_parts.append(
        f"Your Initial Explanation: {prov_expl if prov_expl else 'Not generated or format error.'}"
    )
    reflection_prompt_parts.append(f"Your Initial Answer: {prov_ans if prov_ans else 'Not generated or format error.'}")
    reflection_prompt_parts.append("\n--- Self-Correction Task ---")
    reflection_prompt_parts.append("Critically review your initial decision. Consider:")
    reflection_prompt_parts.append("- Did you adequately address any disagreements among experts?")
    reflection_prompt_parts.append("- Is your explanation clear, concise, and medically accurate?")
    reflection_prompt_parts.append("- Could your reasoning be improved or clarified?")
    reflection_prompt_parts.append("- Based on this reflection, provide your FINAL refined explanation and answer.")
    reflection_prompt_parts.append("Output your FINAL decision in the following format EXACTLY:")
    reflection_prompt_parts.append("Explanation: [Your FINAL refined medical reasoning]")
    reflection_prompt_parts.append("Answer: [A, B, C, or D]")

    reflection_prompt = "\n".join(reflection_prompt_parts)
    raw_final_response = generate_content(
        supervisor_model_id, reflection_prompt, temperature=0.2, max_output_tokens=600
    )  # Temp més baixa per refinament
    final_expl, final_ans = extract_explanation_and_answer(raw_final_response)

    if final_ans is None:  # Si la reflexió també falla el format, utilitzar la provisional si existeix
        state["error_messages"].append("SupervisorNode: Reflection pass failed to produce formatted answer.")
        if prov_ans:  # Si la primera passada va ser bona, fer-la servir
            final_ans = prov_ans
            final_expl = prov_expl
        else:  # Les dues passades han fallat
            final_expl = final_expl or "Failed to generate final explanation due to formatting issues in both passes."

    return {
        "provisional_supervisor_explanation": prov_expl,
        "provisional_supervisor_answer": prov_ans,
        "supervisor_self_reflection_prompt": reflection_prompt,  # Opcional per a debugging
        "final_supervisor_explanation": final_expl,
        "final_supervisor_answer": final_ans,
    }


# --- FUNCIÓ DE BENCHMARKING ---
def benchmark_agentic_workflow(
    strategy_name: str,
    supervisor_llm_id: str,
    num_experts_to_select_param: int,
    num_questions_to_run: int = 10,
    k_shot_examples: int = 5,
    expert_temperature: float = 0.5,
):
    print(f"\n--- Benchmarking: {strategy_name} (Selecting {num_experts_to_select_param} experts) ---")

    medqa_full_test = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    medqa_train_data = (
        load_dataset("GBaker/MedQA-USMLE-4-options", split="train").shuffle(seed=42).select(range(k_shot_examples))
    )

    # Preparar prompts de k-shot
    system_prompt_text = (
        "You are a medical expert answering multiple-choice questions. "
        "Always output EXACTLY in this format, nothing more:\n"
        "Explanation: [Your brief but precise medical reasoning, no code or additional text]\n"
        "Answer: [A or B or C or D]\n"
    )
    explanations_for_k_shot = [
        "A 35-year-old woman presents with various complaints and laboratory testing reveals the presence of anti-centromere antibodies. These antibodies are strongly associated with limited systemic sclerosis (also known as CREST syndrome), which includes Calcinosis, Raynaud's phenomenon, Esophageal dysmotility, Sclerodactyly, and Telangiectasia. Given this context, let's evaluate the given symptoms and signs:\nA: Pallor, cyanosis, and erythema of the hands - These are classic signs of Raynaud's phenomenon, commonly seen in limited systemic sclerosis.\nB: Blanching vascular abnormalities - These are also indicative of Raynaud's phenomenon, expected in this condition.\nC: Hypercoagulable state - Not typically a feature of limited systemic sclerosis; it is more associated with other connective tissue diseases or specific genetic disorders.\nD: Heartburn and regurgitation - These symptoms are consistent with esophageal dysmotility, a common feature of limited systemic sclerosis.",
        "The child's symptoms and physical examination findings—including slurred speech, frequent falls, pes cavus (high-arched feet), hammer toes, and kyphoscoliosis (curvature of the spine)—suggest a progressive neurological disorder. These signs are indicative of Friedreich's ataxia, a trinucleotide repeat disease.\nFriedreich's ataxia is caused by an expansion of the GAA trinucleotide repeat in the frataxin (FXN) gene. This condition typically presents with progressive ataxia, leading to difficulties with speech and coordination, as well as the skeletal abnormalities mentioned.\n",
        "The patient's presentation, including fever, headache, seizures, and altered behavior, along with the MRI findings of edema and hemorrhage in the left temporal lobe, strongly suggests herpes simplex encephalitis (HSE). HSE is a viral infection that primarily affects the temporal lobes and can cause significant brain edema.\nThe primary mechanism of edema in herpes simplex encephalitis is the breakdown of endothelial tight junctions. This disruption of the blood-brain barrier allows fluid to leak into the brain parenchyma, leading to vasogenic edema. This process is driven by the inflammatory response to the viral infection, which damages the endothelial cells and compromises the integrity of the blood-brain barrier.",
        "The patient's presentation includes shortness of breath, cough, severe lower limb edema, signs of right heart failure (jugular engorgement, hepatomegaly, hepatojugular reflux), and findings suggestive of pulmonary fibrosis. The physical examination and diagnostic tests (CT and echocardiogram) reveal right heart failure and severe pulmonary fibrosis. Cor pulmonale is a condition where right heart failure is caused by a primary lung disorder, typically chronic pulmonary hypertension. In this case, the severe pulmonary fibrosis is likely the underlying cause of the pulmonary hypertension, leading to cor pulmonale.",
        "Based on the information provided, the child's recurrent abdominal pain that occurs only at school, with no symptoms at home, and no abnormalities on physical examination or laboratory tests, suggests a functional cause. The child's symptoms are consistent with a functional abdominal pain disorder, possibly related to school avoidance or a behavioral component. Given that the child denies functional pain and there are no alarm symptoms (such as blood in the stool, weight loss, or nighttime symptoms), the next step in management should focus on addressing the potential psychological or behavioral factors contributing to the abdominal pain.",
    ][:k_shot_examples]

    shot_prompt_text_parts = []
    for i, train_ex in enumerate(medqa_train_data):
        expl = explanations_for_k_shot[i] if i < len(explanations_for_k_shot) else "Sample explanation."
        shot_prompt_text_parts.append(
            f"Question: {train_ex['question']}\n"
            f"A: {train_ex['options']['A']}\nB: {train_ex['options']['B']}\nC: {train_ex['options']['C']}\nD: {train_ex['options']['D']}\n"
            f"Explanation: {expl}\nAnswer: {train_ex['answer_idx']}\n\n"
        )
    shot_prompt_text = "".join(shot_prompt_text_parts)

    test_set = medqa_full_test.shuffle(seed=42).select(range(num_questions_to_run))
    results_accumulator = []
    num_correct = 0
    num_no_answer = 0

    # Definir el graf
    workflow_builder = StateGraph(AgenticWorkflowState)
    workflow_builder.add_node("router", dynamic_expert_router_node)

    # Usar partial per passar arguments fixos als nodes
    consult_node_with_args = partial(
        expert_consultation_node,
        system_prompt_template=system_prompt_text,
        shot_prompt_template=shot_prompt_text,
        temperature=expert_temperature,
    )
    workflow_builder.add_node("expert_consultation", consult_node_with_args)

    supervisor_node_with_args = partial(synthesizing_supervisor_node, supervisor_model_id=supervisor_llm_id)
    workflow_builder.add_node("supervisor_synthesis", supervisor_node_with_args)

    workflow_builder.set_entry_point("router")
    workflow_builder.add_edge("router", "expert_consultation")
    workflow_builder.add_edge("expert_consultation", "supervisor_synthesis")
    workflow_builder.add_edge("supervisor_synthesis", END)

    compiled_agent = workflow_builder.compile()

    for test_idx, test_item in tqdm(enumerate(test_set), total=len(test_set), desc=f"Processing {strategy_name}"):
        initial_state_for_run: AgenticWorkflowState = {
            "question_text": test_item["question"],
            "options": test_item["options"],
            "correct_answer_idx": test_item["answer_idx"],
            "available_experts": EXPERT_DEFINITIONS,
            "num_experts_to_select": num_experts_to_select_param,
            "selected_expert_names": [],
            "routing_rationale": None,
            "expert_responses": [],
            "provisional_supervisor_explanation": None,
            "provisional_supervisor_answer": None,
            "supervisor_self_reflection_prompt": None,
            "final_supervisor_explanation": None,
            "final_supervisor_answer": None,
            "error_messages": [],
            "node_trace": [],
        }

        final_run_state = None
        try:
            final_run_state = compiled_agent.invoke(initial_state_for_run)
            agent_answer = final_run_state.get("final_supervisor_answer")

            if agent_answer is None:
                num_no_answer += 1
                is_item_correct = False
            else:
                is_item_correct = agent_answer == test_item["answer_idx"]
                if is_item_correct:
                    num_correct += 1

            results_accumulator.append(
                {
                    "q_idx": test_idx,
                    "model_answer": agent_answer,
                    "is_correct": is_item_correct,
                    "routing": final_run_state.get("routing_rationale"),
                    "selected_experts_count": len(final_run_state.get("selected_expert_names", [])),
                    "expert_answers": [er.get("answer") for er in final_run_state.get("expert_responses", [])],
                    "errors": final_run_state.get("error_messages"),
                }
            )

        except Exception as e:
            num_no_answer += 1
            results_accumulator.append(
                {
                    "q_idx": test_idx,
                    "model_answer": "ERROR_IN_WORKFLOW",
                    "is_correct": False,
                    "errors": [str(e)] + (final_run_state.get("error_messages", []) if final_run_state else []),
                }
            )

    valid_attempts = num_questions_to_run - num_no_answer
    accuracy_val = (num_correct / valid_attempts * 100) if valid_attempts > 0 else 0

    print(f"Strategy: {strategy_name}")
    print(f"  Accuracy: {accuracy_val:.1f}% ({num_correct}/{valid_attempts})")
    print(f"  No answer/Error: {num_no_answer}/{num_questions_to_run}")

    return {
        "model_name": strategy_name,
        "accuracy": accuracy_val,
        "correct": num_correct,
        "no_answer": num_no_answer,
        "total": num_questions_to_run,
        "responses": results_accumulator,
    }


# --- PLOT RESULTS (Simplificat) ---
def plot_benchmark_results(benchmark_outputs):
    if not benchmark_outputs:
        return
    names = [f"Top {x + 1} Experts" for x in range(len(benchmark_outputs))]
    accs = [r["accuracy"] for r in benchmark_outputs]

    # Create vertical bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, accs, color="skyblue", edgecolor="black")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Agentic Workflow Benchmark")
    ax.set_ylim(0, 100)

    # Add labels above bars
    for bar in bars:
        height = bar.get_height()
        i = bars.index(bar)
        label = f"{accs[i]:.1f}%"
        ax.text(bar.get_x() + bar.get_width() / 2, height + 1, label, ha="center", va="bottom", rotation=0)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("agentic_workflow_benchmark.png")
    print("\nPlot saved to agentic_workflow_benchmark.png")


# --- EXECUCIÓ PRINCIPAL ---
if __name__ == "__main__":
    # Paràmetres de configuració per al benchmark
    NUM_QUESTIONS = 1273  # Nombre de preguntes per provar (baix per rapidesa)
    NUM_AGENTS = 5  # Nombre d'experts a seleccionar per cada pregunta

    # ID del model LLM que actuarà com a supervisor
    # Pots triar qualsevol dels teus models afinats o un model general potent si en tens.
    SUPERVISOR_MODEL_ID = EXPERT_DEFINITIONS["Medicina General"]

    # Carregar el model d'embeddings per l'expert router
    _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    AGENTS_EMBEDDING_DIR = "agents_embeddings"
    _agents_embeddings = {}
    for fn in os.listdir(AGENTS_EMBEDDING_DIR):
        if fn.endswith(".npy"):
            agent_name = fn[:-4]  # retalla ".npy"
            emb = np.load(os.path.join(AGENTS_EMBEDDING_DIR, fn))
            _agents_embeddings[agent_name] = emb.reshape(1, -1)

    all_strategy_results = []

    # Provar amb diferents nombres d'experts seleccionats pel router
    for num_experts_routed in range(1, NUM_AGENTS + 1):
        strategy_run_name = f"Dynamic Routing (Top {num_experts_routed} Experts) + Self-Correcting Supervisor"

        current_result = benchmark_agentic_workflow(
            strategy_name=strategy_run_name,
            supervisor_llm_id=SUPERVISOR_MODEL_ID,
            num_experts_to_select_param=num_experts_routed,
            num_questions_to_run=NUM_QUESTIONS,
        )
        all_strategy_results.append(current_result)

        # Get the test set for reference in the output
        test_set = (
            load_dataset("GBaker/MedQA-USMLE-4-options", split="test").shuffle(seed=42).select(range(NUM_QUESTIONS))
        )

    # Mostrar resultats finals i gràfic
    if all_strategy_results:
        plot_benchmark_results(all_strategy_results)
        print("\n\n===== FINAL SUMMARY OF AGENTIC WORKFLOWS =====")
        header = f"{'Strategy':<60} | {'Accuracy':<10} | {'Correct':<10} | {'No Answer':<10} | {'Valid Total':<12}"
        print(header)
        print("-" * len(header))
        for r_item in sorted(all_strategy_results, key=lambda x: x["accuracy"], reverse=True):
            valid_total_calc = r_item["total"] - r_item["no_answer"]
            correct_disp = f"{r_item['correct']}/{valid_total_calc}" if valid_total_calc > 0 else "N/A"
            accuracy_disp = f"{r_item['accuracy']:.1f}%" if valid_total_calc > 0 else "N/A"
            print(
                f"{r_item['model_name']:<60} | {accuracy_disp:<10} | "
                f"{correct_disp:<10} | {r_item['no_answer']:<10} | {valid_total_calc:<12}"
            )
    else:
        print("No benchmark results were generated.")
