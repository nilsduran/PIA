import os
from functools import partial
from typing import TypedDict, List, Dict, Optional, Any
import numpy as np
from langgraph.graph import StateGraph, END
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from funcions_auxiliars import generate_content, extract_explanation_and_answer

# --- CONFIGURATION ---
AGENTS_EMBEDDINGS_DIR_PATH = "agents_embeddings"
SUPERVISOR_MODEL_ID = "gemini-2.5-flash-preview-05-20"


# --- LANGGRAPH STATE DEFINITION ---
class AgenticWorkflowState(TypedDict):
    question_text: str
    options: Optional[Dict[str, str]]
    correct_answer_idx: Optional[str]
    available_experts: Dict[str, str]
    num_experts_to_select: int
    selected_expert_names: List[str]
    expert_responses: List[Dict[str, Any]]
    provisional_supervisor_explanation: Optional[str]
    provisional_supervisor_answer: Optional[str]
    final_supervisor_explanation: Optional[str]
    final_supervisor_answer: Optional[str]


# --- EXPERT DEFINITIONS (Specialists) ---
EXPERT_DEFINITIONS = {
    "Medicina General": "tunedModels/medicinageneralcsv-q4i0ydc9l1uvxbzsxmii8",
    "Ciències Bàsiques": "tunedModels/ciencies-basiques-2-pfg4bpafqcay88df2kr8",
    "Patologia i Farmacologia": "tunedModels/patologia-farmacologia-2-8iy2ixmy5bluqzw",
    "Cirurgia": "tunedModels/cirurgia-2-2c1cy8nkr5ca5mui15tu4wtlpapp8",
    "Pediatria i Ginecologia": "tunedModels/pediatria-ginecologia-2-ss7f3iy509x7x43h",
}
EXPERT_DISPLAY_NAMES = list(EXPERT_DEFINITIONS.keys())  # For UI or other logic

AGENT_EMBEDDING_FILENAME_MAP = {
    "Ciències Bàsiques": "Ciències_Bàsiques",
    "Medicina General": "Medicina_General",
    "Patologia i Farmacologia": "Patologia_Farmacologia",
    "Cirurgia": "Cirurgia",
    "Pediatria i Ginecologia": "Pediatria_Ginecologia",
}

# --- SBERT Model & Textbook Embeddings Loading ---
_embedding_model_sbert = SentenceTransformer("all-MiniLM-L6-v2")
_agents_textbook_embeddings = {}

if os.path.isdir(AGENTS_EMBEDDINGS_DIR_PATH) and _embedding_model_sbert:
    for display_name, filename_stem in AGENT_EMBEDDING_FILENAME_MAP.items():
        file_path = os.path.join(AGENTS_EMBEDDINGS_DIR_PATH, f"{filename_stem}.npy")
        emb = np.load(file_path)
        _agents_textbook_embeddings[display_name] = emb.reshape(1, -1) if emb.ndim == 1 else emb
else:
    raise FileNotFoundError(
        f"Textbook embeddings directory '{AGENTS_EMBEDDINGS_DIR_PATH}' not found or SBERT model not loaded."
    )


# --- LANGGRAPH NODES ---
def dynamic_expert_router_node(state: AgenticWorkflowState) -> Dict[str, any]:
    q_emb = _embedding_model_sbert.encode([state["question_text"]], convert_to_numpy=True)
    scores = []
    for agent_name, textbook_emb in _agents_textbook_embeddings.items():
        sim = float(cosine_similarity(q_emb, textbook_emb)[0][0])
        scores.append((sim, agent_name))

    scores.sort(reverse=True, key=lambda x: x[0])
    selected = [name for _, name in scores[: state["num_experts_to_select"]]]

    return {"selected_expert_names": selected}


def core_expert_consultation_node(
    state: AgenticWorkflowState,
    # Prompts and temperature will be passed via partial from the calling script (benchmark or UI)
    system_prompt: str,
    shot_prompt: Optional[str],  # Only for benchmark
    temperature: float,
    is_benchmark_mode: bool,
):
    expert_responses = []
    for expert_name in state.get("selected_expert_names", []):
        expert_model_id = state["available_experts"].get(expert_name)
        if not expert_model_id:
            continue

        if is_benchmark_mode:
            prompt = (
                system_prompt + "\n" + (shot_prompt or "") + f"Question: {state['question_text']}\n"
                f"A: {state['options'].get('A', 'N/A')}\nB: {state['options'].get('B', 'N/A')}\n"
                f"C: {state['options'].get('C', 'N/A')}\nD: {state['options'].get('D', 'N/A')}\n"
                f"Explanation:"
            )
            max_tok, retry_max_tok = 300, 500
        else:  # UI/Conversational mode
            prompt = (
                f"{system_prompt}\n\nYou are an AI expert in {expert_name}.\n"
                f"Analyze the following case/question from your specialty perspective:\n\n"
                f"Case/Question: {state['question_text']}\n\n"
                f"Provide a concise analysis (around 80-120 words), including key insights and potential considerations. Format:\n"
                f"Explanation: [Your analysis]\nConclusion: [Your main conclusion/summary]"
            )
            max_tok, retry_max_tok = 250, 350

        raw_response = generate_content(expert_model_id, prompt, temperature=temperature, max_output_tokens=max_tok)
        explanation, answer_or_conclusion = extract_explanation_and_answer(raw_response)

        if answer_or_conclusion is None and is_benchmark_mode:
            raw_response = generate_content(
                expert_model_id, prompt, temperature=max(0.1, temperature - 0.2), max_output_tokens=retry_max_tok
            )
            explanation, answer_or_conclusion = extract_explanation_and_answer(raw_response)

        response_key = "answer" if is_benchmark_mode else "conclusion"
        expert_responses.append(
            {
                "model_name": expert_name,
                response_key: answer_or_conclusion or ("No answer." if is_benchmark_mode else "No conclusion."),
                "explanation": explanation or "No explanation.",
            }
        )

    return {"expert_responses": expert_responses}


def core_synthesizing_supervisor_node(state: AgenticWorkflowState, is_benchmark_mode: bool):
    expert_responses = state.get("expert_responses", [])
    question_text = state["question_text"]

    if not expert_responses:
        no_resp = "No specialist input for supervision."
        return {
            "provisional_supervisor_explanation": no_resp,
            "provisional_supervisor_answer": None,
            "final_supervisor_explanation": no_resp,
            "final_supervisor_answer": None,
        }

    temp_initial = 0.4 if is_benchmark_mode else 1
    temp_final = 0.2 if is_benchmark_mode else 0.7
    max_tok_initial = 1500 if is_benchmark_mode else 2000
    max_tok_final = 4000 if is_benchmark_mode else 8000

    answer_key_for_experts = "Answer" if is_benchmark_mode else "Conclusion"
    expected_answer_format_supervisor = "[A or B or C or D]" if is_benchmark_mode else "[Your primary conclusion]"
    options_text_supervisor = ""
    if is_benchmark_mode and state.get("options"):
        options = state["options"]
        options_text_supervisor = (
            f"\n**Options:**\nA: {options.get('A', 'N/A')}\nB: {options.get('B', 'N/A')}"
            f"\nC: {options.get('C', 'N/A')}\nD: {options.get('D', 'N/A')}"
        )

    # --- Initial Synthesis Pass ---
    initial_prompt_parts = [
        "You are a Chief Medical Consultant AI. Synthesize specialist AI inputs for the given "
        "clinical case/question.",
        f"Review all inputs. Provide a consolidated analysis and a primary "
        f"{'answer (A,B,C,D)' if is_benchmark_mode else 'conclusion'}.",
        f"\n**Clinical Case/Question:**\n{question_text}",
        options_text_supervisor,
        "\n\n--- Specialist AI Consultations ---",
    ]
    for resp in expert_responses:
        initial_prompt_parts.append(
            f"\n**{resp['model_name']}** (Proposed {answer_key_for_experts}: "
            f"{resp.get(answer_key_for_experts, 'N/A')}):"
        )
        initial_prompt_parts.append(f"  Explanation: {resp.get('explanation', 'N/A').strip()}")
    initial_prompt_parts.append(
        f"\n\n--- YOUR INITIAL SYNTHESIS ({'MedQA Format' if is_benchmark_mode else 'Report Format'}) ---"
    )
    initial_prompt_parts.append("STRICT Format: Explanation: [Your reasoning] Answer: [Your choice/conclusion]")
    initial_prompt_parts.append("Explanation: [Synthesized reasoning, 150-250 words]")
    initial_prompt_parts.append(f"Answer: {expected_answer_format_supervisor}")

    raw_initial = generate_content(SUPERVISOR_MODEL_ID, "\n".join(initial_prompt_parts), temp_initial, max_tok_initial)
    prov_expl, prov_ans = extract_explanation_and_answer(raw_initial)

    if is_benchmark_mode and (prov_ans is None or prov_ans not in ["A", "B", "C", "D"]):
        prov_expl = prov_expl or "Initial explanation parse error (benchmark)."

    # --- Self-Correction Pass ---
    reflection_prompt_parts = [
        "You are a Chief Medical Consultant AI performing self-correction.",
        "Previously, you synthesized specialist inputs. Review your work.",
        f"\n**Clinical Case/Question:**\n{question_text}",
        options_text_supervisor,
        "\n\n--- Original Specialist Inputs ---",
    ]

    for resp in expert_responses:
        reflection_prompt_parts.append(
            f"\n**{resp['model_name']}** (Proposed {answer_key_for_experts}: "
            f"{resp.get(answer_key_for_experts, 'N/A')}):"
        )
        reflection_prompt_parts.append(f"  Explanation: {resp.get('explanation', 'N/A').strip()}")
    reflection_prompt_parts.append("\n\n--- Your Previous Decision ---")
    reflection_prompt_parts.append(f"Initial Explanation:\n{prov_expl or 'Not generated.'}")
    reflection_prompt_parts.append(f"Initial Answer/Conclusion: {prov_ans or 'Not generated.'}")
    reflection_prompt_parts.append(
        f"\n\n--- SELF-CORRECTION TASK ({'MedQA Format' if is_benchmark_mode else 'Report Format'}) ---"
    )
    reflection_prompt_parts.append(
        "Critically re-evaluate. Ensure your FINAL decision is most accurate and well-supported."
    )
    reflection_prompt_parts.append("STRICT Format: Explanation: [Your reasoning] Answer: [Your choice/conclusion]")
    reflection_prompt_parts.append("Explanation: [FINAL refined reasoning, 200-300 words]")
    reflection_prompt_parts.append(f"Answer: {expected_answer_format_supervisor}")

    raw_final = generate_content(SUPERVISOR_MODEL_ID, "\n".join(reflection_prompt_parts), temp_final, max_tok_final)
    final_expl, final_ans = extract_explanation_and_answer(raw_final)

    if is_benchmark_mode and (final_ans is None or final_ans not in ["A", "B", "C", "D"]):
        if prov_ans and prov_ans in ["A", "B", "C", "D"]:
            final_ans, final_expl = prov_ans, (final_expl or prov_expl)

    return {
        "provisional_supervisor_explanation": prov_expl or "Provisional explanation error.",
        "provisional_supervisor_answer": prov_ans,
        "final_supervisor_explanation": final_expl or "Final explanation error.",
        "final_supervisor_answer": final_ans,
    }


# --- Function to create and compile the graph ---
def create_compiled_agent(
    is_benchmark_mode: bool,
    # Benchmark specific params
    benchmark_expert_system_prompt: Optional[str] = None,
    benchmark_expert_shot_prompt: Optional[str] = None,
    benchmark_expert_temp: float = 0.4,  # Default benchmark expert temp
    # UI specific params
    ui_expert_system_prompt: Optional[str] = None,  # More general prompt for UI experts
    ui_expert_temp: float = 1,  # Default UI expert temp
):
    workflow = StateGraph(AgenticWorkflowState)
    workflow.add_node("router", dynamic_expert_router_node)

    if is_benchmark_mode:
        expert_consult_with_args = partial(
            core_expert_consultation_node,
            system_prompt=benchmark_expert_system_prompt,
            shot_prompt=benchmark_expert_shot_prompt,
            temperature=benchmark_expert_temp,
            is_benchmark_mode=True,
        )
    else:  # UI Mode
        expert_consult_with_args = partial(
            core_expert_consultation_node,
            system_prompt=ui_expert_system_prompt,
            shot_prompt=None,  # No k-shot for general UI queries
            temperature=ui_expert_temp,
            is_benchmark_mode=False,
        )
    workflow.add_node("expert_consultation", expert_consult_with_args)

    supervisor_with_args = partial(core_synthesizing_supervisor_node, is_benchmark_mode=is_benchmark_mode)
    workflow.add_node("supervisor_synthesis", supervisor_with_args)

    workflow.set_entry_point("router")
    workflow.add_edge("router", "expert_consultation")
    workflow.add_edge("expert_consultation", "supervisor_synthesis")
    workflow.add_edge("supervisor_synthesis", END)

    return workflow.compile()
