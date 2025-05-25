import os
from typing import TypedDict, List, Dict, Optional, Any
import numpy as np
import pandas as pd
from langgraph.graph import StateGraph, END
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from funcions_auxiliars import generate_content, _call_single_expert_llm

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
    diversity_option: str
    is_benchmark_mode: bool
    selected_expert_names: List[str]
    initial_expert_responses: List[Dict[str, Any]]
    supervisor_critique: Optional[str]
    revised_expert_outputs: List[Dict[str, Any]]
    final_synthesis: Optional[str]
    expert_temperature: float
    system_prompt: Optional[str]
    benchmark_shot_prompt: Optional[str]


EXPERT_DEFINITIONS = {
    "Medicina General": "tunedModels/medicinageneralcsv-q4i0ydc9l1uvxbzsxmii8",
    "Ciències Bàsiques": "tunedModels/ciencies-basiques-2-pfg4bpafqcay88df2kr8",
    "Patologia i Farmacologia": "tunedModels/patologia-farmacologia-2-8iy2ixmy5bluqzw",
    "Cirurgia": "tunedModels/cirurgia-2-2c1cy8nkr5ca5mui15tu4wtlpapp8",
    "Pediatria i Ginecologia": "tunedModels/pediatria-ginecologia-2-ss7f3iy509x7x43h",
}
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

_agent_similarity_matrix = pd.read_csv("dissimilarity_matrix.csv", index_col=0, header=0).to_dict(orient="index")


# --- LANGGRAPH NODES ---
def dynamic_expert_router_node(state: AgenticWorkflowState) -> Dict[str, any]:
    """
    Dynamic routing node to select experts based on question relevance and diversity.
    If diversity is set to "Baixa", it selects the most relevant expert multiple times.
    If "Mitjana", it selects the top N experts.
    If "Alta", it selects the most relevant AND diverse experts.
    """
    question_text = state["question_text"]
    num_to_select = state["num_experts_to_select"]
    diversity_option = state["diversity_option"]

    available_routable_agents = list(_agents_textbook_embeddings.keys())
    q_emb = _embedding_model_sbert.encode([question_text], convert_to_numpy=True)

    scores_to_question = []
    for agent_name in available_routable_agents:
        textbook_emb = _agents_textbook_embeddings[agent_name]
        sim = float(cosine_similarity(q_emb, textbook_emb)[0][0])
        scores_to_question.append({"name": agent_name, "score": sim})

    scores_to_question.sort(reverse=True, key=lambda x: x["score"])

    num_to_select = max(1, min(num_to_select, len(scores_to_question)))
    selected_experts = []

    if diversity_option == "Baixa":
        most_relevant_agent_name = scores_to_question[0]["name"]
        selected_experts = [most_relevant_agent_name] * num_to_select

    elif diversity_option == "Mitjana":
        selected_experts = [item["name"] for item in scores_to_question[:num_to_select]]

    elif diversity_option == "Alta":
        # Pool of agents sorted by relevance to the question
        relevant_pool = [item["name"] for item in scores_to_question]

        # 1. Select the most relevant agent
        if relevant_pool:
            agent1 = relevant_pool.pop(0)
            selected_experts.append(agent1)

        # 2. Select the most different agent from the first one (if needed and available)
        if num_to_select >= 2 and relevant_pool:
            best_agent2 = None
            min_similarity_to_agent1 = float("inf")

            for candidate_agent in relevant_pool:
                # Make sure agent1 and candidate_agent are in the similarity matrix
                if agent1 in _agent_similarity_matrix and candidate_agent in _agent_similarity_matrix[agent1]:
                    sim = _agent_similarity_matrix[agent1][candidate_agent]
                    if sim < min_similarity_to_agent1:
                        min_similarity_to_agent1 = sim
                        best_agent2 = candidate_agent

            if best_agent2:
                selected_experts.append(best_agent2)
                relevant_pool.remove(best_agent2)  # Remove from pool for next step

        # 3. Fill remaining slots with the next most relevant from the (updated) relevant_pool
        num_still_needed = num_to_select - len(selected_experts)
        if num_still_needed > 0 and relevant_pool:
            selected_experts.extend(relevant_pool[:num_still_needed])

    # print(f"Selected experts: {selected_experts} based on diversity option '{diversity_option}'")
    return {"selected_expert_names": selected_experts}


def initial_expert_consultation_node(state: AgenticWorkflowState) -> Dict[str, Any]:
    # print("Starting initial expert consultation...")
    outputs = []
    selected_names = state.get("selected_expert_names", [])

    for i, expert_name in enumerate(selected_names):
        expert_model_id = state["available_experts"].get(expert_name)
        response_dict = _call_single_expert_llm(
            expert_model_id=expert_model_id,
            question_text=state["question_text"],
            temperature=state["expert_temperature"],
            is_benchmark_mode=state["is_benchmark_mode"],
            options=state.get("options"),
            system_prompt=state.get("system_prompt"),
            benchmark_shot_prompt=state.get("benchmark_shot_prompt"),
            critique_to_include=None,  # No critique for initial pass
        )

        outputs.append(
            {
                "expert_display_id": f"Expert {i+1}",  # Anonymized for supervisor
                "conclusion": response_dict.get("conclusion", response_dict.get("answer", "N/A")),
                "explanation": response_dict.get("explanation", "N/A"),
                "original_model_name_for_revised_pass": expert_name,  # Needed for revised pass
            }
        )

    # print(f"Initial expert responses collected: {outputs}")
    return {"initial_expert_responses": outputs}


def critique_supervisor_node(state: AgenticWorkflowState) -> Dict[str, str]:
    # print("Starting supervisor critique of initial expert outputs...")
    initial_outputs = state.get("initial_expert_responses", [])
    question_text = state["question_text"]

    if not initial_outputs:
        print("No initial expert outputs to critique.")
        return {"supervisor_critique": "No initial expert outputs to critique."}

    prompt_parts = [
        "You are a Chief Medical Consultant AI. Your task is to critically review a set of anonymized analyses from various medical experts regarding a clinical case.",
        "Focus on identifying potential inconsistencies, areas needing more depth, missed perspectives, or assumptions that should be challenged.",
        "Provide a single, consolidated, constructive critique that can be given back to these experts to help them refine their analyses. Do not try to answer the question yourself in this step.",
        f"\n**Clinical Case/Question:**\n{question_text}",
        "\n\n--- Initial Anonymized Expert Analyses ---",
    ]
    for i, resp in enumerate(initial_outputs):
        # Use expert_display_id for anonymity
        prompt_parts.append(
            f"\n**{resp['expert_display_id']}**:\n"
            f"  Conclusion/Answer: {resp.get('conclusion', 'N/A')}\n"
            f"  Explanation: {resp.get('explanation', 'N/A').strip()}"
        )
    prompt_parts.append(
        "\n\n--- YOUR CONSOLIDATED CRITIQUE ---"
        "\nProvide a constructive critique (around 100-150 words) for the experts:"
    )

    # Temperature for critique can be higher to encourage more diverse thought
    critique_temp = 0.7 if not state["is_benchmark_mode"] else 0.5
    critique_text = generate_content(
        SUPERVISOR_MODEL_ID,  # Use the same supervisor model
        "\n".join(prompt_parts),
        temperature=critique_temp,
        max_output_tokens=250,
    )

    # print(f"Supervisor critique generated: {critique_text.strip()}")
    return {"supervisor_critique": critique_text.strip() or "No specific critique generated."}


def revised_expert_consultation_node(state: AgenticWorkflowState) -> Dict[str, Any]:
    # print("Starting revised expert consultation based on supervisor critique...")
    revised_outputs = []
    critique = state.get("supervisor_critique")

    # Iterate based on the original_model_name stored in initial_expert_responses
    # This ensures we re-consult the same experts that were initially selected.
    initial_expert_details = state.get("initial_expert_responses", [])

    if (
        not critique
        or "No initial expert outputs to critique." in critique
        or "No specific critique generated." in critique
    ):
        print("No actionable critique provided, revised outputs will mirror initial outputs if not empty.")
        return {"revised_expert_outputs": state.get("initial_expert_responses", [])}

    for i, expert_detail in enumerate(initial_expert_details):
        expert_name = expert_detail["original_model_name_for_revised_pass"]
        expert_model_id = state["available_experts"].get(expert_name)

        response_dict = _call_single_expert_llm(
            expert_model_id=expert_model_id,
            question_text=state["question_text"],
            temperature=state["expert_temperature"],  # Can use same or slightly different temp
            is_benchmark_mode=state["is_benchmark_mode"],
            options=state.get("options"),
            system_prompt=state.get("system_prompt"),
            benchmark_shot_prompt=state.get("benchmark_shot_prompt"),
            critique_to_include=critique,
        )

        revised_outputs.append(
            {
                "expert_display_id": expert_detail["expert_display_id"],
                "conclusion": response_dict.get("conclusion", response_dict.get("answer", "N/A")),
                "explanation": response_dict.get("explanation", "N/A"),
                "original_model_name_for_revised_pass": expert_name,
            }
        )
    # print(f"Revised expert responses collected: {revised_outputs}")
    return {"revised_expert_outputs": revised_outputs}


def synthesis_supervisor_node(state: AgenticWorkflowState) -> Dict[str, str]:
    # print("Starting final synthesis of expert outputs...")
    # Prioritize revised outputs, fall back to initial if revision didn't happen or was skipped
    expert_outputs_to_synthesize = state.get("revised_expert_outputs", [])
    if not expert_outputs_to_synthesize:  # Check if list is empty
        expert_outputs_to_synthesize = state.get("initial_expert_responses", [])

    question_text = state["question_text"]

    if not expert_outputs_to_synthesize:
        print("No expert outputs available for final synthesis.")
        return {"final_synthesis": "No expert outputs available for final synthesis."}

    prompt_parts = [
        "You are a Chief Medical Consultant AI. Your task is to synthesize a final, comprehensive, and standardized report based on refined (or initial, if no refinement occurred) analyses from several anonymized medical experts.",
        "Focus on creating a coherent narrative that integrates the key insights, addresses the main question, and provides a clear conclusion. If there are differing opinions, acknowledge them if significant but aim for a dominant supported viewpoint.",
        f"\n**Clinical Case/Question:**\n{question_text}",
    ]
    if state.get("is_benchmark_mode") and state.get("options"):
        options = state["options"]
        prompt_parts.append(
            f"\n**Options:**\nA: {options.get('A', 'N/A')}\nB: {options.get('B', 'N/A')}"
            f"\nC: {options.get('C', 'N/A')}\nD: {options.get('D', 'N/A')}"
        )
    prompt_parts.append("\n\n--- Anonymized Expert Analyses for Synthesis ---")

    for resp in expert_outputs_to_synthesize:
        prompt_parts.append(
            f"\n**{resp['expert_display_id']}**:\n"
            f"  Conclusion/Answer: {resp.get('conclusion', 'N/A')}\n"
            f"  Explanation: {resp.get('explanation', 'N/A').strip()}"
        )

    expected_format_description = (
        "Provide your synthesis as a single, well-structured text. "
        "If in benchmark mode and options are provided, ensure your final answer explicitly states the chosen option (e.g., 'Final Answer: A')."
        if state.get("is_benchmark_mode")
        else "Provide your synthesis as a single, well-structured text. Start with a summary of findings and end with a clear overall conclusion."
    )
    prompt_parts.append(
        f"\n\n--- YOUR FINAL SYNTHESIZED REPORT ---" f"\n{expected_format_description}" "\nSynthesized Report:"
    )

    synthesis_temp = 0.5 if not state["is_benchmark_mode"] else 0.2
    synthesized_text = generate_content(
        SUPERVISOR_MODEL_ID,
        "\n".join(prompt_parts),
        temperature=synthesis_temp,
        max_output_tokens=700 if not state["is_benchmark_mode"] else 500,
    )

    # For benchmark mode, you might want to re-parse Explanation and Answer here if needed
    # For UI mode, the full text is likely the desired synthesis.
    # print(f"Final synthesis generated: {synthesized_text.strip()}")
    return {"final_synthesis": synthesized_text.strip() or "Synthesis could not be generated."}


def create_compiled_agent():
    workflow = StateGraph(AgenticWorkflowState)

    workflow.add_node("router", dynamic_expert_router_node)
    workflow.add_node("initial_expert_consultation", initial_expert_consultation_node)
    workflow.add_node("critique_supervisor", critique_supervisor_node)
    workflow.add_node("revised_expert_consultation", revised_expert_consultation_node)
    workflow.add_node("synthesis_supervisor", synthesis_supervisor_node)

    workflow.set_entry_point("router")
    workflow.add_edge("router", "initial_expert_consultation")
    workflow.add_edge("initial_expert_consultation", "critique_supervisor")
    workflow.add_edge("critique_supervisor", "revised_expert_consultation")
    workflow.add_edge("revised_expert_consultation", "synthesis_supervisor")
    workflow.add_edge("synthesis_supervisor", END)

    return workflow.compile()
