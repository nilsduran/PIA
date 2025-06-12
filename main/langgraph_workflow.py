from typing import TypedDict, List, Dict, Optional, Any
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from langgraph.graph import StateGraph, END
from scripts.embeddings_loader import get_sbert_model, get_agents_textbook_embeddings
from scripts.funcions_auxiliars import generate_content, _call_single_expert_llm, extract_explanation_and_answer

# --- CONFIGURATION ---
# AGENTS_EMBEDDINGS_DIR_PATH = "agents_embeddings"
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
EXPERT_DEFINITIONS_REVERSED = {v: k for k, v in EXPERT_DEFINITIONS.items()}

_agent_similarity_matrix = pd.read_csv("data/dissimilarity_matrix.csv", index_col=0, header=0).to_dict(orient="index")


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

    # Obtenir el model i els embeddings a través del loader
    _embedding_model_sbert = get_sbert_model()
    _agents_textbook_embeddings = get_agents_textbook_embeddings()

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
                "explanation": response_dict.get("explanation", "N/A"),
                "answer": response_dict.get("answer", "N/A"),
                "original_model_name_for_revised_pass": expert_name,  # Needed for revised pass
            }
        )

    # print(f"Initial expert responses collected: {outputs}")
    return {"initial_expert_responses": outputs}


def critique_supervisor_node(state: AgenticWorkflowState) -> Dict[str, str]:
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
            f"  Explanation: {resp.get('explanation', 'N/A').strip()}"
            f"  Answer: {resp.get('answer', 'N/A')}\n"
        )
    prompt_parts.append(
        "\n\n--- YOUR CONSOLIDATED CRITIQUE ---"
        "\nProvide a constructive critique (around 100-150 words) for the experts:"
    )

    critique_text = generate_content(
        SUPERVISOR_MODEL_ID,  # Use the same supervisor model
        "\n".join(prompt_parts),
        temperature=state["expert_temperature"],
        max_output_tokens=600,
    )

    # print(f"Supervisor critique generated: {critique_text.strip()}")
    return {"supervisor_critique": critique_text.strip() or "No specific critique generated."}


def revised_expert_consultation_node(state: AgenticWorkflowState) -> Dict[str, Any]:
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
                "explanation": response_dict.get("explanation", "N/A"),
                "answer": response_dict.get("answer", "N/A"),
                "original_model_name_for_revised_pass": expert_name,
            }
        )
    # print(f"Revised expert responses collected: {revised_outputs}")
    return {"revised_expert_outputs": revised_outputs}


def synthesis_supervisor_node(state: AgenticWorkflowState) -> Dict[str, Any]:  # Canvia el tipus de retorn
    expert_outputs_to_synthesize = state.get("revised_expert_outputs", [])
    if not expert_outputs_to_synthesize:
        expert_outputs_to_synthesize = state.get("initial_expert_responses", [])

    question_text = state["question_text"]

    if not expert_outputs_to_synthesize:
        print("No expert outputs available for final synthesis.")
        # Retorna un diccionari amb valors per defecte per a 'explanation' i 'answer'
        return {"final_synthesis": {"explanation": "No expert outputs available.", "answer": None}}

    prompt_parts = [
        "You are a Chief Medical Consultant AI. Your task is to synthesize a final, comprehensive report based on refined (or initial) analyses from several anonymized medical experts.",
        "Focus on creating a coherent narrative that integrates key insights, addresses the main question, and provides a clear conclusion/answer.",
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
            f"  Explanation: {resp.get('explanation', 'N/A').strip()}"
            f"  Answer: {resp.get('answer', 'N/A')}\n"
        )

    # Prompt explícit per al format de sortida del supervisor en benchmark
    if state.get("is_benchmark_mode"):
        prompt_parts.append(
            "\n\n--- YOUR FINAL SYNTHESIZED REPORT (Strict Benchmark Format) ---"
            "\nOutput EXACTLY in this format:\n"
            "Explanation: [Your synthesized reasoning for the chosen option]\n"
            "Answer: [A or B or C or D]"
        )
    else:  # Mode UI/Conversa
        prompt_parts.append(
            "\n\n--- YOUR FINAL SYNTHESIZED REPORT (Well-Formatted for Readability) ---"
            "\nDo not make any direct references to any expert, or how many experts were consulted."
            "\nFocus on providing a clear, comprehensive synthesis of the information provided by the experts."
            "\n\n**Important:**"
            "\n- Do not include any expert names or identifiers in your synthesis."
            "\n- Do not mention the number of experts consulted."
            "\n- Focus on the clinical case and the question at hand."
            "\n- Ensure your synthesis is coherent and addresses the question comprehensively."
            "\nProvide your synthesis as a single, well-structured text using Markdown for clear formatting."
            "Your response should be comprehensive and easy to read."
            "\n\n**Use the following Markdown elements as appropriate:**"
            "\n- Headings (e.g., `## Key Findings`, `### Differential Diagnosis`)"
            "\n- Bold text (`**important points**`)"
            "\n- Bullet points (`- Point 1`, `  - Sub-point 1.1`)"
            "\n- Numbered lists (`1. First step`, `2. Second step`)"
            "\n- Paragraphs for explanations."
            "\n\n**Structure your output STRICTLY as follows:**"
            "\nExplanation: [Your full synthesized report using Markdown. This should be a detailed analysis, addressing the question comprehensively. Aim for a logical flow, starting with a summary or overview if appropriate, then detailing findings, considerations, and rationale.]"
            "\nAnswer: [Your overall textual conclusion or primary answer, as a concise and clear statement. This should be the main takeaway.]"
        )

    max_tokens_synthesis = 1500 if state.get("is_benchmark_mode") else 2000

    raw_synthesized_text = generate_content(
        SUPERVISOR_MODEL_ID,
        "\n".join(prompt_parts),
        temperature=state["expert_temperature"],
        max_output_tokens=max_tokens_synthesis,
    )

    explanation, answer = extract_explanation_and_answer(raw_synthesized_text)

    if state.get("is_benchmark_mode"):
        if not answer or answer not in ["A", "B", "C", "D"]:
            print(
                f"Warning: Supervisor in benchmark mode did not produce a valid A,B,C,D answer. Raw: '{raw_synthesized_text}', Parsed: '{answer}'"
            )
            answer = None
        return {
            "final_synthesis": {"explanation": explanation or "Supervisor explanation not parsed.", "answer": answer}
        }
    else:
        # Per al mode no-benchmark, 'answer' contindrà la conclusió textual
        return {
            "final_synthesis": {
                "explanation": explanation or "Supervisor explanation not parsed.",
                "answer": answer or "Supervisor conclusion not parsed.",
            }
        }


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
