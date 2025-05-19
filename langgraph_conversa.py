import gradio as gr
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Optional, Any
from functools import partial
import time
import os
import re
import random  # For fallback in router
import numpy as np
from sentence_transformers import SentenceTransformer, util
from benchmarking import generate_content

# --- CONFIGURATION ---
# Path to the directory containing individual .npy embedding files
AGENTS_EMBEDDINGS_DIR_PATH = "agents_embeddings"


def extract_explanation_and_answer_from_text(text: str) -> tuple[Optional[str], Optional[str]]:
    if not isinstance(text, str):
        return None, None

    explanation_content = None
    answer_content = None

    # Try to find "Explanation:"
    # Capture everything after "Explanation:" up to either "Answer:", "Conclusion:", or end of string.
    exp_match = re.search(
        r"Explanation:(.*?)(?=\n(?:Answer|Conclusion):|$)", text, re.DOTALL | re.IGNORECASE | re.UNICODE
    )
    if exp_match:
        explanation_content = exp_match.group(1).strip()

    # Try to find "Answer:"
    ans_match = re.search(r"\nAnswer:(.*)", text, re.DOTALL | re.IGNORECASE | re.UNICODE)
    if ans_match:
        answer_content = ans_match.group(1).strip()
    else:
        # If "Answer:" not found, try to find "Conclusion:"
        con_match = re.search(r"\nConclusion:(.*)", text, re.DOTALL | re.IGNORECASE | re.UNICODE)
        if con_match:
            answer_content = con_match.group(1).strip()

    # If explanation_content is still None but we found an answer/conclusion,
    # assume everything before that marker was the explanation.
    if explanation_content is None:
        if ans_match:  # if "Answer:" was found
            explanation_content = text.split("\nAnswer:")[0].strip()
            # If "Explanation:" was at the start, remove it
            if explanation_content.lower().startswith("explanation:"):
                explanation_content = explanation_content[len("explanation:") :].strip()
        elif con_match:  # if "Conclusion:" was found
            explanation_content = text.split("\nConclusion:")[0].strip()
            if explanation_content.lower().startswith("explanation:"):
                explanation_content = explanation_content[len("explanation:") :].strip()
        elif answer_content is None and text.strip():  # No markers, take all as explanation
            explanation_content = text.strip()

    # If no specific answer/conclusion was extracted, but we have an explanation,
    # provide a generic answer. (This part of your original logic is fine)
    if answer_content is None and explanation_content and not (ans_match or con_match):
        answer_content = "See explanation for details."

    return explanation_content, answer_content


# --- LANGGRAPH STATE DEFINITION ---
class AgenticWorkflowState(TypedDict):
    question_text: str
    available_experts: Dict[str, Dict[str, Any]]
    num_experts_to_select: int
    supervisor_model_id_for_run: str
    selected_expert_names: List[str]
    routing_rationale: Optional[str]
    expert_responses: List[Dict[str, Any]]
    provisional_supervisor_explanation: Optional[str]
    provisional_supervisor_conclusion: Optional[str]
    final_supervisor_explanation: Optional[str]
    final_supervisor_conclusion: Optional[str]
    error_messages: List[str]
    node_trace: List[str]


# --- EXPERT DEFINITIONS & EMBEDDINGS ---
ORIGINAL_EXPERT_MODEL_DATA = {
    "General Medicine": {"id": "tunedModels/medicinageneralcsv-q4i0ydc9l1uvxbzsxmii8"},
    "Basic Sciences": {"id": "tunedModels/ciencies-basiques-2-pfg4bpafqcay88df2kr8"},
    "Pathology & Pharmacology": {"id": "tunedModels/patologia-farmacologia-2-8iy2ixmy5bluqzw"},
    "Surgery": {"id": "tunedModels/cirurgia-2-2c1cy8nkr5ca5mui15tu4wtlpapp8"},
    "Pediatrics & Gynecology": {"id": "tunedModels/pediatria-ginecologia-2-ss7f3iy509x7x43h"},
}
EXPERT_DISPLAY_NAMES = list(ORIGINAL_EXPERT_MODEL_DATA.keys())

# Mapping from display names to the filenames (without .npy) used for embeddings
AGENT_EMBEDDING_FILENAME_MAP = {
    "Basic Sciences": "Ciències_Bàsiques",
    "General Medicine": "Medicina_General",
    "Pathology & Pharmacology": "Patologia_Farmacologia",
    "Surgery": "Cirurgia",
    "Pediatrics & Gynecology": "Pediatria_Ginecologia",
}


def load_expert_embeddings_from_npy_files(
    embeddings_dir_path: str, expert_data: Dict[str, dict], filename_map: Dict[str, str]
) -> Dict[str, Dict[str, Any]]:
    """
    Loads embeddings from individual .npy files for each expert and merges
    them with other expert data (like model IDs).
    """
    expert_definitions_with_embeddings = {}

    if not os.path.isdir(embeddings_dir_path):
        print(f"Warning UI: Embeddings directory '{embeddings_dir_path}' not found. Routing will rely on fallback.")
        # Populate with basic info if embeddings dir is missing
        for expert_name, data in expert_data.items():
            expert_definitions_with_embeddings[expert_name] = {
                "id": data["id"],
                "embedding": None,
                "description_source": "No embedding file found.",
                "short_description": f"AI expert specializing in {expert_name}.",
            }
        return expert_definitions_with_embeddings

    print(f"Loading expert embeddings from directory: {embeddings_dir_path}")
    for display_name, data in expert_data.items():
        model_id = data["id"]
        embedding_filename_key = filename_map.get(display_name)  # Get the filename stem from the map
        expert_embedding = None
        description_source = f"Default description for {display_name}"

        if embedding_filename_key:
            embedding_file_path = os.path.join(embeddings_dir_path, f"{embedding_filename_key}.npy")
            if os.path.exists(embedding_file_path):
                try:
                    emb = np.load(embedding_file_path)
                    # Ensure embedding is 2D [1, dim] if it's loaded as 1D
                    expert_embedding = emb.reshape(1, -1) if emb.ndim == 1 else emb
                    description_source = f"Textbook embedding loaded from {embedding_filename_key}.npy"
                    print(f"  Successfully loaded embedding for '{display_name}' from '{embedding_filename_key}.npy'")
                except Exception as e:
                    print(
                        f"  Warning UI: Error loading embedding for '{display_name}' from '{embedding_file_path}': {e}"
                    )
            else:
                print(
                    f"  Warning UI: Embedding file '{embedding_filename_key}.npy' not found for '{display_name}' in '{embeddings_dir_path}'."
                )
        else:
            print(f"  Warning UI: No embedding filename mapping found for expert '{display_name}'.")

        expert_definitions_with_embeddings[display_name] = {
            "id": model_id,
            "embedding": expert_embedding,  # This is the textbook embedding (or None)
            "description_source": description_source,
            "short_description": f"AI expert specializing in {display_name}.",
        }
        if expert_embedding is None:
            print(
                f"    -> Note: Routing for '{display_name}' might be less accurate due to missing textbook embedding."
            )

    return expert_definitions_with_embeddings


try:
    query_embedding_model_ui = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    print(f"Error UI: Loading SentenceTransformer for queries: {e}. Semantic routing may fail.")
    query_embedding_model_ui = None

# Load master definitions using the new function
MASTER_EXPERT_DEFINITIONS_UI = load_expert_embeddings_from_npy_files(
    AGENTS_EMBEDDINGS_DIR_PATH, ORIGINAL_EXPERT_MODEL_DATA, AGENT_EMBEDDING_FILENAME_MAP
)

# --- LANGGRAPH NODES (dynamic_expert_router_node, expert_consultation_node, synthesizing_supervisor_node) ---
# The nodes themselves don't need to change if MASTER_EXPERT_DEFINITIONS_UI is prepared correctly.
# dynamic_expert_router_node will receive `state["available_experts"]` which is MASTER_EXPERT_DEFINITIONS_UI.
# It will look for the "embedding" key there.


def dynamic_expert_router_node(state: AgenticWorkflowState):
    state["node_trace"].append("dynamic_expert_router_node")
    if not query_embedding_model_ui:
        state["error_messages"].append("RouterNode: Query embedding model not available.")
        available_names = list(state["available_experts"].keys())
        sel_names = random.sample(available_names, min(len(available_names), state["num_experts_to_select"]))
        return {
            "selected_expert_names": sel_names,
            "routing_rationale": "Fallback: Query embedding model failed, selected experts randomly.",
        }

    question_text = state["question_text"]
    # available_experts is MASTER_EXPERT_DEFINITIONS_UI passed through state
    available_experts = state["available_experts"]
    num_to_select = state["num_experts_to_select"]
    try:
        question_embedding = query_embedding_model_ui.encode(question_text, convert_to_tensor=False)
    except Exception as e:
        state["error_messages"].append(f"RouterNode: Error encoding question: {e}")
        available_names = list(state["available_experts"].keys())
        sel_names = random.sample(available_names, min(len(available_names), state["num_experts_to_select"]))
        return {
            "selected_expert_names": sel_names,
            "routing_rationale": "Fallback: Question encoding failed, selected experts randomly.",
        }

    scores = []
    for name, info in available_experts.items():
        textbook_emb = info.get("embedding")  # This should be the pre-loaded textbook embedding
        similarity = -2.0  # Default to very low similarity

        if textbook_emb is not None and isinstance(textbook_emb, np.ndarray):
            try:
                # Ensure question_embedding is also 2D for cos_sim if textbook_emb is [1, dim]
                q_emb_2d = question_embedding.reshape(1, -1) if question_embedding.ndim == 1 else question_embedding
                similarity = util.cos_sim(q_emb_2d, textbook_emb).item()
            except Exception as e_sim:
                print(f"  Router: Error calculating similarity for {name} with textbook_emb: {e_sim}")
                similarity = -1.0
        else:
            # Fallback to short_description if no textbook embedding
            if query_embedding_model_ui and info.get("short_description"):
                try:
                    desc_emb = query_embedding_model_ui.encode(info["short_description"], convert_to_tensor=False)
                    desc_emb_2d = desc_emb.reshape(1, -1) if desc_emb.ndim == 1 else desc_emb
                    q_emb_2d = question_embedding.reshape(1, -1) if question_embedding.ndim == 1 else question_embedding
                    similarity = util.cos_sim(q_emb_2d, desc_emb_2d).item() * 0.6  # Penalize description match
                    # print(f"  Router: Used short_description for {name}, sim: {similarity:.3f} (after penalty)")
                except Exception as e_desc_sim:
                    print(f"  Router: Error with short_description embedding for {name}: {e_desc_sim}")
                    similarity = -1.5
        scores.append({"name": name, "similarity": similarity})

    scores.sort(key=lambda x: x["similarity"], reverse=True)
    selected_names = [expert["name"] for expert in scores[:num_to_select]]

    # Fallback logic (same as before)
    if not selected_names and available_experts:
        selected_names = random.sample(list(available_experts.keys()), min(len(available_experts), num_to_select))
        rationale = (
            f"Selected {len(selected_names)} experts by fallback (random sampling - no strong similarities found)."
        )
    elif len(selected_names) < num_to_select and available_experts:
        additional_needed = num_to_select - len(selected_names)
        remaining_experts_sorted = [
            ex["name"] for ex in scores[len(selected_names) :] if ex["name"] not in selected_names
        ]
        selected_names.extend(remaining_experts_sorted[:additional_needed])
        rationale = (
            f"Selected {len(selected_names)} experts. Top by semantic similarity, filled with next available if needed."
        )
    else:
        top_scores_str = ", ".join(
            [f"{s['name']}: {s['similarity']:.3f}" for s in scores[:num_to_select] if s["similarity"] > -2.0]
        )
        rationale = f"Selected {len(selected_names)} experts based on semantic similarity. Top scores: [{top_scores_str if top_scores_str else 'None above threshold'}]"

    return {"selected_expert_names": selected_names, "routing_rationale": rationale}


def expert_consultation_node(state: AgenticWorkflowState, expert_temperature: float):
    state["node_trace"].append("expert_consultation_node")
    expert_responses = []
    expert_system_prompt = "You are a specialized medical AI assistant. Your role is to provide expert insights."

    for expert_name in state.get("selected_expert_names", []):
        expert_info = state["available_experts"].get(expert_name)
        if not expert_info or not expert_info.get("id"):
            state["error_messages"].append(f"ExpertConsultation: Info/ID for '{expert_name}' not found.")
            continue

        prompt = (
            f"{expert_system_prompt}\n\n"
            f"You are an AI expert in {expert_name}.\n"
            f"Analyze the following clinical case/question from your specialty perspective:\n\n"
            f"Case/Question: {state['question_text']}\n\n"
            f"Provide a concise analysis (around 80-120 words), including key insights, potential differential diagnoses (if applicable), and specific recommendations or considerations related to your field. "
            f"Structure your response clearly. Use the following format:\n"
            f"Explanation: [Your detailed analysis and reasoning]\n"
            f"Conclusion: [Your main conclusion or summary statement, can be free text]"
        )

        raw_response = generate_content(
            expert_info["id"], prompt, temperature=expert_temperature, max_output_tokens=300
        )
        explanation, conclusion_text = extract_explanation_and_answer_from_text(raw_response)

        expert_responses.append(
            {
                "model_name": expert_name,
                "conclusion": conclusion_text or "No specific conclusion provided.",
                "explanation": explanation or "No detailed explanation provided.",
            }
        )
        time.sleep(0.6)
    return {"expert_responses": expert_responses}


def synthesizing_supervisor_node(state: AgenticWorkflowState):
    state["node_trace"].append("synthesizing_supervisor_node")
    expert_responses = state.get("expert_responses", [])
    question_text = state["question_text"]
    supervisor_model_id_for_run = state["supervisor_model_id_for_run"]

    if not supervisor_model_id_for_run:
        state["error_messages"].append("SupervisorNode: Supervisor Model ID not set in state.")
        err_text = "Error: Supervisor model not specified for this run."
        return {
            "provisional_supervisor_explanation": err_text,
            "provisional_supervisor_conclusion": "N/A",
            "final_supervisor_explanation": err_text,
            "final_supervisor_conclusion": "N/A",
        }

    if not expert_responses:
        state["error_messages"].append("SupervisorNode: No expert responses to synthesize.")
        no_resp_text = "No specialist consultations were available or they failed to provide input for supervision."
        return {
            "provisional_supervisor_explanation": no_resp_text,
            "provisional_supervisor_conclusion": "N/A",
            "final_supervisor_explanation": no_resp_text,
            "final_supervisor_conclusion": "N/A",
        }

    initial_prompt_parts = [
        "You are a Chief Medical Consultant AI. Your task is to synthesize insights from several specialist AI consultations for the given clinical case/question.",
        "Review all specialist inputs carefully. Identify common themes, agreements, discrepancies, and unique perspectives.",
        "Then, provide a comprehensive, consolidated analysis and a primary conclusion.",
        f"\nClinical Case/Question:\n{question_text}\n\n--- Specialist AI Consultations ---",
    ]
    for resp in expert_responses:
        initial_prompt_parts.append(f"\n**Consultation from {resp['model_name']}**:")
        initial_prompt_parts.append(f"  Specialist's Conclusion: {resp.get('conclusion', 'Not specified')}")
        initial_prompt_parts.append(
            f"  Specialist's Explanation/Analysis:\n  {resp.get('explanation', 'Not provided')}"
        )
    initial_prompt_parts.append("\n--- Your Initial Synthesis Task ---")
    initial_prompt_parts.append(
        "Based on the above, provide your initial consolidated report. Use the following format STRICTLY:"
    )
    initial_prompt_parts.append(
        "Explanation: [Your synthesized analysis, integrating key points from specialists. Be thorough and clear, around 150-250 words.]"
    )
    initial_prompt_parts.append(
        "Conclusion: [Your primary overall conclusion or synthesized statement based on the analyses.]"
    )

    raw_initial_response = generate_content(
        supervisor_model_id_for_run, "\n".join(initial_prompt_parts), temperature=0.7, max_output_tokens=500
    )
    prov_expl, prov_conc = extract_explanation_and_answer_from_text(raw_initial_response)

    reflection_prompt_parts = [
        "You are a Chief Medical Consultant AI performing a self-correction and refinement step.",
        "You have already provided an initial synthesis based on specialist AI inputs. Now, critically review your own work.",
        f"\nClinical Case/Question:\n{question_text}\n\n--- Original Specialist AI Consultations (for reference) ---",
    ]
    for resp in expert_responses:
        reflection_prompt_parts.append(f"\n**Consultation from {resp['model_name']}**:")
        reflection_prompt_parts.append(f"  Specialist's Conclusion: {resp.get('conclusion', 'Not specified')}")
        reflection_prompt_parts.append(
            f"  Specialist's Explanation/Analysis:\n  {resp.get('explanation', 'Not provided')}"
        )
    reflection_prompt_parts.append("\n--- Your Previous Initial Synthesis ---")
    reflection_prompt_parts.append(f"Your Initial Explanation:\n{prov_expl or 'Not generated or format error.'}")
    reflection_prompt_parts.append(f"Your Initial Conclusion:\n{prov_conc or 'Not generated or format error.'}")
    reflection_prompt_parts.append("\n--- Self-Correction & Refinement Task ---")
    reflection_prompt_parts.append("Critically review your initial synthesis. Consider the following:")
    reflection_prompt_parts.append("- Have all specialist inputs been adequately considered and integrated?")
    reflection_prompt_parts.append(
        "- Are there any remaining ambiguities, contradictions, or overlooked crucial points?"
    )
    reflection_prompt_parts.append("- Is the explanation sufficiently clear, comprehensive, and medically sound?")
    reflection_prompt_parts.append("- Is the conclusion well-supported by the synthesized evidence and analysis?")
    reflection_prompt_parts.append(
        "Provide your FINAL, refined, and comprehensive report. Use the following format STRICTLY:"
    )
    reflection_prompt_parts.append(
        "Explanation: [Your FINAL refined and comprehensive analysis, integrating all relevant points and addressing any reflections. Aim for 200-300 words.]"
    )
    reflection_prompt_parts.append(
        "Conclusion: [Your FINAL overall conclusion or synthesized statement, ensuring it is robust and well-justified.]"
    )

    raw_final_response = generate_content(
        supervisor_model_id_for_run, "\n".join(reflection_prompt_parts), temperature=0.4, max_output_tokens=600
    )
    final_expl, final_conc = extract_explanation_and_answer_from_text(raw_final_response)

    return {
        "provisional_supervisor_explanation": prov_expl or "Error in initial synthesis.",
        "provisional_supervisor_conclusion": prov_conc or "Error in initial conclusion.",
        "final_supervisor_explanation": final_expl or (prov_expl or "Error in final synthesis."),
        "final_supervisor_conclusion": final_conc or (prov_conc or "Error in final conclusion."),
    }


# --- LANGGRAPH WORKFLOW SETUP ---
EXPERT_CONSULT_TEMP_UI = 0.7

workflow_builder_ui = StateGraph(AgenticWorkflowState)
workflow_builder_ui.add_node("router", dynamic_expert_router_node)
consult_node_ui_with_args = partial(expert_consultation_node, expert_temperature=EXPERT_CONSULT_TEMP_UI)
workflow_builder_ui.add_node("expert_consultation", consult_node_ui_with_args)
workflow_builder_ui.add_node("supervisor_synthesis", synthesizing_supervisor_node)

workflow_builder_ui.set_entry_point("router")
workflow_builder_ui.add_edge("router", "expert_consultation")
workflow_builder_ui.add_edge("expert_consultation", "supervisor_synthesis")
workflow_builder_ui.add_edge("supervisor_synthesis", END)

compiled_agent_ui = workflow_builder_ui.compile()


# --- GRADIO UI FUNCTION ---
def format_report_for_display(final_state: AgenticWorkflowState, case_text: str) -> str:
    report_parts = [f"# Multi-Agent Medical Analysis Report\n\n## Case Presentation:\n{case_text}\n"]
    report_parts.append("## 1. Expert Routing Phase:")
    report_parts.append(f"- **Routing Rationale:** {final_state.get('routing_rationale', 'N/A')}")
    selected_experts = final_state.get("selected_expert_names", [])
    report_parts.append(
        f"- **Selected Experts ({len(selected_experts)}):** {', '.join(selected_experts) if selected_experts else 'None'}\n"
    )
    report_parts.append("## 2. Specialist Consultations:")
    if final_state.get("expert_responses"):
        for i, resp in enumerate(final_state.get("expert_responses", [])):
            report_parts.append(f"\n### 2.{i+1} Consultation: {resp['model_name']}")
            report_parts.append(
                f"- **Specialist's Conclusion:** {resp.get('conclusion', 'Not provided.')}"
            )  # Changed 'answer' to 'conclusion'
            report_parts.append(f"- **Specialist's Analysis:**\n{resp.get('explanation', 'Not provided.')}")
    else:
        report_parts.append("*No specialist consultations were performed or they yielded no response.*\n")
    report_parts.append("\n## 3. Supervising Consultant's Synthesis (Initial):")
    report_parts.append(
        f"- **Initial Synthesized Analysis:**\n{final_state.get('provisional_supervisor_explanation', 'N/A')}"
    )
    report_parts.append(
        f"- **Initial Overall Conclusion:** {final_state.get('provisional_supervisor_conclusion', 'N/A')}\n"
    )
    report_parts.append("\n## 4. Supervising Consultant's Report (Final Refined):")
    report_parts.append(
        f"- **Final Comprehensive Analysis:**\n{final_state.get('final_supervisor_explanation', 'N/A')}"
    )
    report_parts.append(f"- **Final Overall Conclusion:** {final_state.get('final_supervisor_conclusion', 'N/A')}\n")
    if final_state.get("error_messages"):
        report_parts.append("\n## System Notes & Errors:")
        for err in final_state.get("error_messages", []):
            report_parts.append(f"- {err}")
    return "\n".join(report_parts)


def run_agentic_simulation_for_gradio(case_text_input: str, diversity_level: int, supervisor_choice_display_name: str):
    if not case_text_input.strip():
        return "Please enter a clinical case or question to analyze."

    supervisor_model_id_to_use = None
    if supervisor_choice_display_name in MASTER_EXPERT_DEFINITIONS_UI:  # Check against loaded definitions
        supervisor_model_id_to_use = MASTER_EXPERT_DEFINITIONS_UI[supervisor_choice_display_name]["id"]

    if not supervisor_model_id_to_use:
        return (
            f"Error: Could not find a valid model ID for the selected supervisor '{supervisor_choice_display_name}'. "
            f"Ensure it's defined in ORIGINAL_EXPERT_MODEL_DATA and embeddings are processed if needed. "
            f"Available in UI: {list(MASTER_EXPERT_DEFINITIONS_UI.keys())}"
        )

    if not query_embedding_model_ui:
        return "System Error: The query embedding model (SentenceTransformer) is not loaded. Semantic routing cannot proceed."

    yield "Processing your request... Routing to specialists..."
    initial_state_ui: AgenticWorkflowState = {
        "question_text": case_text_input,
        "available_experts": MASTER_EXPERT_DEFINITIONS_UI,  # Pass the definitions with loaded embeddings
        "num_experts_to_select": int(diversity_level),
        "supervisor_model_id_for_run": supervisor_model_id_to_use,
        "selected_expert_names": [],
        "routing_rationale": None,
        "expert_responses": [],
        "provisional_supervisor_explanation": None,
        "provisional_supervisor_conclusion": None,
        "final_supervisor_explanation": None,
        "final_supervisor_conclusion": None,
        "error_messages": [],
        "node_trace": [],
    }

    final_report_md = "### Interim Update: Consulting Experts...\nThis may take a moment."
    yield final_report_md
    time.sleep(0.1)

    try:
        final_state_ui = compiled_agent_ui.invoke(initial_state_ui)

        interim_update_parts = [
            format_report_for_display(final_state_ui, case_text_input).split("## 3.")[0]
        ]  # Show up to expert consultations
        interim_update_parts.append("## 3. Supervising Consultant's Synthesis (Initial):")
        interim_update_parts.append("*Processing initial synthesis...*")
        yield "\n".join(interim_update_parts)
        time.sleep(0.1)

        final_report_md = format_report_for_display(final_state_ui, case_text_input)

    except Exception as e:
        import traceback

        tb_str = traceback.format_exc()
        final_report_md = (
            f"# Error During Analysis\n\n"
            f"An unexpected error occurred while processing the case:\n**{str(e)}**\n\n"
            f"**Traceback:**\n```\n{tb_str}\n```"
        )
    yield final_report_md


# --- GRADIO INTERFACE ---
with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="sky"), title="Multi-Agent Medical AI Simulator"
) as demo_ui_med:
    gr.Markdown("# 🩺 Multi-Agent Medical AI Simulator")
    gr.Markdown(
        "Enter a clinical case or medical question. The system will use semantic routing to select relevant AI specialists, "
        "gather their analyses, and a supervising AI will synthesize a final report, including a self-correction step."
    )
    with gr.Row():
        with gr.Column(scale=3):
            case_text_area_ui = gr.Textbox(
                label="Clinical Case / Medical Question",
                lines=8,
                placeholder="Enter the detailed case or your question here...",
            )
        with gr.Column(scale=1):
            diversity_slider_ui = gr.Slider(
                minimum=1,
                maximum=len(MASTER_EXPERT_DEFINITIONS_UI) if MASTER_EXPERT_DEFINITIONS_UI else 1,
                value=min(3, len(MASTER_EXPERT_DEFINITIONS_UI) if MASTER_EXPERT_DEFINITIONS_UI else 1),
                step=1,
                label="Expert Diversity (Number to Select)",
            )
            supervisor_dropdown_ui = gr.Dropdown(
                label="Select Supervising Consultant Model",
                choices=EXPERT_DISPLAY_NAMES,
                value=(
                    "General Medicine"
                    if "General Medicine" in EXPERT_DISPLAY_NAMES
                    else (EXPERT_DISPLAY_NAMES[0] if EXPERT_DISPLAY_NAMES else None)
                ),
                interactive=True,
            )
            submit_button_gradio = gr.Button(
                "🔬 Initiate Multi-Agent Analysis", variant="primary", elem_id="submit_button_main"
            )
            clear_button_gradio = gr.Button("🗑️ Clear All")

    gr.Markdown("---")
    output_report_md = gr.Markdown(label="Generated Medical Analysis Report")

    submit_button_gradio.click(
        fn=run_agentic_simulation_for_gradio,
        inputs=[case_text_area_ui, diversity_slider_ui, supervisor_dropdown_ui],
        outputs=output_report_md,
    )
    # Ensure lambda for clear button correctly targets the output
    clear_button_gradio.click(
        lambda: (
            "",
            min(3, len(MASTER_EXPERT_DEFINITIONS_UI) if MASTER_EXPERT_DEFINITIONS_UI else 1),  # Reset slider
            (
                "General Medicine"
                if "General Medicine" in EXPERT_DISPLAY_NAMES
                else (EXPERT_DISPLAY_NAMES[0] if EXPERT_DISPLAY_NAMES else None)
            ),  # Reset dropdown
            "",
        ),
        outputs=[case_text_area_ui, diversity_slider_ui, supervisor_dropdown_ui, output_report_md],
    )


if __name__ == "__main__":
    demo_ui_med.launch(share=False)
