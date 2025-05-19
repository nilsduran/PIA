from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Annotated, Dict
import operator
import gradio as gr
import requests
import time


MODELS = {
    "General Medicine": "medicinageneralcsv-q4i0ydc9l1uvxbzsxmii8",
    "Basic Sciences": "cincies-bsiques-5x23mkxv2ftipprirc4i4714",
    "Pathology and Pharmacology": "patologia-i-farmacologia-3ipo0rdy5dkze8q",
    "Surgery": "cirurgia-6rm1gub7hny7bzm3hjgghwcf3tws7ar",
    "Pediatrics and Gynecology": "pediatria-i-ginecologia-q4n2dg2t5sweqdt9",
}

API_KEY = "AIzaSyDxk7cxcrDx3mcofYIosCggfkVbyHedO4w"


# Advanced generation configuration
def generate_content(api_key, tuned_model_id, prompt, temperature=0.7, max_output_tokens=2048):
    url = f"https://generativelanguage.googleapis.com/v1/tunedModels/{tuned_model_id}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
            "topP": 0.95,
            "topK": 40,
        },
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    return f"Error {response.status_code}: {response.text}"


class CollaborationState(TypedDict):
    case: str
    analyses: Annotated[List[Dict[str, str]], operator.add]
    debate: Annotated[List[str], operator.add]
    current_step: int
    full_history: str


def create_workflow(debate_rounds=3):
    workflow = StateGraph(CollaborationState)

    # Main analysis nodes
    def expert_analysis_node(model_name):
        def node_action(state: CollaborationState):
            prompt = f"""As an expert in {model_name}, analyze this case:
            {state['case']}

            Complete history:
            {state['full_history'] if state['full_history'] else "None"}

            Focus on:
            1. Key aspects related to your specialty
            2. Relevant differential diagnoses
            3. Specific recommendations
            4. Points of controversy

            Limit your response to 80 words and use precise technical terms."""

            response = generate_content(API_KEY, MODELS[model_name], prompt, temperature=0.6)

            return {
                "analyses": [{"specialty": model_name, "content": response}],
                "full_history": f"{state['full_history']}\n\n{model_name}:\n{response}",
                "current_step": state["current_step"] + 1,
            }

        return node_action

    # Critical debate node
    def debate_node(model_name):
        def node_action(state: CollaborationState):
            last_analysis = state["analyses"][-1]
            prompt = f"""As {model_name}, critique this previous analysis:
            {last_analysis['content']}

            Original case: {state['case']}

            Point out:
            1. Strengths of the analysis
            2. Weaknesses or omissions
            3. Possible alternatives
            4. Contradictory evidence

            Respond in maximum 60 words with a constructive critical approach."""

            response = generate_content(API_KEY, MODELS[model_name], prompt, temperature=0.7)

            return {
                "debate": [f"{model_name} critique:\n{response}"],
                "full_history": f"{state['full_history']}\n\n{model_name} Debate:\n{response}",
                "current_step": state["current_step"] + 1,
            }

        return node_action

    # Add main nodes
    for specialty in MODELS:
        workflow.add_node(specialty, expert_analysis_node(specialty))
        workflow.add_node(f"Debate_{specialty}", debate_node(specialty))

    # Configure flow with debates
    workflow.add_edge("General Medicine", "Debate_General Medicine")
    workflow.add_edge("Debate_General Medicine", "Basic Sciences")
    workflow.add_edge("Basic Sciences", "Debate_Basic Sciences")
    workflow.add_edge("Debate_Basic Sciences", "Pathology and Pharmacology")
    workflow.add_edge("Pathology and Pharmacology", "Debate_Pathology and Pharmacology")
    workflow.add_edge("Debate_Pathology and Pharmacology", "Surgery")
    workflow.add_edge("Surgery", "Debate_Surgery")
    workflow.add_edge("Debate_Surgery", "Pediatrics and Gynecology")
    workflow.add_edge("Pediatrics and Gynecology", "Debate_Pediatrics and Gynecology")
    workflow.add_edge("Debate_Pediatrics and Gynecology", END)

    workflow.set_entry_point("General Medicine")
    return workflow.compile()


def generate_final_diagnosis(analyses: List[Dict[str, str]], case: str) -> str:
    formatted_analyses = "\n".join(f"*{a['specialty']}*:\n{a['content']}\n" for a in analyses)

    consensus_prompt = f"""Generate a comprehensive medical report based on:
    *Case*: {case}

    *Expert Analyses*:
    {formatted_analyses}

    *Required Structure*:
    1. Case summary (50 words)
    2. Differential diagnoses with probabilities (%)
    3. Recommended diagnostic tests (ordered by priority)
    4. Treatment options (risk/benefit table)
    5. Follow-up plan
    6. Final conclusions and main recommendation

    Use markdown format and precise medical terminology."""

    return generate_content(API_KEY, MODELS["General Medicine"], consensus_prompt, temperature=0.3)


def run_collaboration(case):
    workflow = create_workflow()

    initial_state = {"case": case, "analyses": [], "debate": [], "current_step": 0, "full_history": ""}

    final_state = workflow.invoke(initial_state)

    output = []
    for analysis in final_state["analyses"]:
        output.append((f"Analysis {analysis['specialty']}", analysis["content"]))
    for debate in final_state["debate"]:
        output.append(("Debate", debate))

    output.append(("FINAL DIAGNOSIS", generate_final_diagnosis(final_state["analyses"], case)))

    return output


# Improved interface with metrics
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🩺 Interdisciplinary Medical Consensus Simulator")

    with gr.Row():
        case_input = gr.Textbox(label="Clinical Case", lines=4, placeholder="Enter the complete clinical case here...")
        with gr.Column():
            submit_btn = gr.Button("Start Simulation", variant="primary")
            gr.Markdown("### Evaluation Parameters")
            complexity = gr.Slider(1, 5, value=3, label="Complexity Level")

    with gr.Tabs():
        with gr.TabItem("Complete Process"):
            chatbot = gr.Chatbot(label="Interactions", height=500)

        with gr.TabItem("Final Metrics"):
            gr.Markdown("### Quality Indicators")
            accuracy = gr.Textbox(label="Estimated Diagnostic Accuracy")
            consistency = gr.Textbox(label="Inter-Judge Consistency")
            efficiency = gr.Textbox(label="Decision-Making Efficiency")

    submit_btn.click(fn=run_collaboration, inputs=case_input, outputs=chatbot)

demo.launch()
