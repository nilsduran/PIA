from typing import Dict, Any, Union, Optional
from main.langgraph_workflow import AgenticWorkflowState, EXPERT_DEFINITIONS, create_compiled_agent


def conversation_agentic_workflow(
    question_text: str,
    num_experts_to_select: int,
    diversity_option: str = "Mitjana",
    # These will be set in the initial state
    expert_temperature: float = 0.7,
    system_prompt: str = "You are an AI medical expert. Your role is to provide clear and concise information based on your specialty.",
    benchmark_shot_prompt: Optional[str] = None,
    is_benchmark_mode: bool = False,  # Add this parameter
    options: Optional[Dict[str, str]] = None,  # For benchmark
    correct_answer_idx: Optional[str] = None,  # For benchmark
) -> Union[Dict[str, Any], Optional[str]]:

    initial_state_for_run: AgenticWorkflowState = {
        "question_text": question_text,
        "options": options,
        "correct_answer_idx": correct_answer_idx,
        "available_experts": EXPERT_DEFINITIONS,
        "num_experts_to_select": num_experts_to_select,
        "diversity_option": diversity_option,
        "is_benchmark_mode": is_benchmark_mode,
        "selected_expert_names": [],
        "initial_expert_responses": [],
        "supervisor_critique": None,
        "revised_expert_outputs": [],
        "final_synthesis": None,
        # Pass configurations for expert calls
        "expert_temperature": expert_temperature,
        "system_prompt": system_prompt,
        "benchmark_shot_prompt": benchmark_shot_prompt,
    }

    compiled_agent = create_compiled_agent()
    final_run_state_dict = compiled_agent.invoke(initial_state_for_run)  # invoke returns a dict

    return {
        "selected_expert_names": final_run_state_dict.get("selected_expert_names"),
        "initial_expert_responses": final_run_state_dict.get("initial_expert_responses"),
        "supervisor_critique": final_run_state_dict.get("supervisor_critique"),
        "revised_expert_outputs": final_run_state_dict.get("revised_expert_outputs"),
        "final_synthesis": final_run_state_dict.get("final_synthesis"),
    }
