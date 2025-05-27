from datasets import load_dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
from langgraph_workflow import AgenticWorkflowState, EXPERT_DEFINITIONS, create_compiled_agent


def benchmark_agentic_workflow(
    strategy_name: str,
    num_experts_to_select: int,
    num_questions_to_run: int = 10,
    diversity_option: str = "Mitjana",  # Default diversity option for benchmark
    k_shot: int = 5,
    expert_temperature: float = 0.4,  # Default expert temp for benchmark
):
    print(f"\n--- Benchmarking: {strategy_name} (Selecting {num_experts_to_select} experts) ---")

    medqa_test = (
        load_dataset("GBaker/MedQA-USMLE-4-options", split="test").shuffle().select(range(num_questions_to_run))
    )
    medqa_train_data = (
        load_dataset("GBaker/MedQA-USMLE-4-options", split="train").shuffle(seed=42).select(range(k_shot))
    )
    system_prompt_benchmark = (
        "You are a medical expert answering multiple-choice questions. "
        "Always output EXACTLY in this format, nothing more:\n"
        "Explanation: [Your brief but precise medical reasoning, no code or additional text]\n"
        "Answer: [A or B or C or D]\n"
    )
    explanations = [
        "A 35-year-old woman presents with various complaints and laboratory testing reveals the presence of anti-centromere antibodies. These antibodies are strongly associated with limited systemic sclerosis (also known as CREST syndrome), which includes Calcinosis, Raynaud's phenomenon, Esophageal dysmotility, Sclerodactyly, and Telangiectasia. Given this context, let's evaluate the given symptoms and signs:\nA: Pallor, cyanosis, and erythema of the hands - These are classic signs of Raynaud's phenomenon, commonly seen in limited systemic sclerosis.\nB: Blanching vascular abnormalities - These are also indicative of Raynaud's phenomenon, expected in this condition.\nC: Hypercoagulable state - Not typically a feature of limited systemic sclerosis; it is more associated with other connective tissue diseases or specific genetic disorders.\nD: Heartburn and regurgitation - These symptoms are consistent with esophageal dysmotility, a common feature of limited systemic sclerosis.",
        "The child's symptoms and physical examination findings—including slurred speech, frequent falls, pes cavus (high-arched feet), hammer toes, and kyphoscoliosis (curvature of the spine)—suggest a progressive neurological disorder. These signs are indicative of Friedreich's ataxia, a trinucleotide repeat disease.\nFriedreich's ataxia is caused by an expansion of the GAA trinucleotide repeat in the frataxin (FXN) gene. This condition typically presents with progressive ataxia, leading to difficulties with speech and coordination, as well as the skeletal abnormalities mentioned.\n",
        "The patient's presentation, including fever, headache, seizures, and altered behavior, along with the MRI findings of edema and hemorrhage in the left temporal lobe, strongly suggests herpes simplex encephalitis (HSE). HSE is a viral infection that primarily affects the temporal lobes and can cause significant brain edema.\nThe primary mechanism of edema in herpes simplex encephalitis is the breakdown of endothelial tight junctions. This disruption of the blood-brain barrier allows fluid to leak into the brain parenchyma, leading to vasogenic edema. This process is driven by the inflammatory response to the viral infection, which damages the endothelial cells and compromises the integrity of the blood-brain barrier.",
        "The patient's presentation includes shortness of breath, cough, severe lower limb edema, signs of right heart failure (jugular engorgement, hepatomegaly, hepatojugular reflux), and findings suggestive of pulmonary fibrosis. The physical examination and diagnostic tests (CT and echocardiogram) reveal right heart failure and severe pulmonary fibrosis. Cor pulmonale is a condition where right heart failure is caused by a primary lung disorder, typically chronic pulmonary hypertension. In this case, the severe pulmonary fibrosis is likely the underlying cause of the pulmonary hypertension, leading to cor pulmonale.",
        "Based on the information provided, the child's recurrent abdominal pain that occurs only at school, with no symptoms at home, and no abnormalities on physical examination or laboratory tests, suggests a functional cause. The child's symptoms are consistent with a functional abdominal pain disorder, possibly related to school avoidance or a behavioral component. Given that the child denies functional pain and there are no alarm symptoms (such as blood in the stool, weight loss, or nighttime symptoms), the next step in management should focus on addressing the potential psychological or behavioral factors contributing to the abdominal pain.",
    ][:k_shot]

    shot_prompt_benchmark_parts = []
    for i, train_ex in enumerate(medqa_train_data):
        expl = explanations[i]
        shot_prompt_benchmark_parts.append(
            f"Question: {train_ex['question']}\n"
            f"A: {train_ex['options']['A']}\nB: {train_ex['options']['B']}\nC: {train_ex['options']['C']}\nD: {train_ex['options']['D']}\n"
            f"Explanation: {expl}\nAnswer: {train_ex['answer_idx']}\n\n"
        )
    shot_prompt_benchmark = "".join(shot_prompt_benchmark_parts)

    results_accumulator = []
    num_correct = 0
    num_no_answer = 0

    for test_idx, test_item in tqdm(enumerate(medqa_test), total=len(medqa_test), desc=f"Processing {strategy_name}"):
        initial_state_for_run: AgenticWorkflowState = {
            "question_text": test_item["question"],
            "options": test_item["options"],
            "correct_answer_idx": test_item["answer_idx"],
            "available_experts": EXPERT_DEFINITIONS,
            "num_experts_to_select": num_experts_to_select,
            "diversity_option": diversity_option,
            "is_benchmark_mode": True,
            "selected_experts_names": [],
            "initial_expert_responses": [],
            "supervisor_critique": None,
            "revised_expert_outputs": None,
            "final_synthesis": None,
            "expert_temperature": expert_temperature,
            "system_prompt": system_prompt_benchmark,
            "benchmark_shot_prompt": shot_prompt_benchmark,
        }

        compiled_agent = create_compiled_agent()

        try:
            final_run_state = compiled_agent.invoke(initial_state_for_run)
            agent_answer = final_run_state.get("final_synthesis", {}).get("answer", None)

            if agent_answer is None or agent_answer not in ["A", "B", "C", "D"]:
                num_no_answer += 1
            else:
                is_item_correct = agent_answer == test_item["answer_idx"]
                if is_item_correct:
                    num_correct += 1

            results_accumulator.append(
                {
                    "q_idx": test_idx,
                    "model_answer": agent_answer,
                    "is_correct": is_item_correct if agent_answer else False,
                }
            )
        except Exception as e:
            num_no_answer += 1
            results_accumulator.append(
                {"q_idx": test_idx, "model_answer": "WORKFLOW_ERROR", "is_correct": False, "errors": [str(e)]}
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


# --- EXECUCIÓ PRINCIPAL ---
if __name__ == "__main__":
    # Paràmetres de configuració per al benchmark
    NUM_QUESTIONS = 1273  # Nombre total de preguntes a processar
    NUM_QUESTIONS = 200  # Per proves ràpides
    K_SHOT = 5  # Nombre d'exemples few-shot per als experts
    EXPERT_TEMP = 0.4  # Temperatura per als models experts
    MIN_EXPERTS_TO_TEST = 1
    MAX_EXPERTS_TO_TEST = 5
    DIVERSITY_OPTIONS = ["Baixa", "Mitjana", "Alta"]  # Opcions de diversitat per al router

    all_strategy_results = []

    # Provar amb diferents opcions de diversitat i nombres d'experts
    for diversity_option in DIVERSITY_OPTIONS:
        for num_experts_routed in range(MIN_EXPERTS_TO_TEST, MAX_EXPERTS_TO_TEST + 1):
            strategy_run_name = f"Bench - Diversitat {diversity_option} - Top {num_experts_routed} Experts"
            print(f"\nStarting benchmark run: {strategy_run_name}")

            current_result = benchmark_agentic_workflow(
                strategy_name=strategy_run_name,
                num_experts_to_select=num_experts_routed,
                num_questions_to_run=NUM_QUESTIONS,
                diversity_option=diversity_option,
                k_shot=K_SHOT,
                expert_temperature=EXPERT_TEMP,
            )
            all_strategy_results.append(current_result)

            print(f"Finished benchmark run: {strategy_run_name}")

    # Mostrar resultats finals i gràfic
    if all_strategy_results:
        print("\n\n===== FINAL SUMMARY OF AGENTIC WORKFLOWS =====")
        header = f"{'Strategy':<30} | {'Accuracy':<10} | {'Correct':<10} | {'No Answer':<10} | {'Valid Total':<12}"
        print(header)
        print("-" * (len(header) + 2))  # Adjusted separator length
        for r_item in all_strategy_results:
            valid_total_calc = r_item["total"] - r_item["no_answer"]
            correct_disp = f"{r_item['correct']}/{valid_total_calc}" if valid_total_calc > 0 else "N/A"
            accuracy_disp = f"{r_item['accuracy']:.1f}%" if valid_total_calc > 0 else "N/A"
            print(
                f"{r_item['model_name']:<30} | {accuracy_disp:<10} | "
                f"{correct_disp:<10} | {r_item['no_answer']:<10} | {valid_total_calc:<12}"
            )
    else:
        print("No benchmark results were generated.")
