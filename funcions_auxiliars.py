from datasets import load_dataset
import re
from tqdm import tqdm
from google import genai
from google.genai import types
from typing import Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer


def generate_content(
    model_id: str,
    prompt: str,
    temperature: float = 0.7,
    max_output_tokens: int = 500,
) -> str:
    """
    Genera contingut utilitzant la API adequada segons el model_id:
    - Models Gemini (2.5 Flash Preview) usant el SDK de Google.
    - Models afinats antics (p. ex., tunedModels/medicinageneralcsv-...) usant requests.
    """
    api_key = "AIzaSyDxk7cxcrDx3mcofYIosCggfkVbyHedO4w"
    client = genai.Client(api_key=api_key)

    # Instanciem el model específic
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]

    if model_id == "gemini-2.5-flash-preview-05-20":
        generate_content_config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=0),  # Disable thinking budget for now
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
            ],
        )
    else:
        generate_content_config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
            ],
        )

    response = client.models.generate_content(
        model=model_id,
        contents=contents,
        config=generate_content_config,
    )

    return response.text


def _call_single_expert_llm(
    expert_model_id: str,
    question_text: str,
    temperature: float,
    is_benchmark_mode: bool,
    options: Optional[Dict[str, str]],  # For benchmark
    system_prompt: Optional[str],
    benchmark_shot_prompt: Optional[str],
    critique_to_include: Optional[str] = None,
) -> Dict[str, str]:
    """Calls a single expert LLM and extracts explanation and conclusion/answer."""

    base_system_prompt = system_prompt or "You are a helpful AI assistant."  # Default

    critique_prompt_addition = ""
    if critique_to_include:
        critique_prompt_addition = (
            f"\n\nPlease consider the following critique of initial analyses and refine your response:\n"
            f"CRITIQUE:\n{critique_to_include}\n\n"
            f"Your refined analysis based on this critique:\n"
        )

    if is_benchmark_mode:
        current_system_prompt = base_system_prompt
        current_shot_prompt = benchmark_shot_prompt or ""
        prompt = (
            f"{current_system_prompt}\n{current_shot_prompt}\n"
            f"Question: {question_text}\n"
            f"A: {options.get('A', 'N/A')}\nB: {options.get('B', 'N/A')}\n"
            f"C: {options.get('C', 'N/A')}\nD: {options.get('D', 'N/A')}\n"
            f"{critique_prompt_addition}"  # Add critique here
            f"Explanation:"
        )
        max_tok, retry_max_tok = 400, 700
    else:  # UI/Conversational mode
        prompt = (
            f"{base_system_prompt}\nYou are an AI expert in medicine and healthcare.\n"
            f"Analyze the following case/question from your perspective:\n\n"
            f"Case/Question: {question_text}\n\n"
            f"{critique_prompt_addition}"  # Add critique here
            f"Provide a concise analysis (around 100-200 words), including key insights and potential considerations. Format:\n"
            f"Explanation: [Your analysis]\nConclusion: [Your main conclusion/summary]"
        )
        max_tok, retry_max_tok = 1000, 2000

    raw_response = generate_content(expert_model_id, prompt, temperature=temperature, max_output_tokens=max_tok)
    explanation, answer_or_conclusion = extract_explanation_and_answer_or_conclusion(raw_response)

    if (
        answer_or_conclusion is None and is_benchmark_mode and not critique_to_include
    ):  # Only retry if no answer in benchmark mode on first pass
        raw_response = generate_content(
            expert_model_id, prompt, temperature=max(0.1, temperature - 0.2), max_output_tokens=retry_max_tok
        )
        explanation, answer_or_conclusion = extract_explanation_and_answer_or_conclusion(raw_response)

    response_key = "answer" if is_benchmark_mode else "conclusion"
    return {
        response_key: answer_or_conclusion or ("No answer." if is_benchmark_mode else "No conclusion."),
        "explanation": explanation or "No explanation.",
    }


def extract_answer(text: str) -> Optional[str]:
    m = re.search(r"Answer:\s*([ABCD])\b", text, re.IGNORECASE)
    return m.group(1).upper() if m else None


def extract_conclusion(text: str) -> Optional[str]:
    # De Conclusion fins Explanation o Answer
    m = re.search(r"Conclusion:(.*?)(?=\n(?:Explanation|Answer):|$)", text, re.DOTALL | re.IGNORECASE | re.UNICODE)
    # Si no troba Explanation/Answer després de Conclusion, busca només Conclusion
    if not m:
        m = re.search(r"Conclusion:(.*)", text, re.DOTALL | re.IGNORECASE | re.UNICODE)
    return m.group(1).strip() if m else None


def extract_explanation(text: str) -> Optional[str]:
    # De Explanation fins Conclusion o Answer
    m = re.search(r"Explanation:(.*?)(?=\n(?:Answer|Conclusion):|$)", text, re.DOTALL | re.IGNORECASE | re.UNICODE)
    # Si no troba Conclusion/Answer després de Explanation, busca només Explanation
    if not m:
        m = re.search(r"Explanation:(.*)", text, re.DOTALL | re.IGNORECASE | re.UNICODE)
    return m.group(1).strip() if m else None


def extract_explanation_and_answer_or_conclusion(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extracts explanation and answer/conclusion from the response text."""
    explanation = extract_explanation(text)

    answer_or_conclusion = extract_answer(text)

    if not answer_or_conclusion:  # Si no hi ha resposta, intenta amb Conclusion
        answer_or_conclusion = extract_conclusion(text)
    elif explanation is None and answer_or_conclusion is None and text.strip():
        # Si no s'ha trobat ni explicació ni resposta, utilitza el text complet i resposta amb None
        explanation = text.strip()

    return explanation, answer_or_conclusion


def benchmark_model(model_id, model_name, num_questions=10, temperature=0.6, k_shot=5):
    """Benchmark a single model on multiple-choice medical questions."""
    print(f"\nBenchmarking model: {model_name}")

    medqa = load_dataset("GBaker/MedQA-USMLE-4-options", split="test").shuffle(seed=42).select(range(num_questions))
    medqa_train = load_dataset("GBaker/MedQA-USMLE-4-options", split="train").shuffle(seed=42).select(range(k_shot))
    examples = medqa_train

    system_prompt = (
        "You are a medical expert answering multiple-choice questions. "
        "Always output EXACTLY in this format, nothing more:\n"
        "Explanation: [Your brief but precise medical reasoning, no code or additional text]\n"
        "Answer: [A or B or C or D]\n"
    )

    explanations_few_shot = [
        "A 35-year-old woman presents with various complaints and laboratory testing reveals the presence of anti-centromere antibodies. These antibodies are strongly associated with limited systemic sclerosis (also known as CREST syndrome), which includes Calcinosis, Raynaud's phenomenon, Esophageal dysmotility, Sclerodactyly, and Telangiectasia. Given this context, let's evaluate the given symptoms and signs:\nA: Pallor, cyanosis, and erythema of the hands - These are classic signs of Raynaud's phenomenon, commonly seen in limited systemic sclerosis.\nB: Blanching vascular abnormalities - These are also indicative of Raynaud's phenomenon, expected in this condition.\nC: Hypercoagulable state - Not typically a feature of limited systemic sclerosis; it is more associated with other connective tissue diseases or specific genetic disorders.\nD: Heartburn and regurgitation - These symptoms are consistent with esophageal dysmotility, a common feature of limited systemic sclerosis.",
        "The child's symptoms and physical examination findings—including slurred speech, frequent falls, pes cavus (high-arched feet), hammer toes, and kyphoscoliosis (curvature of the spine)—suggest a progressive neurological disorder. These signs are indicative of Friedreich's ataxia, a trinucleotide repeat disease.\nFriedreich's ataxia is caused by an expansion of the GAA trinucleotide repeat in the frataxin (FXN) gene. This condition typically presents with progressive ataxia, leading to difficulties with speech and coordination, as well as the skeletal abnormalities mentioned.",
        "The patient's presentation, including fever, headache, seizures, and altered behavior, along with the MRI findings of edema and hemorrhage in the left temporal lobe, strongly suggests herpes simplex encephalitis (HSE). HSE is a viral infection that primarily affects the temporal lobes and can cause significant brain edema.\nThe primary mechanism of edema in herpes simplex encephalitis is the breakdown of endothelial tight junctions. This disruption of the blood-brain barrier allows fluid to leak into the brain parenchyma, leading to vasogenic edema. This process is driven by the inflammatory response to the viral infection, which damages the endothelial cells and compromises the integrity of the blood-brain barrier.",
        "The patient's presentation includes shortness of breath, cough, severe lower limb edema, signs of right heart failure (jugular engorgement, hepatomegaly, hepatojugular reflux), and findings suggestive of pulmonary fibrosis. The physical examination and diagnostic tests (CT and echocardiogram) reveal right heart failure and severe pulmonary fibrosis. Cor pulmonale is a condition where right heart failure is caused by a primary lung disorder, typically chronic pulmonary hypertension. In this case, the severe pulmonary fibrosis is likely the underlying cause of the pulmonary hypertension, leading to cor pulmonale.",
        "Based on the information provided, the child's recurrent abdominal pain that occurs only at school, with no symptoms at home, and no abnormalities on physical examination or laboratory tests, suggests a functional cause. The child's symptoms are consistent with a functional abdominal pain disorder, possibly related to school avoidance or a behavioral component. Given that the child denies functional pain and there are no alarm symptoms (such as blood in the stool, weight loss, or nighttime symptoms), the next step in management should focus on addressing the potential psychological or behavioral factors contributing to the abdominal pain.",
    ][:k_shot]

    shot_prompt = ""
    for i, ex in enumerate(examples):
        shot_prompt += (
            f"Question: {ex['question']}\n"
            f"A: {ex['options']['A']}\n"
            f"B: {ex['options']['B']}\n"
            f"C: {ex['options']['C']}\n"
            f"D: {ex['options']['D']}\n"
            f"Explanation: {explanations_few_shot[i]}\n"
            f"Answer: {ex['answer_idx']}\n\n"
        )

    correct = 0
    no_answer_count = 0
    responses_data = []

    for idx, ex in tqdm(enumerate(medqa), total=len(medqa), desc=f"Benchmarking {model_name}"):
        prompt = (
            system_prompt + "\n" + shot_prompt + f"Question: {ex['question']}\n"
            f"A: {ex['options']['A']}\n"
            f"B: {ex['options']['B']}\n"
            f"C: {ex['options']['C']}\n"
            f"D: {ex['options']['D']}\n"
            f"Explanation:"
        )

        response_text = generate_content(model_id, prompt, temperature=temperature, max_output_tokens=300)
        model_explanation, model_ans = extract_explanation_and_answer_or_conclusion(response_text)

        if model_ans is None:  # Si no hi ha resposta, intenta amb més tokens
            response_text = generate_content(model_id, prompt, temperature=temperature, max_output_tokens=500)
            model_ans = extract_answer(response_text)
            model_explanation = extract_explanation(response_text)
            if model_ans is None:
                no_answer_count += 1
                responses_data.append(
                    {
                        "question_idx": idx,
                        "question_text": ex["question"],
                        "correct_answer": ex["answer_idx"],
                        "model_answer": None,
                        "model_explanation": model_explanation,
                        "raw_response": response_text,
                        "is_correct": False,
                    }
                )
                continue  # Si encara no dona resposta, salta a la següent pregunta

        is_correct = model_ans == ex["answer_idx"]
        if is_correct:
            correct += 1

        responses_data.append(
            {
                "question_idx": idx,
                "question_text": ex["question"],
                "correct_answer": ex["answer_idx"],
                "model_answer": model_ans,
                "model_explanation": model_explanation,
                "raw_response": response_text,
                "is_correct": is_correct,
            }
        )

    valid_responses = num_questions - no_answer_count
    accuracy = (correct / valid_responses * 100) if valid_responses > 0 else 0
    print(f"No answer provided by model {model_name}: {no_answer_count} out of {num_questions} questions.")
    print(f"Accuracy for {model_name}: {correct}/{valid_responses} = {accuracy:.1f}%")

    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "correct_count": correct,
        "no_answer_count": no_answer_count,
        "total_questions": num_questions,
        "responses_data": responses_data,
    }


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def get_embedding(text):
    """Genera l'embedding per a un text donat."""
    return embedding_model.encode(text)
