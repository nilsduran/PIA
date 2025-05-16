from datasets import load_dataset
from xat_models_finetuned import generate_content, API_KEY, MODELS
import re


def extract_answer(text):
    m = re.search(r"Answer:\s*([ABCD])\b", text, re.IGNORECASE)
    return m.group(1).upper() if m else None


# Choose the model to benchmark
MODEL_NAME = "Medicina General"
MODEL_ID = MODELS[MODEL_NAME]

# Load MedQA dataset (test split)
medqa = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
medqa_train = load_dataset("GBaker/MedQA-USMLE-4-options", split="train").shuffle(seed=42).select(range(5))

# Prepare 5-shot context (first 5 examples)
examples = medqa_train

system_prompt = "You are a medical expert specializing in answering multiple-choice questions based on clinical scenarios. Each question has four options (A, B, C, D) and you must provide a detailed explanation for your choice. Your response should include the reasoning behind your answer and any relevant medical knowledge that supports it. The format of your response should be:\n\n"
system_prompt += "Explanation: [Your detailed explanation here]\nAnswer: [Your answer here (A, B, C, or D)]\n\n"
system_prompt = """
You are a medical expert answering multiple‑choice questions.
– Always output EXACTLY in this format, nothing more: 

Explanation: [Your brief but precise medical reasoning, no code or additional text]
Answer: [A or B or C or D]
"""
explanations = [
    "A 35-year-old woman presents with various complaints and laboratory testing reveals the presence of anti-centromere antibodies. These antibodies are strongly associated with limited systemic sclerosis (also known as CREST syndrome), which includes Calcinosis, Raynaud's phenomenon, Esophageal dysmotility, Sclerodactyly, and Telangiectasia. Given this context, let's evaluate the given symptoms and signs:\nA: Pallor, cyanosis, and erythema of the hands - These are classic signs of Raynaud's phenomenon, commonly seen in limited systemic sclerosis.\nB: Blanching vascular abnormalities - These are also indicative of Raynaud's phenomenon, expected in this condition.\nC: Hypercoagulable state - Not typically a feature of limited systemic sclerosis; it is more associated with other connective tissue diseases or specific genetic disorders.\nD: Heartburn and regurgitation - These symptoms are consistent with esophageal dysmotility, a common feature of limited systemic sclerosis.",
    "The child's symptoms and physical examination findings—including slurred speech, frequent falls, pes cavus (high-arched feet), hammer toes, and kyphoscoliosis (curvature of the spine)—suggest a progressive neurological disorder. These signs are indicative of Friedreich's ataxia, a trinucleotide repeat disease.\nFriedreich's ataxia is caused by an expansion of the GAA trinucleotide repeat in the frataxin (FXN) gene. This condition typically presents with progressive ataxia, leading to difficulties with speech and coordination, as well as the skeletal abnormalities mentioned.\n",
    "The patient's presentation, including fever, headache, seizures, and altered behavior, along with the MRI findings of edema and hemorrhage in the left temporal lobe, strongly suggests herpes simplex encephalitis (HSE). HSE is a viral infection that primarily affects the temporal lobes and can cause significant brain edema.\nThe primary mechanism of edema in herpes simplex encephalitis is the breakdown of endothelial tight junctions. This disruption of the blood-brain barrier allows fluid to leak into the brain parenchyma, leading to vasogenic edema. This process is driven by the inflammatory response to the viral infection, which damages the endothelial cells and compromises the integrity of the blood-brain barrier.",
    "The patient's presentation includes shortness of breath, cough, severe lower limb edema, signs of right heart failure (jugular engorgement, hepatomegaly, hepatojugular reflux), and findings suggestive of pulmonary fibrosis. The physical examination and diagnostic tests (CT and echocardiogram) reveal right heart failure and severe pulmonary fibrosis. Cor pulmonale is a condition where right heart failure is caused by a primary lung disorder, typically chronic pulmonary hypertension. In this case, the severe pulmonary fibrosis is likely the underlying cause of the pulmonary hypertension, leading to cor pulmonale.",
    "Based on the information provided, the child's recurrent abdominal pain that occurs only at school, with no symptoms at home, and no abnormalities on physical examination or laboratory tests, suggests a functional cause. The child's symptoms are consistent with a functional abdominal pain disorder, possibly related to school avoidance or a behavioral component. Given that the child denies functional pain and there are no alarm symptoms (such as blood in the stool, weight loss, or nighttime symptoms), the next step in management should focus on addressing the potential psychological or behavioral factors contributing to the abdominal pain.",
]


shot_prompt = ""
for i, ex in enumerate(examples):
    shot_prompt += (
        f"Question: {ex['question']}\n"
        f"A. {ex['options']['A']}\n"
        f"B. {ex['options']['B']}\n"
        f"C. {ex['options']['C']}\n"
        f"D. {ex['options']['D']}\n"
        f"Explanation: {explanations[i]}\n"
        f"Answer: {ex['answer_idx']}\n\n"
    )

# Evaluate on the next
test_examples = medqa.shuffle(seed=42).select(range(10))
correct = 0

for idx, ex in enumerate(test_examples):
    prompt = (
        system_prompt + "\n" + shot_prompt + f"Question: {ex['question']}\n"
        f"A: {ex['options']['A']}\n"
        f"B: {ex['options']['B']}\n"
        f"C: {ex['options']['C']}\n"
        f"D: {ex['options']['D']}\n"
        f"Explanation:"
    )

    # print(f"\n--- Example {idx + 1} ---")
    # print(f"Question: {ex['question']}")
    # print(
    #     f"Options: A) {ex['options']['A']}, B) {ex['options']['B']}, C) {ex['options']['C']}, D) {ex['options']['D']}"
    # )
    print(f"Prompt:\n-------------------------\n{prompt}")
    print("end prompt\n------------------------")
    print(f"Correct answer: {ex['answer']}")
    response = generate_content(API_KEY, MODEL_ID, prompt, temperature=0.0, max_output_tokens=50)
    print(f"Model response: {response}")
    print(f"Length of response: {len(response)} characters")

    model_ans = extract_answer(response)

    if model_ans == ex["answer"]:
        correct += 1
        print("✅ Correct")
    else:
        print("❌ Incorrect")

print(f"\nAccuracy: {correct}/{len(test_examples)} = {correct/len(test_examples)*100:.1f}%")
