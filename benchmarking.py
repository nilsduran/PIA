from datasets import load_dataset
import re
import requests
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from google.api_core.exceptions import ResourceExhausted, TooManyRequests
import collections
import random


def generate_content(tuned_model_id, prompt, temperature=0.7, max_output_tokens=2048):
    """Generate content using the fine-tuned model API."""
    # Construct the endpoint URL
    url = f"https://generativelanguage.googleapis.com/v1/{tuned_model_id}:generateContent?key=AIzaSyDxk7cxcrDx3mcofYIosCggfkVbyHedO4w"

    # Set the request headers
    headers = {"Content-Type": "application/json"}

    # Prepare the payload
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_output_tokens},
    }

    # Make the POST request to the API endpoint
    response = requests.post(url, headers=headers, json=payload)

    # Check for a successful response
    if response.status_code == 200:
        response_json = response.json()
        try:
            # Extract the text from the response
            return response_json["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            return "Error: Could not extract response text from API response."
    else:
        return f"Error {response.status_code}: {response.text}"


def extract_answer(text):
    m = re.search(r"Answer:\s*([ABCD])\b", text, re.IGNORECASE)
    return m.group(1).upper() if m else None


def benchmark_model(model_id, model_name, num_questions=10, temperature=0.6, k_shot=5):
    """Benchmark a single model on multiple-choice medical questions."""
    print(f"\nBenchmarking model: {model_name}")

    # Load MedQA dataset
    medqa = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    medqa_train = []
    medqa_train = load_dataset("GBaker/MedQA-USMLE-4-options", split="train").shuffle(seed=42).select(range(k_shot))

    # Get examples for few-shot learning
    examples = medqa_train

    # Create system prompt (fix line length issue)
    system_prompt = (
        "You are a medical expert answering multiple-choice questions. "
        "Always output EXACTLY in this format, nothing more:\n"
        "Explanation: [Your brief but precise medical reasoning, no code or additional text]\n"
        "Answer: [A or B or C or D]\n"
    )

    # Example explanations (previously provided)
    explanations = [
        "A 35-year-old woman presents with various complaints and laboratory testing reveals the presence of anti-centromere antibodies. These antibodies are strongly associated with limited systemic sclerosis (also known as CREST syndrome), which includes Calcinosis, Raynaud's phenomenon, Esophageal dysmotility, Sclerodactyly, and Telangiectasia. Given this context, let's evaluate the given symptoms and signs:\nA: Pallor, cyanosis, and erythema of the hands - These are classic signs of Raynaud's phenomenon, commonly seen in limited systemic sclerosis.\nB: Blanching vascular abnormalities - These are also indicative of Raynaud's phenomenon, expected in this condition.\nC: Hypercoagulable state - Not typically a feature of limited systemic sclerosis; it is more associated with other connective tissue diseases or specific genetic disorders.\nD: Heartburn and regurgitation - These symptoms are consistent with esophageal dysmotility, a common feature of limited systemic sclerosis.",
        "The child's symptoms and physical examination findings—including slurred speech, frequent falls, pes cavus (high-arched feet), hammer toes, and kyphoscoliosis (curvature of the spine)—suggest a progressive neurological disorder. These signs are indicative of Friedreich's ataxia, a trinucleotide repeat disease.\nFriedreich's ataxia is caused by an expansion of the GAA trinucleotide repeat in the frataxin (FXN) gene. This condition typically presents with progressive ataxia, leading to difficulties with speech and coordination, as well as the skeletal abnormalities mentioned.\n",
        "The patient's presentation, including fever, headache, seizures, and altered behavior, along with the MRI findings of edema and hemorrhage in the left temporal lobe, strongly suggests herpes simplex encephalitis (HSE). HSE is a viral infection that primarily affects the temporal lobes and can cause significant brain edema.\nThe primary mechanism of edema in herpes simplex encephalitis is the breakdown of endothelial tight junctions. This disruption of the blood-brain barrier allows fluid to leak into the brain parenchyma, leading to vasogenic edema. This process is driven by the inflammatory response to the viral infection, which damages the endothelial cells and compromises the integrity of the blood-brain barrier.",
        "The patient's presentation includes shortness of breath, cough, severe lower limb edema, signs of right heart failure (jugular engorgement, hepatomegaly, hepatojugular reflux), and findings suggestive of pulmonary fibrosis. The physical examination and diagnostic tests (CT and echocardiogram) reveal right heart failure and severe pulmonary fibrosis. Cor pulmonale is a condition where right heart failure is caused by a primary lung disorder, typically chronic pulmonary hypertension. In this case, the severe pulmonary fibrosis is likely the underlying cause of the pulmonary hypertension, leading to cor pulmonale.",
        "Based on the information provided, the child's recurrent abdominal pain that occurs only at school, with no symptoms at home, and no abnormalities on physical examination or laboratory tests, suggests a functional cause. The child's symptoms are consistent with a functional abdominal pain disorder, possibly related to school avoidance or a behavioral component. Given that the child denies functional pain and there are no alarm symptoms (such as blood in the stool, weight loss, or nighttime symptoms), the next step in management should focus on addressing the potential psychological or behavioral factors contributing to the abdominal pain.",
    ]

    # Generate few-shot prompt
    shot_prompt = ""
    for i, ex in enumerate(examples):
        shot_prompt += (
            f"Question: {ex['question']}\n"
            f"A: {ex['options']['A']}\n"
            f"B: {ex['options']['B']}\n"
            f"C: {ex['options']['C']}\n"
            f"D: {ex['options']['D']}\n"
            f"Explanation: {explanations[i]}\n"
            f"Answer: {ex['answer_idx']}\n\n"
        )

    # Test examples
    test_examples = medqa.shuffle(seed=42).select(range(num_questions))
    correct = 0
    no_answer = 0
    responses = []

    # Process each test question
    for idx, ex in tqdm(enumerate(test_examples), total=len(test_examples)):
        # Generate the prompt with the question
        prompt = (
            system_prompt + "\n" + shot_prompt + f"Question: {ex['question']}\n"
            f"A: {ex['options']['A']}\n"
            f"B: {ex['options']['B']}\n"
            f"C: {ex['options']['C']}\n"
            f"D: {ex['options']['D']}\n"
            f"Explanation:"
        )

        response = generate_content(model_id, prompt, temperature=temperature, max_output_tokens=300)
        model_ans = extract_answer(response)

        # If no answer, try once more with lower temperature
        if model_ans is None:
            response = generate_content(model_id, prompt, temperature=temperature, max_output_tokens=500)
            model_ans = extract_answer(response)
            if model_ans is None:
                no_answer += 1
                responses.append(
                    {
                        "question_idx": idx,
                        "correct_answer": ex["answer_idx"],
                        "model_answer": None,
                        "response": response,
                        "is_correct": False,
                    }
                )
            else:
                is_correct = model_ans == ex["answer_idx"]
                if is_correct:
                    correct += 1
                responses.append(
                    {
                        "question_idx": idx,
                        "correct_answer": ex["answer_idx"],
                        "model_answer": model_ans,
                        "response": response,
                        "is_correct": is_correct,
                    }
                )
        else:
            is_correct = model_ans == ex["answer_idx"]
            if is_correct:
                correct += 1
            responses.append(
                {
                    "question_idx": idx,
                    "correct_answer": ex["answer_idx"],
                    "model_answer": model_ans,
                    "response": response,
                    "is_correct": is_correct,
                }
            )

    # Calculate results
    accuracy = correct / (len(test_examples) - no_answer) * 100
    print(f"No answer provided by model: {no_answer} out of {len(test_examples)}")
    print(f"Accuracy: {correct}/{len(test_examples)} = {accuracy:.1f}%")

    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "correct": correct,
        "no_answer": no_answer,
        "total": len(test_examples),
        "responses": responses,
    }


def plot_results(results):
    """Plot benchmark results for multiple models."""
    model_names = [r["model_name"] for r in results]
    accuracies = [r["accuracy"] for r in results]

    # Create a bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, accuracies, color="skyblue")

    # Add data labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height + 1, f"{height:.1f}%", ha="center", va="bottom")

    plt.title("Model Accuracy on Medical Questions", fontsize=14)
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.ylim(0, 100)  # Set y-axis from 0 to 100%
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Save the figure
    plt.savefig("model_benchmarks.png")


if __name__ == "__main__":
    # Llista de models de Google disponibles:
    # models/gemini-1.5-pro
    # models/gemini-1.5-flash
    # models/gemini-1.5-flash-001-tuning
    # models/gemini-1.5-flash-8b
    # models/gemini-2.5-pro-exp-03-25
    # models/gemini-2.5-pro-preview-03-25
    # models/gemini-2.5-flash-preview-04-17
    # models/gemini-2.5-flash-preview-04-17-thinking
    # models/gemini-2.5-pro-preview-05-06
    # models/gemini-2.0-flash
    # models/gemini-2.0-flash-lite
    # models/gemini-2.0-pro-exp
    # models/gemini-exp-1206
    # models/gemini-2.0-flash-thinking-exp
    # models/learnlm-2.0-flash-experimental
    # models/gemma-3-1b-it
    # models/gemma-3-4b-it
    # models/gemma-3-12b-it
    # models/gemma-3-27b-it

    # Available models
    MODELS = {
        "Gemini 1.5 Flash Tuning": "models/gemini-1.5-flash-001-tuning",
        "Medicina General": "tunedModels/medicinageneralcsv-q4i0ydc9l1uvxbzsxmii8",
        # "Ciències Bàsiques": "tunedModels/cincies-bsiques-5x23mkxv2ftipprirc4i4714",
        # "Patologia i Farmacologia": "tunedModels/patologia-i-farmacologia-3ipo0rdy5dkze8q",
        # "Cirurgia": "tunedModels/cirurgia-6rm1gub7hny7bzm3hjgghwcf3tws7ar",
        # "Pediatria i Ginecologia": "tunedModels/pediatria-i-ginecologia-q4n2dg2t5sweqdt9",
        # "Medicina General 2": "tunedModels/medicinageneral2-htffsvts97ttozkz18abl80",
        "Ciències Bàsiques": "tunedModels/ciencies-basiques-2-pfg4bpafqcay88df2kr8",
        "Patologia i Farmacologia": "tunedModels/patologia-farmacologia-2-8iy2ixmy5bluqzw",
        "Cirurgia": "tunedModels/cirurgia-2-2c1cy8nkr5ca5mui15tu4wtlpapp8",
        "Pediatria i Ginecologia": "tunedModels/pediatria-ginecologia-2-ss7f3iy509x7x43h",
    }

    # API key
    num_questions = 1273
    num_questions = 100
    all_results = []

    # Benchmark each model
    for model_name, model_id in MODELS.items():
        try:
            result = benchmark_model(model_id, model_name, num_questions)
            all_results.append(result)
        except (ResourceExhausted, TooManyRequests) as e:
            print(f"API rate limit reached for {model_name}: {e}. Sleeping before retry.")
            time.sleep(2)
            try:
                result = benchmark_model(model_id, model_name, num_questions)
                all_results.append(result)
            except Exception as e2:
                print(f"Retry failed for {model_name}: {e2}")
        except Exception as e:
            print(f"Error benchmarking {model_name}: {e}")

    # Add a "Majority Vote" model that picks the most common answer per question
    # collect correct answers from the first model's responses
    correct_map = {resp["question_idx"]: resp["correct_answer"] for resp in all_results[0]["responses"]}
    # build vote lists
    votes = {i: [] for i in range(num_questions)}
    for res in all_results:
        for resp in res["responses"]:
            ans = resp["model_answer"]
            if ans is not None:
                votes[resp["question_idx"]].append(ans)

    agg_correct = 0
    agg_no_answer = 0
    agg_responses = []
    for q_idx in range(num_questions):
        ans_list = votes[q_idx]
        if not ans_list:
            model_ans = None
            agg_no_answer += 1
        else:
            ctr = collections.Counter(ans_list)
            max_count = max(ctr.values())
            candidates = [a for a, cnt in ctr.items() if cnt == max_count]
            model_ans = random.choice(candidates)
        correct_ans = correct_map.get(q_idx)
        is_correct = model_ans == correct_ans
        if is_correct:
            agg_correct += 1
        agg_responses.append(
            {
                "question_idx": q_idx,
                "correct_answer": correct_ans,
                "model_answer": model_ans,
                "response": None,
                "is_correct": is_correct,
            }
        )

    agg_accuracy = agg_correct / (num_questions - agg_no_answer) * 100
    all_results.append(
        {
            "model_name": "Majority Vote",
            "accuracy": agg_accuracy,
            "correct": agg_correct,
            "no_answer": agg_no_answer,
            "total": num_questions,
            "responses": agg_responses,
        }
    )

    plot_results(all_results)
    print("\n\n===== SUMMARY =====")
    print(f"{'Model':<25} | {'Accuracy':<10} | {'Correct':<10} | {'No Answer':<10}")
    print("-" * 60)
    for r in sorted(all_results, key=lambda x: x["accuracy"], reverse=True):
        print(
            f"{r['model_name']:<25} | {r['accuracy']:.1f}% | "
            f"{r['correct']}/{r['total'] - r['no_answer']} | {r['no_answer']}"
        )
