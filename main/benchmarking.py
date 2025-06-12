import matplotlib.pyplot as plt
import time
from google.api_core.exceptions import ResourceExhausted, TooManyRequests
import collections
import random
from scripts.funcions_auxiliars import benchmark_model


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
    plt.savefig("plots/model_benchmarks.png")


if __name__ == "__main__":
    MODELS = {
        "Medicina General": "tunedModels/medicinageneralcsv-q4i0ydc9l1uvxbzsxmii8",
        "Ciències Bàsiques": "tunedModels/ciencies-basiques-2-pfg4bpafqcay88df2kr8",
        "Patologia i Farmacologia": "tunedModels/patologia-farmacologia-2-8iy2ixmy5bluqzw",
        "Cirurgia": "tunedModels/cirurgia-2-2c1cy8nkr5ca5mui15tu4wtlpapp8",
        "Pediatria i Ginecologia": "tunedModels/pediatria-ginecologia-2-ss7f3iy509x7x43h",
    }

    num_questions = 1273  # MedQA-USMLE-4-options test complet
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
