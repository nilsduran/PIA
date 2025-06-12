import matplotlib.pyplot as plt
import numpy as np
import time
import json
from main.benchmarking import benchmark_model


def optimize_temperature(model_id, model_name, num_questions=100):
    """Test a range of temperatures to find the optimal setting."""
    print(f"Finding optimal temperature for model: {model_name}")

    # Test both linear and geometric ranges
    # Linear range from 0.1 to 2.0
    temperatures = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]
    results = []

    for temp in temperatures:
        try:
            result = benchmark_model(model_id, model_name, num_questions, temperature=temp, k_shot=5)
            result["temperature"] = temp
            results.append(result)
            # Save results after each temperature in case of interruption
            with open(f"temp_results_{model_name.replace(' ', '_')}.json", "w") as f:
                json.dump(results, f, indent=2)
            # Small delay to avoid API rate limiting
            time.sleep(1)
        except Exception as e:
            print(f"Error benchmarking at temperature {temp}: {e}")
            # Continue with the next temperature if one fails
            continue

    # Plot the results
    plot_temperature_results(results, model_name)

    # Find the optimal temperature
    optimal_result = max(results, key=lambda x: x["accuracy"])
    print(f"\nOptimal temperature for {model_name}: {optimal_result['temperature']}")
    print(f"Accuracy at optimal temperature: {optimal_result['accuracy']:.1f}%")

    return results


def plot_temperature_results(results, model_name):
    """Plot accuracy vs temperature using bar charts whose x‐positions and widths
    reflect the actual temperature values."""

    # Sort results by temperature
    results = sorted(results, key=lambda x: x["temperature"])
    temperatures = [r["temperature"] for r in results]
    accuracies = [r["accuracy"] for r in results]
    no_answers = [r["no_answer"] / r["total"] * 100 for r in results]

    # Compute bar widths from the spacing between temperature points
    temps_arr = np.array(temperatures)
    diffs = np.diff(temps_arr)
    # for the last bar, reuse the last interval
    diffs = np.append(diffs, diffs[-1] if len(diffs) else 0.1)
    bar_widths = diffs * 0.8  # scale factor so bars don't touch

    plt.figure(figsize=(12, 8))

    # Accuracy bar chart
    plt.subplot(2, 1, 1)
    plt.bar(temps_arr, accuracies, width=bar_widths, color="skyblue", edgecolor="black", align="center")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.title(f"Accuracy vs Temperature for {model_name}", fontsize=16)
    plt.ylabel("Accuracy (%)", fontsize=14)
    plt.ylim(0, 100)

    # Mark the optimal bar
    max_idx = int(np.argmax(accuracies))
    opt_temp = temperatures[max_idx]
    max_acc = accuracies[max_idx]
    plt.axvline(opt_temp, color="green", linestyle="--", alpha=0.7)
    plt.annotate(
        f"Optimal: {opt_temp:.1f} ({max_acc:.1f}%)",
        xy=(opt_temp, max_acc),
        xytext=(opt_temp + 0.1, max_acc - 10),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=8),
    )

    # No-answer bar chart
    plt.subplot(2, 1, 2)
    plt.bar(temps_arr, no_answers, width=bar_widths, color="salmon", edgecolor="black", align="center")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.title(f"No Answer Rate vs Temperature for {model_name}", fontsize=16)
    plt.xlabel("Temperature", fontsize=14)
    plt.ylabel("No Answer Rate (%)", fontsize=14)
    plt.ylim(0, 100)

    plt.tight_layout()
    plt.savefig(f"plots/temperature_optimization_{model_name.replace(' ', '_')}.png")


if __name__ == "__main__":
    # Available models
    MODELS = {
        "Medicina General": "tunedModels/medicinageneralcsv-q4i0ydc9l1uvxbzsxmii8",
        "Ciències Bàsiques": "tunedModels/ciencies-basiques-2-pfg4bpafqcay88df2kr8",
        "Patologia i Farmacologia": "tunedModels/patologia-farmacologia-2-8iy2ixmy5bluqzw",
        "Cirurgia": "tunedModels/cirurgia-2-2c1cy8nkr5ca5mui15tu4wtlpapp8",
        "Pediatria i Ginecologia": "tunedModels/pediatria-ginecologia-2-ss7f3iy509x7x43h",
    }

    NUM_QUESTIONS = 20

    # Run optimization for all models
    all_results = {}
    for name, model_id in MODELS.items():
        print(f"\nOptimizing model: {name}")
        all_results[name] = optimize_temperature(model_id, name, NUM_QUESTIONS)

    # Print each model's ideal temperature
    for name, res in all_results.items():
        opt = max(res, key=lambda x: x["accuracy"])
        print(f"Model {name}: optimal temperature {opt['temperature']}," f" accuracy {opt['accuracy']:.1f}%")

    # Build accuracy matrix per temperature
    temps = [r["temperature"] for r in all_results[next(iter(all_results))]]
    acc_matrix = {t: [] for t in temps}
    for res in all_results.values():
        for r in res:
            acc_matrix[r["temperature"]].append(r["accuracy"])

    # Boxplot of accuracies and no-answer rates across models for each temperature
    # Prepare data matrices
    temps = sorted([t for t in temps])  # Ensure temperatures are sorted
    temps_str = [str(t) for t in temps]
    acc_data = [acc_matrix[t] for t in temps]
    # Build no-answer percentage matrix
    no_ans_matrix = {t: [] for t in temps}
    for res in all_results.values():
        for r in res:
            no_rate = r["no_answer"] / r["total"] * 100
            no_ans_matrix[r["temperature"]].append(no_rate)
    no_data = [no_ans_matrix[t] for t in temps]

    # Create subplots: top for accuracy, bottom for no-answer rate
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Set positions based on actual temperature values
    positions = temps  # Use actual temperature values for x-axis

    # Accuracy violin plot with proper spacing
    parts0 = axes[0].violinplot(acc_data, positions=positions, widths=0.1 * np.ones(len(temps)))
    for pc in parts0["bodies"]:
        pc.set_facecolor("skyblue")
        pc.set_edgecolor("black")

    # Add individual points
    for i, temp in enumerate(temps):
        x = np.random.normal(temp, 0.01, size=len(acc_matrix[temp]))
        axes[0].scatter(x, acc_matrix[temp], alpha=0.6, s=20, color="darkblue")

    # Mean line
    acc_means = [np.mean(acc_matrix[t]) for t in temps]
    axes[0].plot(temps, acc_means, "-o", color="red", label="Mean Accuracy")
    axes[0].set_title("Accuracy Distribution Across Models per Temperature")
    axes[0].set_xlabel("Temperature")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # No-answer rate violin plot with proper spacing
    parts1 = axes[1].violinplot(no_data, positions=positions, widths=0.1 * np.ones(len(temps)))
    for pc in parts1["bodies"]:
        pc.set_facecolor("salmon")
        pc.set_edgecolor("black")

    # Add individual points
    for i, temp in enumerate(temps):
        x = np.random.normal(temp, 0.01, size=len(no_ans_matrix[temp]))
        axes[1].scatter(x, no_ans_matrix[temp], alpha=0.6, s=20, color="darkred")

    # Mean line
    no_means = [np.mean(no_ans_matrix[t]) for t in temps]
    axes[1].plot(temps, no_means, "-o", color="blue", label="Mean No-Answer %")
    axes[1].set_title("No-Answer Rate Distribution Across Models per Temperature")
    axes[1].set_xlabel("Temperature")
    axes[1].set_ylabel("No Answer Rate (%)")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("plots/distribution_per_temperature.png")
