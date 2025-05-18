import matplotlib.pyplot as plt
from statsmodels.stats.contingency_tables import mcnemar
from benchmarking import benchmark_model
import numpy as np


if __name__ == "__main__":
    MODELS = {
        "Medicina General": [
            "tunedModels/medicinageneralcsv-q4i0ydc9l1uvxbzsxmii8",
            "tunedModels/medicinageneral2-htffsvts97ttozkz18abl80",
        ],
        "Ciències Bàsiques": [
            "tunedModels/cincies-bsiques-5x23mkxv2ftipprirc4i4714",
            "tunedModels/ciencies-basiques-2-pfg4bpafqcay88df2kr8",
        ],
        "Patologia i Farmacologia": [
            "tunedModels/patologia-i-farmacologia-3ipo0rdy5dkze8q",
            "tunedModels/patologia-farmacologia-2-8iy2ixmy5bluqzw",
        ],
        "Cirurgia": [
            "tunedModels/cirurgia-6rm1gub7hny7bzm3hjgghwcf3tws7ar",
            "tunedModels/cirurgia-2-2c1cy8nkr5ca5mui15tu4wtlpapp8",
        ],
        "Pediatria i Ginecologia": [
            "tunedModels/pediatria-i-ginecologia-q4n2dg2t5sweqdt9",
            "tunedModels/pediatria-ginecologia-2-ss7f3iy509x7x43h",
        ],
    }
    num_questions = 100
    all_model_names, all_acc1, all_acc2 = [], [], []

    for model_name, model_paths in MODELS.items():
        results_v1 = benchmark_model(model_id=model_paths[0], model_name=f"{model_name}", num_questions=num_questions)
        results_v2 = benchmark_model(model_id=model_paths[1], model_name=f"{model_name} 2", num_questions=num_questions)

        resp1 = {r["question_idx"]: r for r in results_v1["responses"]}
        resp2 = {r["question_idx"]: r for r in results_v2["responses"]}

        # --------- 2. Calcula head-to-head -----------
        both_correct = 0
        v1_only = 0
        v2_only = 0
        both_incorrect = 0

        for q in range(num_questions):
            c1 = resp1[q]["is_correct"]
            c2 = resp2[q]["is_correct"]
            if c1 and c2:
                both_correct += 1
            elif c1 and not c2:
                v1_only += 1
            elif not c1 and c2:
                v2_only += 1
            else:
                both_incorrect += 1

        print("Head-to-Head results:")
        print(f"  Both correct   : {both_correct}")
        print(f"  V1 only correct: {v1_only}")
        print(f"  V2 only correct: {v2_only}")
        print(f"  Both incorrect : {both_incorrect}")

        # --------- 3. McNemar’s test -----------
        # La matriu de contingència per McNemar ha de ser:
        table = [[both_correct, v1_only], [v2_only, both_incorrect]]
        res = mcnemar(table, exact=True)
        print("\nMcNemar's test (exact):")
        print(f"  Statistic = {res.statistic}")
        print(f"  p-value   = {res.pvalue:.4f}")

        alpha = 0.05
        if res.pvalue < alpha:
            print("  ⇒ Diferència significativa (p < 0.05), V2 és millor o pitjor que V1 segons la taula discordant.")
        else:
            print("  ⇒ No hi ha evidència suficient per dir que V2 difereixi de V1 (p ≥ 0.05).")

        # --------- 4. Comparativa d’accuracy -----------
        acc1 = results_v1["correct"] / max(results_v1["total"] - results_v1["no_answer"], 1)
        acc2 = results_v2["correct"] / max(results_v2["total"] - results_v2["no_answer"], 1)
        print("\nAccuracy summary:")
        print(
            f"  V1 Accuracy = {acc1*100:.1f}% ({results_v1['correct']}/{results_v1['total']-results_v1['no_answer']})"
        )
        print(
            f"  V2 Accuracy = {acc2*100:.1f}% ({results_v2['correct']}/{results_v2['total']-results_v2['no_answer']})"
        )

        # --------- 5. Gràfic de comparació -----------
        plt.figure(figsize=(8, 6))
        acc1_pct = acc1 * 100
        acc2_pct = acc2 * 100
        diff_pct = abs(acc2_pct - acc1_pct)

        # Stacked bar: base = smaller value, top = difference
        bottom_val = min(acc1_pct, acc2_pct)
        plt.bar(
            ["Accuracy"],
            [bottom_val],
            color="skyblue",
            edgecolor="black",
            linewidth=1.2,
            width=0.6,
        )
        plt.bar(
            ["Accuracy"],
            [diff_pct],
            bottom=[bottom_val],
            color="lightblue",
            edgecolor="black",
            linewidth=1.2,
            width=0.6,
        )

        # Draw red arrow from acc1 to acc2
        plt.annotate(
            "",
            xy=(0, acc2_pct),
            xytext=(0, acc1_pct),
            arrowprops=dict(color="red", arrowstyle="->", lw=2),
        )

        plt.title(f"Accuracy Comparison: {model_name}")
        plt.ylabel("Accuracy (%)")
        plt.ylim(0, 100)
        plt.xticks([0], [""])
        plt.text(0, -5, f"V1: {acc1_pct:.1f}%\nV2: {acc2_pct:.1f}%", ha="center", va="top")
        plt.tight_layout()
        plt.savefig(f"accuracy_comparison_{model_name.replace(' ', '_').lower()}.png")

        # Accumulate accuracy values across models
        all_model_names.append(model_name)
        all_acc1.append(acc1_pct)
        all_acc2.append(acc2_pct)

        # Once we've processed all models, draw a combined chart with the same visual style
        if len(all_model_names) == len(MODELS):
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(all_model_names))
            width = 0.6

            for i, (name, acc1, acc2) in enumerate(zip(all_model_names, all_acc1, all_acc2)):
                # For each model, create a stacked bar like in the single model charts
                bottom_val = min(acc1, acc2)
                diff_val = abs(acc2 - acc1)

                # Base bar (smaller value)
                ax.bar([i], [bottom_val], width=width, color="skyblue", edgecolor="black", linewidth=1.2)

                # Top bar (difference)
                ax.bar(
                    [i],
                    [diff_val],
                    bottom=[bottom_val],
                    width=width,
                    color="lightblue",
                    edgecolor="black",
                    linewidth=1.2,
                )

                # Add arrow indicating direction of change
                ax.annotate(
                    "",
                    xy=(i, acc2),
                    xytext=(i, acc1),
                    arrowprops=dict(color="red" if acc2 < acc1 else "green", arrowstyle="->", lw=2),
                )

            # Set up the chart
            ax.set_xticks(x)
            ax.set_xticklabels(all_model_names, rotation=45, ha="right")
            ax.set_ylabel("Accuracy (%)")
            ax.set_ylim(0, 100)
            ax.set_title("Accuracy Comparison Across Models (V1 vs V2)")
            plt.tight_layout()
            plt.savefig("accuracy_comparison_all_models.png", dpi=300)
            plt.close()
