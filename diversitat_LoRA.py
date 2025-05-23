from matplotlib import pyplot as plt
from scipy.stats import pearsonr
import time
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import seaborn as sns
import numpy as np
import collections
from tqdm import tqdm
from funcions_auxiliars import (
    benchmark_model,
    get_embedding,
)
from googleapiclient.errors import HttpError


def calculate_model_diversity_scores(all_results):
    """
    Calcula un score de diversitat per a cada model.
    L'score representa com de diferents són les explicacions d'un model
    respecte a les explicacions dels altres models, de mitjana.
    """
    print("\nCalculating model diversity scores...")

    num_questions = all_results[0]["total_questions"]
    model_diversity_scores = {res["model_name"]: [] for res in all_results}

    for q_idx in tqdm(range(num_questions), desc="Processing questions for diversity"):
        question_explanations = []  # Llista d'explicacions (text) per a la pregunta actual, una per model
        valid_model_indices_for_q = []  # Índexs dels models que van donar una explicació vàlida

        for model_idx, result in enumerate(all_results):
            # Troba la resposta/explicació per a q_idx
            # Les respostes estan ordenades per l'índex original del test_examples
            # així que podem assumir que responses_data[q_idx] és la correcta si tots els q_idx són iguals.
            # Per seguretat, podríem buscar per 'question_idx' si no fos així.
            current_q_response_data = next((r for r in result["responses_data"] if r["question_idx"] == q_idx), None)
            if current_q_response_data and current_q_response_data["model_explanation"]:
                question_explanations.append(current_q_response_data["model_explanation"])
                valid_model_indices_for_q.append(model_idx)
            else:
                question_explanations.append(None)  # Placeholder per mantenir l'ordre

        if len(valid_model_indices_for_q) < 2:  # Necessitem almenys dues explicacions per comparar
            continue

        # Genera embeddings per a les explicacions vàlides d'aquesta pregunta
        valid_explanations_texts = [question_explanations[i] for i in valid_model_indices_for_q]
        embeddings = [get_embedding(exp) for exp in valid_explanations_texts]

        # Filtra embeddings None (si get_embedding retorna None per alguna raó)
        # i manté la correspondència amb valid_model_indices_for_q
        final_embeddings = []
        final_model_indices = []
        for i, emb in enumerate(embeddings):
            if emb is not None:
                final_embeddings.append(emb)
                final_model_indices.append(valid_model_indices_for_q[i])

        if len(final_embeddings) < 2:
            continue

        embeddings_matrix = np.array(final_embeddings)
        sim_matrix = cosine_similarity(embeddings_matrix)

        # Calcula la diversitat per a cada model que va contribuir amb una explicació vàlida
        for i, model_idx_k in enumerate(final_model_indices):  # model_idx_k és l'índex global del model
            model_k_name = all_results[model_idx_k]["model_name"]

            # Distàncies del model_k a tots els altres models (l != k) per aquesta pregunta
            distances_k_vs_others = []
            for j, model_idx_l in enumerate(final_model_indices):
                if i == j:  # No comparar amb si mateix
                    continue
                # La similitud està en sim_matrix[i][j]
                # Distància = 1 - Similitud
                distances_k_vs_others.append(1 - sim_matrix[i, j])

            if distances_k_vs_others:  # Si hi ha altres models amb què comparar
                avg_dist_k_vs_others = np.mean(distances_k_vs_others)
                model_diversity_scores[model_k_name].append(avg_dist_k_vs_others)

    # Calcula la mitjana de les distàncies per a cada model
    final_model_diversities = {}
    for model_name, scores in model_diversity_scores.items():
        if scores:
            final_model_diversities[model_name] = np.mean(scores)
        else:
            final_model_diversities[model_name] = 0  # O np.nan si prefereixes
            print(f"Warning: No valid diversity scores for model {model_name} to average.")

    return final_model_diversities


def plot_diversity_accuracy_correlation(model_accuracies, model_diversities, all_results_list):
    """Plota la correlació entre la diversitat i l'accuracy dels models."""
    model_names = [res["model_name"] for res in all_results_list]

    accuracies = [model_accuracies.get(name, 0) for name in model_names]
    diversities = [model_diversities.get(name, 0) for name in model_names]

    if not accuracies or not diversities or len(accuracies) < 2:
        print("Not enough data to plot correlation.")
        return

    # Càlcul de la correlació de Pearson
    corr, p_value = pearsonr(diversities, accuracies)
    print(f"\nCorrelation between Model Diversity and Accuracy:")
    print(f"Pearson correlation coefficient: {corr:.3f}")
    print(f"P-value: {p_value:.3f}")

    # Gràfic de dispersió
    plt.figure(figsize=(10, 8))
    for i, model_name in enumerate(model_names):
        plt.scatter(diversities[i], accuracies[i], label=model_name, s=100)
        plt.text(diversities[i] + 0.001, accuracies[i] + 0.1, model_name, fontsize=9)

    # Línia de tendència (regressió lineal simple)
    if len(diversities) >= 2:  # np.polyfit necessita almenys 2 punts
        m, b = np.polyfit(diversities, accuracies, 1)  # m = pendent, b = intercepció
        plt.plot(np.array(diversities), m * np.array(diversities) + b, "-", color="grey", alpha=0.5)

    plt.title("Diversitat vs. Accuracy")
    plt.xlabel("Model Explanation Diversity Score")
    plt.ylabel("Model Accuracy (%)")
    plt.legend(loc="best", bbox_to_anchor=(1.05, 1))  # LLegenda fora del plot si hi ha molts models
    plt.grid(True)
    plt.tight_layout()  # Ajusta per evitar que la llegenda es talli

    # Guarda el gràfic
    plot_filename = "diversity_accuracy_correlation.png"
    plt.savefig(plot_filename)


def calculate_model_explanation_analysis(all_results):
    """
    Calcula:
    1. Un score de diversitat individual per a cada model (com de diferent és dels altres).
    2. Una matriu de dissimilaritat (distància mitjana) entre tots els parells de models.
    Basat en les seves explicacions.
    """
    print("\nCalculating model explanation analysis (individual diversity & pairwise dissimilarity)...")

    num_models = len(all_results)
    if num_models < 2:
        print("Need at least two models for this analysis.")
        # Retorna diccionaris buits o None per indicar que no es pot calcular
        return {}, None

    model_names = [res["model_name"] for res in all_results]
    num_questions = all_results[0]["total_questions"]

    # Per a la diversitat individual (distància mitjana d'un model als altres)
    individual_model_diversity_contributions = {name: [] for name in model_names}

    # Per a la matriu de dissimilaritat entre parells (distància mitjana entre model A i model B)
    # Inicialitzem una llista de distàncies per a cada parell.
    # (model_idx1, model_idx2) -> [dist_q1, dist_q2, ...]
    pairwise_distances_lists = collections.defaultdict(list)

    for q_idx in tqdm(range(num_questions), desc="Processing questions for explanation analysis"):
        # Recollir explicacions i els índexs dels models que les van proporcionar
        # (exactament com a la teva funció original)
        question_explanations_texts = [None] * num_models
        valid_model_indices_for_q = []

        for model_idx, result in enumerate(all_results):
            current_q_response_data = next((r for r in result["responses_data"] if r["question_idx"] == q_idx), None)
            if current_q_response_data and current_q_response_data["model_explanation"]:
                question_explanations_texts[model_idx] = current_q_response_data["model_explanation"]
                valid_model_indices_for_q.append(model_idx)

        if len(valid_model_indices_for_q) < 2:
            continue

        # Generar embeddings només per a les explicacions vàlides d'aquesta pregunta
        # i mantenir la correspondència amb els índexs originals dels models
        embeddings_for_q = {}  # model_idx -> embedding
        for model_idx in valid_model_indices_for_q:
            emb = get_embedding(question_explanations_texts[model_idx])
            if emb is not None:
                embeddings_for_q[model_idx] = emb

        if len(embeddings_for_q) < 2:  # Si després de filtrar embeddings None, en queden menys de 2
            continue

        # Convertim a una llista d'embeddings i una llista d'índexs de models per a sklearn
        current_q_model_indices = sorted(embeddings_for_q.keys())  # Ordena per consistència
        current_q_embeddings_matrix = np.array([embeddings_for_q[idx] for idx in current_q_model_indices])

        # Calcula la matriu de distàncies cosinus per a la pregunta actual
        # cosine_distances = 1 - cosine_similarity
        dist_matrix_q = 1 - cosine_similarity(current_q_embeddings_matrix)

        # Ara, actualitza les mètriques
        for i, model_idx_k_local in enumerate(
            range(len(current_q_model_indices))
        ):  # índex local dins de la dist_matrix_q
            actual_model_idx_k = current_q_model_indices[model_idx_k_local]  # índex global del model k
            model_k_name = model_names[actual_model_idx_k]

            # 1. Per a la diversitat individual del model k
            distances_k_vs_others_q = []
            for j, model_idx_l_local in enumerate(range(len(current_q_model_indices))):
                if i == j:
                    continue
                distances_k_vs_others_q.append(dist_matrix_q[i, j])

            if distances_k_vs_others_q:
                avg_dist_k_vs_others_q = np.mean(distances_k_vs_others_q)
                individual_model_diversity_contributions[model_k_name].append(avg_dist_k_vs_others_q)

            # 2. Per a la matriu de dissimilaritat entre parells
            for j, model_idx_l_local in enumerate(
                range(model_idx_k_local + 1, len(current_q_model_indices))
            ):  # Només i < j per evitar duplicats i auto-comparacions
                actual_model_idx_l = current_q_model_indices[model_idx_l_local]  # índex global del model l

                # Clau del parell ordenada per consistència (min_idx, max_idx)
                pair_key = tuple(sorted((actual_model_idx_k, actual_model_idx_l)))
                pairwise_distances_lists[pair_key].append(dist_matrix_q[model_idx_k_local, model_idx_l_local])

    # Calcula els scores finals
    # 1. Diversitat individual mitjana
    final_individual_diversities = {}
    for model_name, scores in individual_model_diversity_contributions.items():
        if scores:
            final_individual_diversities[model_name] = np.mean(scores)
        else:
            final_individual_diversities[model_name] = np.nan

    # 2. Matriu de dissimilaritat entre parells
    dissimilarity_matrix = pd.DataFrame(np.nan, index=model_names, columns=model_names)
    for i in range(num_models):
        dissimilarity_matrix.iloc[i, i] = 0.0  # Distància d'un model a si mateix és 0
        for j in range(i + 1, num_models):
            pair_key = tuple(sorted((i, j)))  # Índexs, no noms de model aquí
            model_i_name = model_names[i]
            model_j_name = model_names[j]

            if pairwise_distances_lists[pair_key]:
                avg_pairwise_dist = np.mean(pairwise_distances_lists[pair_key])
                dissimilarity_matrix.loc[model_i_name, model_j_name] = avg_pairwise_dist
                dissimilarity_matrix.loc[model_j_name, model_i_name] = avg_pairwise_dist  # Matriu simètrica
            else:
                # Si no hi ha dades per a un parell (rar si tots els models responen a algunes preguntes)
                dissimilarity_matrix.loc[model_i_name, model_j_name] = np.nan
                dissimilarity_matrix.loc[model_j_name, model_i_name] = np.nan

    return final_individual_diversities, dissimilarity_matrix


def plot_dissimilarity_matrix(matrix_df, filename="dissimilarity_matrix_heatmap.png"):
    """Plota la matriu de dissimilaritat com un heatmap."""
    if matrix_df is None or matrix_df.empty:
        print("Dissimilarity matrix is empty, skipping heatmap plot.")
        return

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix_df, annot=True, fmt=".3f", cmap="viridis_r", cbar_kws={"label": "Avg. Cosine Distance"})
    plt.title("Pairwise Model Dissimilarity (Based on Explanations)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename)


if __name__ == "__main__":
    MODELS = {
        "Medicina General": "tunedModels/medicinageneralcsv-q4i0ydc9l1uvxbzsxmii8",
        "Ciències Bàsiques": "tunedModels/ciencies-basiques-2-pfg4bpafqcay88df2kr8",
        "Patologia i Farmacologia": "tunedModels/patologia-farmacologia-2-8iy2ixmy5bluqzw",
        "Cirurgia": "tunedModels/cirurgia-2-2c1cy8nkr5ca5mui15tu4wtlpapp8",
        "Pediatria i Ginecologia": "tunedModels/pediatria-ginecologia-2-ss7f3iy509x7x43h",
    }

    # num_questions = 1273  # MedQA-USMLE-4-options test complet
    num_questions = 5
    all_benchmark_results = []

    for model_name, model_id in MODELS.items():
        try:
            result = benchmark_model(model_id, model_name, num_questions, temperature=0.4, k_shot=5)
            all_benchmark_results.append(result)
        except HttpError as e:
            # Check if the error is related to resource exhaustion or rate limits
            if e.resp.status in [429, 403]:  # 429: Too Many Requests, 403: Quota exceeded
                print(f"API rate limit reached for {model_name}: {e}. Sleeping before retry.")
                time.sleep(5)
                try:
                    print(f"Retrying for {model_name}...")
                    result = benchmark_model(model_id, model_name, num_questions, temperature=0.4, k_shot=5)
                    all_benchmark_results.append(result)
                except Exception as e2:
                    print(f"Retry failed for {model_name}: {e2}")
        except Exception as e:
            print(f"Error benchmarking {model_name}: {e}")

    individual_diversities, dissimilarity_matrix_df = calculate_model_explanation_analysis(all_benchmark_results)

    print("\nModel Individual Diversity Scores (Avg. Cosine Distance to others' explanations):")
    if individual_diversities:
        for name, score in individual_diversities.items():
            print(f"- {name}: {score:.4f}")
    else:
        print("Could not calculate individual diversity scores.")

    print("\nPairwise Model Dissimilarity Matrix (Avg. Cosine Distance):")
    if dissimilarity_matrix_df is not None and not dissimilarity_matrix_df.empty:
        print(dissimilarity_matrix_df.to_string(float_format="%.3f"))
        plot_dissimilarity_matrix(dissimilarity_matrix_df)
    else:
        print("Could not calculate dissimilarity matrix.")

    # Correlació entre diversitat individual i accuracy
    model_accuracies = {res["model_name"]: res["accuracy"] for res in all_benchmark_results}
    if individual_diversities:  # Assegura't que tenim els scores individuals
        plot_diversity_accuracy_correlation(model_accuracies, individual_diversities, all_benchmark_results)

    # Calculem la diversitat de cada model
    model_diversities = calculate_model_diversity_scores(all_benchmark_results)
    print("\nModel Diversity Scores (Avg. Cosine Distance to others' explanations):")
    for name, score in model_diversities.items():
        print(f"- {name}: {score:.4f}")

    print("\n\n===== SUMMARY OF BENCHMARKS =====")
    print(f"{'Model':<30} | {'Accuracy':<10} | {'Correct':<15} | {'No Answer':<10}")
    print("-" * 75)
    # Ordena per accuracy per a la impressió
    sorted_results_for_print = sorted(all_benchmark_results, key=lambda x: x["accuracy"], reverse=True)
    for r in sorted_results_for_print:
        correct_str = f"{r['correct_count']}/{r['total_questions'] - r['no_answer_count']}"
        print(f"{r['model_name']:<30} | {r['accuracy']:.1f}% | " f"{correct_str:<15} | {r['no_answer_count']}")
