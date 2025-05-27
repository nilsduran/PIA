# Calculate elo rating from match results
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def elo_rating(player1_rating, player2_rating, score1, score2, k=32):
    """
    Calculate the new Elo ratings for two players after a match.
    """
    expected_score1 = 1 / (1 + 10 ** ((player2_rating - player1_rating) / 400))
    expected_score2 = 1 / (1 + 10 ** ((player1_rating - player2_rating) / 400))
    new_player1_rating = player1_rating + k * (score1 - expected_score1)
    new_player2_rating = player2_rating + k * (score2 - expected_score2)
    return new_player1_rating, new_player2_rating


def calculate_elo_ratings(matches, initial_rating=1000, k=32):
    """
    Calculate Elo ratings for players based on match results.
    """
    ratings = {}
    for _, match in matches.iterrows():
        p1 = match["model_A_config"]
        p2 = match["model_B_config"]
        s1 = match["score_A_vs_B"]
        s2 = 1 - s1
        ratings.setdefault(p1, initial_rating)
        ratings.setdefault(p2, initial_rating)
        r1, r2 = elo_rating(ratings[p1], ratings[p2], s1, s2, k)
        ratings[p1], ratings[p2] = r1, r2
    return pd.DataFrame(list(ratings.items()), columns=["player", "rating"])


def elo_with_confidence_intervals(matches, initial_rating=1000, k=32, n_bootstrap=1000, alpha=0.05):
    """
    Calculate Elo ratings plus 95% confidence intervals via bootstrap.
    """
    # base ratings (to list all players)
    base_df = calculate_elo_ratings(matches, initial_rating, k)
    players = base_df["player"].tolist()

    # collect bootstrap samples
    boot_ratings = {p: [] for p in players}
    for _ in range(n_bootstrap):
        sample = matches.sample(frac=1, replace=True)
        df_bs = calculate_elo_ratings(sample, initial_rating, k)
        bs_dict = dict(zip(df_bs["player"], df_bs["rating"]))
        for p in players:
            boot_ratings[p].append(bs_dict.get(p, initial_rating))

    # compute CIs
    lower_pct = 100 * (alpha / 2)
    upper_pct = 100 * (1 - alpha / 2)
    rows = []
    for p in players:
        arr = np.array(boot_ratings[p])
        lo, hi = np.percentile(arr, [lower_pct, upper_pct])
        mean = round(base_df.loc[base_df["player"] == p, "rating"].iloc[0])
        lower_diff = round(mean - lo, 1)
        upper_diff = round(hi - mean, 1)
        rows.append((p, mean, upper_diff, lower_diff))

    return pd.DataFrame(rows, columns=["player", "rating", "ci_upper", "ci_lower"])


def win_probability(player1_rating, player2_rating):
    """
    Calculate the expected score of player 1 against player 2 based on their Elo ratings.
    """
    return 1 / (1 + 10 ** ((player2_rating - player1_rating) / 400))


def expected_score_matrix(ratings):
    """
    Calculate the expected score matrix for all players based on their Elo ratings.
    """
    n = len(ratings)
    expected_scores = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                expected_scores[i, j] = win_probability(ratings[i], ratings[j])
    return expected_scores


def update_elo_ratings():
    """
    Update Elo ratings from the battle votes CSV file.
    """
    matches_df = pd.read_csv("battle_votes.csv")
    ratings_ci_df = elo_with_confidence_intervals(matches_df, initial_rating=1000, k=32, n_bootstrap=1000, alpha=0.05)
    ratings_ci_df = ratings_ci_df.sort_values("rating", ascending=False)
    ratings_ci_df.to_csv("elo_ratings_with_ci.csv", index=False)


# Emparellaments òptims per a convergir l'elo més ràpidament
def optimal_pairings(matches_df=None):
    """
    Generate optimal pairings for matches to converge Elo ratings quickly.
    Also, give expected information gain per match.
    This function assumes matches_df has columns 'model_A_config', 'model_B_config', and 'score_A_vs_B'.
    """
    if not isinstance(matches_df, pd.DataFrame):
        matches_df = pd.read_csv("battle_votes.csv")
    pairings = []
    players = matches_df["model_A_config"].unique().tolist()
    ratings = calculate_elo_ratings(matches_df).set_index("player")["rating"].to_dict()
    ratings = {k: v for k, v in ratings.items() if k in players}
    for i in range(len(players)):
        for j in range(i + 1, len(players)):
            p1 = players[i]
            p2 = players[j]
            if p1 in ratings and p2 in ratings:
                expected_score_p1 = win_probability(ratings[p1], ratings[p2])
                expected_score_p2 = win_probability(ratings[p2], ratings[p1])
                expected_score_difference = abs(expected_score_p1 - expected_score_p2)
                # Calculate information gain from expected_score_difference
                information_gain = np.log2(1 / expected_score_difference)

                pairings.append((p1, p2, information_gain))
    # Sort pairings by expected score difference (descending)
    pairings.sort(key=lambda x: x[2], reverse=True)
    top_pairing = pairings.pop(0)

    return top_pairing[0], top_pairing[1]


if __name__ == "__main__":
    op = optimal_pairings()  # Get the top pairing
    print("Optimal pairing:", op[0], "vs", op[1])
    matches_df = pd.read_csv("battle_votes.csv")
    emparellaments_optims = optimal_pairings(matches_df)
    print("Optimal pairings for matches:")
    for p1, p2, v, v2 in emparellaments_optims[:10]:  # Show only the top 10 pairings
        print(f"{p1} vs {p2}: {v:.3f} |---| {v2:.3f}")
    ratings_df = calculate_elo_ratings(matches_df, initial_rating=1000, k=32)
    models_order = [
        "Medicina General",
        "Ciències Bàsiques",
        "Patologia i Farmacologia",
        "Cirurgia",
        "Pediatria i Ginecologia",
        "experts_1_diversitat_Baixa",
        "experts_2_diversitat_Baixa",
        "experts_3_diversitat_Baixa",
        "experts_4_diversitat_Baixa",
        "experts_5_diversitat_Baixa",
        "experts_1_diversitat_Mitjana",
        "experts_2_diversitat_Mitjana",
        "experts_3_diversitat_Mitjana",
        "experts_4_diversitat_Mitjana",
        "experts_5_diversitat_Mitjana",
        "experts_1_diversitat_Alta",
        "experts_2_diversitat_Alta",
        "experts_3_diversitat_Alta",
        "experts_4_diversitat_Alta",
        "experts_5_diversitat_Alta",
    ]

    # Reindex ratings_df according to models_order, filling missing values with initial rating
    ratings_df = ratings_df.set_index("player").reindex(models_order).fillna(1000).reset_index()

    expected_score_matrix = expected_score_matrix(ratings_df["rating"].values)
    print("Elo ratings:")
    for _, row in ratings_df.iterrows():
        print(f"{row['player']}: {row['rating']:.1f}")
    plt.figure(figsize=(10, 10))
    # Create a mask to set diagonal values to NaN
    mask = np.zeros_like(expected_score_matrix)
    np.fill_diagonal(mask, True)

    # heatmap with colors red-green but if 0 then white
    sns.heatmap(
        expected_score_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0.5,
        vmin=0,
        vmax=1,
        xticklabels=ratings_df["player"],
        yticklabels=ratings_df["player"],
        mask=mask,  # Apply the mask
        cbar_kws={"label": "Expected Score"},  # Add a label to the colorbar
    )
    plt.title("Expected Score Matrix")
    plt.xlabel("Player")
    plt.ylabel("Player")
    plt.tight_layout()
    plt.savefig("expected_scores_matrix.png")
    plt.show()
