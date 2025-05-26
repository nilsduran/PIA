# Calculate elo rating from match results
import numpy as np
import pandas as pd


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


def elo_from_csv(file_path, initial_rating=1000, k=32):
    """
    Calculate Elo ratings from a CSV file containing match results.
    """
    matches = pd.read_csv(file_path)
    return calculate_elo_ratings(matches, initial_rating, k)


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


if __name__ == "__main__":
    matches_df = pd.read_csv("battle_votes.csv")
    ratings_ci_df = elo_with_confidence_intervals(matches_df, initial_rating=1000, k=32, n_bootstrap=1000, alpha=0.05)
    ratings_ci_df = ratings_ci_df.sort_values("rating", ascending=False)
    print("Elo ratings with 95% CIs:")
    for _, row in ratings_ci_df.iterrows():
        print(f"{row['player']}: {row['rating']:.1f} " f"(+{row['ci_upper']:.1f}/-{row['ci_lower']:.1f})")

    ratings_ci_df.to_csv("elo_ratings_with_ci.csv", index=False)
