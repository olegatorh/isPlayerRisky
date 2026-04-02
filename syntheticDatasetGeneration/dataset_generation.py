import datetime

import numpy as np
import pandas as pd


def generate_gambling_rg_dataset(n_rows=30000, seed=42):
    rng = np.random.default_rng(seed)
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def normalize(series):
        std = series.std()
        if std == 0:
            return pd.Series(np.zeros(len(series)), index=series.index)
        return (series - series.mean()) / std

    segments = np.array(["casual", "regular", "engaged", "risk_prone"])
    segment = rng.choice(segments, size=n_rows, p=[0.40, 0.35, 0.18, 0.07])

    country = rng.choice(
        ["PL", "UA", "DE", "ES", "BR", "IN"],
        size=n_rows,
        p=[0.34, 0.24, 0.18, 0.12, 0.07, 0.05]
    )

    device_type = rng.choice(
        ["android", "ios", "desktop"],
        size=n_rows,
        p=[0.45, 0.20, 0.35]
    )

    verification_status = rng.choice(
        ["verified", "pending", "limited"],
        size=n_rows,
        p=[0.82, 0.12, 0.06]
    )

    days_since_registration = []
    for seg in segment:
        if seg == "casual":
            days_since_registration.append(rng.integers(20, 1200))
        elif seg == "regular":
            days_since_registration.append(rng.integers(30, 1500))
        elif seg == "engaged":
            days_since_registration.append(rng.integers(10, 1200))
        else:
            days_since_registration.append(rng.integers(5, 500))
    days_since_registration = np.array(days_since_registration)

    deposit_lambda_map = {
        "casual": 2.0,
        "regular": 4.0,
        "engaged": 8.0,
        "risk_prone": 11.0,
    }
    deposit_count_30d = np.array([rng.poisson(deposit_lambda_map[s]) for s in segment])

    deposit_mean_map = {
        "casual": 3.3,
        "regular": 4.0,
        "engaged": 4.8,
        "risk_prone": 4.9,
    }
    deposit_amount_30d = np.array([
        rng.lognormal(mean=deposit_mean_map[s], sigma=0.55) * max(c, 1)
        for s, c in zip(segment, deposit_count_30d)
    ])
    deposit_amount_30d = np.round(deposit_amount_30d, 2)

    session_boost_map = {
        "casual": 2,
        "regular": 5,
        "engaged": 10,
        "risk_prone": 13,
    }
    session_count_30d = np.array([
        max(0, int(rng.normal(loc=c + session_boost_map[s], scale=2.5)))
        for s, c in zip(segment, deposit_count_30d)
    ])

    bet_multiplier_map = {
        "casual": 3.5,
        "regular": 6.0,
        "engaged": 10.0,
        "risk_prone": 12.0,
    }
    bet_count_30d = np.array([
        max(0, int(rng.normal(loc=sc * bet_multiplier_map[s], scale=8)))
        for s, sc in zip(segment, session_count_30d)
    ])

    avg_bet_base_map = {
        "casual": 0.03,
        "regular": 0.035,
        "engaged": 0.045,
        "risk_prone": 0.05,
    }
    avg_bet_amount = np.array([
        max(0.5, rng.normal(loc=max(dep, 10) * avg_bet_base_map[s] / max(dc, 1), scale=2.0))
        for s, dep, dc in zip(segment, deposit_amount_30d, deposit_count_30d)
    ])
    avg_bet_amount = np.round(avg_bet_amount, 2)

    duration_mean_map = {
        "casual": 18,
        "regular": 32,
        "engaged": 52,
        "risk_prone": 68,
    }
    avg_session_minutes = np.array([
        max(3, rng.normal(loc=duration_mean_map[s], scale=8))
        for s in segment
    ])
    avg_session_minutes = np.round(avg_session_minutes, 1)

    night_alpha_beta_map = {
        "casual": (1.2, 8.0),
        "regular": (1.8, 5.5),
        "engaged": (2.4, 4.2),
        "risk_prone": (4.8, 2.8),
    }
    night_session_ratio = np.array([
        rng.beta(*night_alpha_beta_map[s]) for s in segment
    ])
    night_session_ratio = np.round(night_session_ratio, 3)

    withdrawal_count_30d = np.array([
        max(0, int(rng.normal(loc=dc * 0.35, scale=1.2)))
        for dc in deposit_count_30d
    ])

    failed_dep_lambda_map = {
        "casual": 0.1,
        "regular": 0.3,
        "engaged": 0.6,
        "risk_prone": 2.4,
    }
    failed_deposit_count_30d = np.array([
        rng.poisson(failed_dep_lambda_map[s]) for s in segment
    ])

    margin_map = {
        "casual": 0.05,
        "regular": 0.07,
        "engaged": 0.10,
        "risk_prone": 0.16,
    }
    loss_amount_30d = np.array([
        max(0, rng.normal(loc=dep * margin_map[s], scale=max(dep * 0.06, 5)))
        for s, dep in zip(segment, deposit_amount_30d)
    ])
    loss_amount_30d = np.round(loss_amount_30d, 2)

    loss_streak_max = np.array([
        max(0, int(rng.normal(
            loc={"casual": 2, "regular": 4, "engaged": 7, "risk_prone": 12}[s],
            scale=2.0
        )))
        for s in segment
    ])

    chasing_score = (
        0.015 * bet_count_30d
        + 0.004 * loss_amount_30d
        + 1.8 * night_session_ratio
        + 0.35 * failed_deposit_count_30d
        + rng.normal(0, 1.2, n_rows)
    )
    chasing_score = np.clip(chasing_score, 0, None)
    chasing_score = np.round(chasing_score, 2)

    bonus_claim_lambda_map = {
        "casual": 0.3,
        "regular": 0.8,
        "engaged": 1.4,
        "risk_prone": 2.0,
    }
    bonus_claim_count_30d = np.array([
        rng.poisson(bonus_claim_lambda_map[s]) for s in segment
    ])

    bonus_to_deposit_ratio = np.array([
        min(1.0, max(0.0, rng.normal(
            loc={"casual": 0.08, "regular": 0.12, "engaged": 0.15, "risk_prone": 0.22}[s],
            scale=0.06
        )))
        for s in segment
    ])
    bonus_to_deposit_ratio = np.round(bonus_to_deposit_ratio, 3)

    df = pd.DataFrame({
        "country": country,
        "device_type": device_type,
        "verification_status": verification_status,
        "player_segment": segment,
        "days_since_registration": days_since_registration.astype(int),
        "deposit_count_30d": deposit_count_30d.astype(int),
        "deposit_amount_30d": deposit_amount_30d,
        "withdrawal_count_30d": withdrawal_count_30d.astype(int),
        "failed_deposit_count_30d": failed_deposit_count_30d.astype(int),
        "session_count_30d": session_count_30d.astype(int),
        "bet_count_30d": bet_count_30d.astype(int),
        "avg_bet_amount": avg_bet_amount,
        "avg_session_minutes": avg_session_minutes,
        "night_session_ratio": night_session_ratio,
        "loss_amount_30d": loss_amount_30d,
        "loss_streak_max": loss_streak_max.astype(int),
        "chasing_score": chasing_score,
        "bonus_claim_count_30d": bonus_claim_count_30d.astype(int),
        "bonus_to_deposit_ratio": bonus_to_deposit_ratio,
    })

    risk_score = (
        0.85 * normalize(df["night_session_ratio"])
        + 0.75 * normalize(df["loss_amount_30d"])
        + 0.90 * normalize(df["failed_deposit_count_30d"])
        + 0.80 * normalize(df["chasing_score"])
        + 0.55 * normalize(df["avg_session_minutes"])
        + 0.35 * normalize(df["deposit_count_30d"])
        + 0.20 * normalize(df["bonus_to_deposit_ratio"])
        - 0.30 * normalize(df["days_since_registration"])
        + rng.normal(0, 0.7, n_rows)
    )

    risk_probability = sigmoid(risk_score - 1.45)
    df["is_risky_player"] = rng.binomial(1, risk_probability, n_rows).astype(int)

    for col, frac in {
        "avg_session_minutes": 0.04,
        "bonus_to_deposit_ratio": 0.03,
        "verification_status": 0.02,
        "country": 0.01,
    }.items():
        idx = rng.choice(df.index, size=int(len(df) * frac), replace=False)
        df.loc[idx, col] = np.nan

    return df


if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y:%m:%d|%H:%M:%S")
    df = generate_gambling_rg_dataset(n_rows=30000, seed=42)
    df.to_csv(f"syntheticDatasetGeneration/Datasets/gambling_rg_dataset({timestamp}).csv", index=False)

    print(df.head())
    print("\nShape:", df.shape)
    print("\nTarget rate:")
    print(df["is_risky_player"].mean())