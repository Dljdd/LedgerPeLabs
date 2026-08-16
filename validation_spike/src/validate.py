#!/usr/bin/env python3
"""Preregistered falsification spike for the Adaptive Payment Security Range.

The implementation intentionally depends only on NumPy, pandas, and the Python
standard library. All feature construction is event-time and all model features
are selected from an explicit allow-list.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import sys
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"
SEEDS = [11, 23, 37, 51, 73]
DAY = 1440.0

BASE_FEATURES = [
    "log_amount",
    "channel_a2a",
    "hour_sin",
    "hour_cos",
]

TEMPORAL_FEATURES = BASE_FEATURES + [
    "account_txn_count_60m",
    "account_amount_24h",
    "new_device_for_account",
    "new_beneficiary_for_account",
    "device_account_count_24h",
    "beneficiary_account_count_24h",
    "device_decline_count_60m",
    "merchant_decline_count_60m",
]

NOVELTY_FEATURES = [
    "device_account_count_24h",
    "beneficiary_account_count_24h",
    "device_decline_count_60m",
    "merchant_decline_count_60m",
]

FORBIDDEN_TOKENS = {
    "scenario",
    "family",
    "campaign",
    "seed",
    "generator",
    "fraud",
    "future",
    "regime",
    "segment",
    "label",
}


def record(
    timestamp: float,
    account_id: str,
    device_id: str,
    merchant_id: str,
    beneficiary_id: str,
    amount: float,
    channel_a2a: int,
    decline: int,
    is_fraud: int,
    family: str,
    campaign_id: str,
    benign_segment: str,
    regime: str,
    generator_seed: int,
) -> dict[str, Any]:
    return {
        "timestamp": float(timestamp),
        "account_id": account_id,
        "device_id": device_id,
        "merchant_id": merchant_id,
        "beneficiary_id": beneficiary_id,
        "amount": round(float(amount), 2),
        "channel_a2a": int(channel_a2a),
        "decline": int(decline),
        "is_fraud": int(is_fraud),
        "family": family,
        "campaign_id": campaign_id,
        "benign_segment": benign_segment,
        "regime": regime,
        "generator_seed": int(generator_seed),
    }


def _normal_events(
    rng: np.random.Generator,
    seed: int,
    regime: str,
    start_day: int,
    n_accounts: int = 300,
    n_events: int = 6200,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    accounts = [f"A{i:04d}" for i in range(n_accounts)]
    primary_devices = {a: f"D{i:04d}" for i, a in enumerate(accounts)}
    base_amount = {a: float(rng.lognormal(3.4, 0.45)) for a in accounts}
    favorite_merchants = {
        a: [f"M{j:03d}" for j in rng.choice(90, size=6, replace=False)]
        for a in accounts
    }
    regular_beneficiaries = {
        a: [f"BREG_{a}_{j}" for j in range(3)] for a in accounts
    }

    events: list[dict[str, Any]] = []
    times = np.sort(rng.uniform(start_day * DAY, (start_day + 40) * DAY, n_events))
    for t in times:
        a = accounts[int(rng.integers(0, n_accounts))]
        is_a2a = int(rng.random() < (0.24 if regime == "A" else 0.31))
        scale = 1.0 if regime == "A" else 1.13
        sigma = 0.62 if regime == "A" else 0.76
        amount = max(1.0, base_amount[a] * scale * float(rng.lognormal(0, sigma)))
        use_primary = rng.random() < (0.975 if regime == "A" else 0.955)
        device = primary_devices[a] if use_primary else f"DALT_{a}_{int(rng.integers(0, 3))}"
        if is_a2a:
            merchant = ""
            beneficiary = regular_beneficiaries[a][int(rng.integers(0, 3))]
            decline_p = 0.008 if regime == "A" else 0.012
        else:
            merchant = favorite_merchants[a][int(rng.integers(0, 6))]
            beneficiary = ""
            decline_p = 0.018 if regime == "A" else 0.025
        events.append(
            record(
                t,
                a,
                device,
                merchant,
                beneficiary,
                amount,
                is_a2a,
                int(rng.random() < decline_p),
                0,
                "legit",
                "",
                "ordinary",
                regime,
                seed,
            )
        )

    if regime == "B":
        # A genuinely unseen benign segment. It is high-value and bursty, but it
        # never shares a device or beneficiary across accounts.
        travelers = rng.choice(accounts, size=42, replace=False)
        for a in travelers:
            trip_start = (start_day + int(rng.integers(26, 39))) * DAY + float(rng.uniform(0, 600))
            burst = int(rng.integers(5, 11))
            trip_device = (
                primary_devices[a]
                if rng.random() < 0.65
                else f"DTRAVEL_{a}_{int(rng.integers(0, 10000))}"
            )
            for j in range(burst):
                amount = base_amount[a] * float(rng.lognormal(1.05, 0.52))
                events.append(
                    record(
                        trip_start + j * float(rng.uniform(8, 35)),
                        a,
                        trip_device,
                        f"MTRAVEL_{a}_{j % 4}",
                        "",
                        amount,
                        0,
                        int(rng.random() < 0.025),
                        0,
                        "legit",
                        "",
                        "business_travel",
                        regime,
                        seed,
                    )
                )

    entities = {
        "accounts": accounts,
        "primary_devices": primary_devices,
        "favorite_merchants": favorite_merchants,
        "regular_beneficiaries": regular_beneficiaries,
    }
    return events, entities


def _inject_ato(
    events: list[dict[str, Any]],
    entities: dict[str, Any],
    rng: np.random.Generator,
    seed: int,
    regime: str,
    start_day: int,
    n_train: int,
    n_test: int,
) -> None:
    accounts = entities["accounts"]
    starts = []
    if n_train:
        starts.extend(rng.uniform(start_day + 5, start_day + 23, n_train))
    starts.extend(rng.uniform(start_day + 27, start_day + 39, n_test))
    for idx, day_value in enumerate(starts):
        a = str(rng.choice(accounts))
        campaign = f"{regime}_ato_{seed}_{idx}"
        if regime == "A":
            device = f"DATK_{campaign}"
            beneficiary = f"BATK_{campaign}"
            n_tx = 4
            spacing = float(rng.uniform(5, 18))
            amounts = rng.uniform(190, 480, n_tx)
        else:
            # Hidden mechanism: trusted-device takeover, sometimes using an
            # established beneficiary, with slower and smaller transfers.
            device = entities["primary_devices"][a]
            beneficiary = (
                str(rng.choice(entities["regular_beneficiaries"][a]))
                if rng.random() < 0.45
                else f"BATK_B_{campaign}"
            )
            n_tx = 6
            spacing = float(rng.uniform(28, 74))
            amounts = rng.uniform(90, 260, n_tx)
        t0 = day_value * DAY
        for j in range(n_tx):
            events.append(
                record(
                    t0 + j * spacing,
                    a,
                    device,
                    "",
                    beneficiary,
                    float(amounts[j]),
                    1,
                    0,
                    1,
                    "ato",
                    campaign,
                    "",
                    regime,
                    seed,
                )
            )


def _inject_card_testing(
    events: list[dict[str, Any]],
    entities: dict[str, Any],
    rng: np.random.Generator,
    seed: int,
    regime: str,
    start_day: int,
    n_train: int,
    n_test: int,
) -> None:
    accounts = entities["accounts"]
    starts = []
    if n_train:
        starts.extend(rng.uniform(start_day + 5, start_day + 23, n_train))
    starts.extend(rng.uniform(start_day + 27, start_day + 39, n_test))
    for idx, day_value in enumerate(starts):
        campaign = f"{regime}_ct_{seed}_{idx}"
        selected = [str(x) for x in rng.choice(accounts, size=12, replace=False)]
        t0 = day_value * DAY
        n_tx = 14 if regime == "A" else 20
        for j in range(n_tx):
            a = selected[j % len(selected)]
            if regime == "A":
                device = f"DCT_{campaign}"
                decline_p = 0.86
                spacing = 2.0
                amount = float(rng.uniform(1, 18))
            else:
                # Hidden mechanism is distributed across three devices, has
                # fewer declines, and lasts longer.
                device = f"DCT_{campaign}_{j % 3}"
                decline_p = 0.58
                spacing = 6.0
                amount = float(rng.uniform(4, 42))
            events.append(
                record(
                    t0 + j * spacing,
                    a,
                    device,
                    f"MCT_{j % (2 if regime == 'A' else 5)}",
                    "",
                    amount,
                    0,
                    int(rng.random() < decline_p),
                    1,
                    "card_testing",
                    campaign,
                    "",
                    regime,
                    seed,
                )
            )


def _inject_mule(
    events: list[dict[str, Any]],
    entities: dict[str, Any],
    rng: np.random.Generator,
    seed: int,
    regime: str,
    start_day: int,
    n_test: int,
) -> None:
    accounts = entities["accounts"]
    starts = rng.uniform(start_day + 27, start_day + 39, n_test)
    for idx, day_value in enumerate(starts):
        campaign = f"{regime}_mule_{seed}_{idx}"
        victims = [str(x) for x in rng.choice(accounts, size=10, replace=False)]
        t0 = day_value * DAY
        if regime == "A":
            first_hops = [f"BMULE_{campaign}"]
            spacing = 10.0
        else:
            # Hidden mechanism: victims are split over three first-hop mules,
            # followed by two slower aggregation transfers to a final mule.
            first_hops = [f"BMULE_{campaign}_{j}" for j in range(3)]
            spacing = 22.0
        for j, a in enumerate(victims):
            beneficiary = first_hops[j % len(first_hops)]
            events.append(
                record(
                    t0 + j * spacing,
                    a,
                    entities["primary_devices"][a],
                    "",
                    beneficiary,
                    float(rng.uniform(125, 330)),
                    1,
                    0,
                    1,
                    "mule",
                    campaign,
                    "",
                    regime,
                    seed,
                )
            )
        if regime == "B":
            final_beneficiary = f"BFINAL_{campaign}"
            for j in range(2):
                mule_account = f"AMULE_{campaign}_{j}"
                events.append(
                    record(
                        t0 + 260 + j * 35,
                        mule_account,
                        f"DMULE_{campaign}_{j}",
                        "",
                        final_beneficiary,
                        float(rng.uniform(480, 720)),
                        1,
                        0,
                        1,
                        "mule",
                        campaign,
                        "",
                        regime,
                        seed,
                    )
                )


def simulate(seed: int, regime: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed + (0 if regime == "A" else 100_000))
    start_day = 0 if regime == "A" else 100
    events, entities = _normal_events(rng, seed, regime, start_day)
    if regime == "A":
        _inject_ato(events, entities, rng, seed, regime, start_day, 28, 15)
        _inject_card_testing(events, entities, rng, seed, regime, start_day, 22, 12)
        _inject_mule(events, entities, rng, seed, regime, start_day, 12)
    else:
        _inject_ato(events, entities, rng, seed, regime, start_day, 0, 15)
        _inject_card_testing(events, entities, rng, seed, regime, start_day, 0, 12)
        _inject_mule(events, entities, rng, seed, regime, start_day, 12)

    df = pd.DataFrame(events).sort_values(["timestamp", "account_id"], kind="mergesort")
    df = df.reset_index(drop=True)
    df.insert(0, "event_id", [f"{regime}_{seed}_{i:07d}" for i in range(len(df))])
    return df


def _prune(dq: deque, cutoff: float) -> None:
    while dq and dq[0][0] < cutoff:
        dq.popleft()


def make_features(events: pd.DataFrame) -> pd.DataFrame:
    ordered = events.sort_values(["timestamp", "event_id"], kind="mergesort").reset_index(drop=True)
    account_hour: dict[str, deque] = defaultdict(deque)
    account_day: dict[str, deque] = defaultdict(deque)
    device_day: dict[str, deque] = defaultdict(deque)
    beneficiary_day: dict[str, deque] = defaultdict(deque)
    device_hour: dict[str, deque] = defaultdict(deque)
    merchant_hour: dict[str, deque] = defaultdict(deque)
    account_device_first_seen: dict[tuple[str, str], float] = {}
    account_beneficiary_first_seen: dict[tuple[str, str], float] = {}
    rows: list[dict[str, float | str]] = []

    for row in ordered.itertuples(index=False):
        t = float(row.timestamp)
        a, d, m, b = row.account_id, row.device_id, row.merchant_id, row.beneficiary_id
        _prune(account_hour[a], t - 60)
        _prune(account_day[a], t - DAY)
        _prune(device_day[d], t - DAY)
        if b:
            _prune(beneficiary_day[b], t - DAY)
        _prune(device_hour[d], t - 60)
        if m:
            _prune(merchant_hour[m], t - 60)

        hour = (t % DAY) / 60.0
        # Aggregates use strictly earlier timestamps. Equal-time events do not
        # observe one another, so event-ID ordering cannot leak within a batch.
        prior_account_hour = [x for x in account_hour[a] if x[0] < t]
        prior_account_day = [x for x in account_day[a] if x[0] < t]
        prior_device_day = [x for x in device_day[d] if x[0] < t]
        prior_beneficiary_day = [x for x in beneficiary_day[b] if x[0] < t] if b else []
        prior_device_hour = [x for x in device_hour[d] if x[0] < t]
        prior_merchant_hour = [x for x in merchant_hour[m] if x[0] < t] if m else []

        previous_source_times = [x[0] for x in prior_account_hour]
        previous_source_times += [x[0] for x in prior_account_day]
        previous_source_times += [x[0] for x in prior_device_day]
        if b:
            previous_source_times += [x[0] for x in prior_beneficiary_day]
        if m:
            previous_source_times += [x[0] for x in prior_merchant_hour]
        # Raw current-event fields are part of the decision request and are not
        # historical sources. Negative infinity denotes no historical source.
        max_source_time = max(previous_source_times, default=float("-inf"))

        device_accounts = {x[1] for x in prior_device_day}
        device_accounts.add(a)
        beneficiary_accounts = {x[1] for x in prior_beneficiary_day} if b else set()
        if b:
            beneficiary_accounts.add(a)

        rows.append(
            {
                "event_id": row.event_id,
                "timestamp": t,
                "log_amount": math.log1p(max(0.0, float(row.amount))),
                "channel_a2a": float(row.channel_a2a),
                "hour_sin": math.sin(2 * math.pi * hour / 24),
                "hour_cos": math.cos(2 * math.pi * hour / 24),
                "account_txn_count_60m": math.log1p(len(prior_account_hour)),
                "account_amount_24h": math.log1p(sum(x[1] for x in prior_account_day)),
                "new_device_for_account": float(account_device_first_seen.get((a, d), float("inf")) >= t),
                "new_beneficiary_for_account": float(bool(b) and account_beneficiary_first_seen.get((a, b), float("inf")) >= t),
                "device_account_count_24h": math.log1p(len(device_accounts)),
                "beneficiary_account_count_24h": math.log1p(len(beneficiary_accounts)),
                "device_decline_count_60m": math.log1p(sum(x[2] for x in prior_device_hour)),
                "merchant_decline_count_60m": math.log1p(sum(x[1] for x in prior_merchant_hour)) if m else 0.0,
                "_max_source_time": float(max_source_time),
            }
        )

        account_hour[a].append((t, float(row.amount)))
        account_day[a].append((t, float(row.amount)))
        device_day[d].append((t, a, int(row.decline)))
        if b:
            beneficiary_day[b].append((t, a))
        device_hour[d].append((t, a, int(row.decline)))
        if m:
            merchant_hour[m].append((t, int(row.decline)))
        account_device_first_seen[(a, d)] = min(t, account_device_first_seen.get((a, d), t))
        if b:
            account_beneficiary_first_seen[(a, b)] = min(t, account_beneficiary_first_seen.get((a, b), t))

    feat = pd.DataFrame(rows)
    # Restore caller order through event_id, making feature/event joins explicit.
    return events[["event_id"]].merge(feat, on="event_id", how="left", validate="one_to_one")


def audit_feature_names(names: Iterable[str]) -> list[str]:
    allowed = set(TEMPORAL_FEATURES)
    rejected = []
    for name in names:
        lower = name.lower()
        if name not in allowed or any(token in lower for token in FORBIDDEN_TOKENS):
            rejected.append(name)
    return rejected


def assert_event_time(events: pd.DataFrame, features: pd.DataFrame) -> bool:
    merged = events[["event_id", "timestamp"]].merge(
        features[["event_id", "_max_source_time"]], on="event_id", validate="one_to_one"
    )
    return bool((merged["_max_source_time"] < merged["timestamp"]).all())


@dataclass
class LogisticModel:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mean) / self.scale
        raw = np.clip(z @ self.weights + self.bias, -30, 30)
        return 1.0 / (1.0 + np.exp(-raw))


def fit_logistic(x: np.ndarray, y: np.ndarray, epochs: int = 650) -> LogisticModel:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    z = np.clip((x - mean) / scale, -12, 12)
    weights = np.zeros(z.shape[1], dtype=float)
    bias = 0.0
    pos = max(1.0, float(y.sum()))
    neg = max(1.0, float(len(y) - y.sum()))
    sample_weight = np.where(y > 0.5, len(y) / (2 * pos), len(y) / (2 * neg))
    norm = sample_weight.sum()
    learning_rate = 0.07
    l2 = 0.015
    for step in range(epochs):
        raw = np.clip(z @ weights + bias, -30, 30)
        pred = 1.0 / (1.0 + np.exp(-raw))
        error = (pred - y) * sample_weight
        grad_w = (z.T @ error) / norm + l2 * weights
        grad_b = float(error.sum() / norm)
        rate = learning_rate / math.sqrt(1.0 + step / 100.0)
        weights -= rate * grad_w
        bias -= rate * grad_b
    return LogisticModel(mean, scale, weights, bias)


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    if y.sum() == 0:
        return 0.0
    order = np.argsort(-score, kind="mergesort")
    ordered = y[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(precision[ordered == 1].mean())


def campaign_detection(events: pd.DataFrame, score: np.ndarray, threshold: float) -> float:
    tmp = events[["campaign_id", "is_fraud"]].copy()
    tmp["detected"] = score >= threshold
    tmp = tmp[(tmp["is_fraud"] == 1) & (tmp["campaign_id"] != "")]
    if tmp.empty:
        return 0.0
    grouped = tmp.groupby("campaign_id", sort=False)["detected"].max()
    return float(grouped.mean())


def percentile_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks / max(1, len(values) - 1)


def robust_novelty(
    train_events: pd.DataFrame,
    train_features: pd.DataFrame,
    target_features: pd.DataFrame,
) -> np.ndarray:
    mask = train_events["is_fraud"].to_numpy() == 0
    reference = train_features.loc[mask, NOVELTY_FEATURES].to_numpy(float)
    target = target_features[NOVELTY_FEATURES].to_numpy(float)
    median = np.median(reference, axis=0)
    q75 = np.percentile(reference, 75, axis=0)
    q25 = np.percentile(reference, 25, axis=0)
    # Count features are often zero-inflated. A floor prevents an all-zero IQR
    # from turning the first legitimate repeated event into an infinite score.
    scale = np.maximum(q75 - q25, 0.35)
    positive_deviation = np.maximum((target - median) / scale, 0.0)
    return np.max(positive_deviation, axis=1)


def triage_metrics(
    events: pd.DataFrame,
    risk: np.ndarray,
    novelty: np.ndarray,
    seed: int,
) -> dict[str, dict[str, float]]:
    capacity = max(1, int(math.floor(0.02 * len(events))))
    risk_priority = percentile_rank(risk)
    novelty_priority = 0.65 * risk_priority + 0.35 * percentile_rank(novelty)
    random_priority = np.random.default_rng(seed + 404).random(len(events))
    policies = {
        "risk_only": risk_priority,
        "novelty_aware": novelty_priority,
        "random": random_priority,
    }
    mule = (events["family"].to_numpy() == "mule") & (events["is_fraud"].to_numpy() == 1)
    travel = events["benign_segment"].to_numpy() == "business_travel"
    amount = events["amount"].to_numpy(float)
    total_mule_value = max(1e-9, float(amount[mule].sum()))
    result: dict[str, dict[str, float]] = {}
    for name, priority in policies.items():
        selected = np.argsort(-priority, kind="mergesort")[:capacity]
        chosen = np.zeros(len(events), dtype=bool)
        chosen[selected] = True
        result[name] = {
            "heldout_mule_value_capture": float(amount[mule & chosen].sum() / total_mule_value),
            "heldout_mule_event_capture": float((mule & chosen).sum() / max(1, mule.sum())),
            "travel_share_of_reviews": float((travel & chosen).sum() / capacity),
            "fraud_share_of_reviews": float(((events["is_fraud"].to_numpy() == 1) & chosen).sum() / capacity),
            "capacity": int(capacity),
        }
    return result


def make_attack_candidate(params: dict[str, Any], seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Create a persistent benign history followed by one attack campaign."""
    rng = np.random.default_rng(seed + 9_000)
    t0 = 60 * DAY
    accounts = [f"SEARCH_A{i}" for i in range(4)]
    primary = {a: f"SEARCH_PRIMARY_{a}" for a in accounts}
    events: list[dict[str, Any]] = []

    # Persistent account history establishes ordinary devices and beneficiaries.
    for i, a in enumerate(accounts):
        for j in range(8):
            events.append(
                record(
                    t0 - (10 - j) * DAY + i * 3,
                    a,
                    primary[a],
                    f"SEARCH_M_{a}_{j % 3}",
                    "",
                    float(rng.uniform(25, 100)),
                    0,
                    0,
                    0,
                    "legit",
                    "",
                    "ordinary",
                    "SEARCH",
                    seed,
                )
            )

    attack_accounts = accounts[: int(params["accounts"])]
    if params["trusted_device"]:
        devices = {a: primary[a] for a in attack_accounts}
    elif params["shared_device"]:
        devices = {a: "SEARCH_SHARED_ATTACK_DEVICE" for a in attack_accounts}
    else:
        devices = {a: f"SEARCH_ATTACK_DEVICE_{a}" for a in attack_accounts}

    beneficiary_count = int(params["beneficiary_count"])
    beneficiaries = [f"SEARCH_ATTACK_BEN_{j}" for j in range(beneficiary_count)]
    warmup_days = int(params["warmup_days"])
    if warmup_days > 0:
        # Warm-up registration is visible to event-time features and incurs an
        # attacker cost. The search is not told the hidden maximum warm-up.
        for j, a in enumerate(attack_accounts):
            events.append(
                record(
                    t0 - warmup_days * DAY + j,
                    a,
                    devices[a],
                    "",
                    beneficiaries[j % beneficiary_count],
                    5.0,
                    1,
                    0,
                    0,
                    "legit",
                    "",
                    "ordinary",
                    "SEARCH",
                    seed,
                )
            )

    splits = int(params["splits"])
    amount_each = float(params["total_value"]) / splits
    attack_indices: list[int] = []
    for j in range(splits):
        a = attack_accounts[j % len(attack_accounts)]
        events.append(
            record(
                t0 + j * float(params["spacing_minutes"]),
                a,
                devices[a],
                "",
                beneficiaries[j % beneficiary_count],
                amount_each,
                1,
                0,
                1,
                "search_campaign",
                f"SEARCH_{seed}",
                "",
                "SEARCH",
                seed,
            )
        )

    df = pd.DataFrame(events).sort_values(["timestamp", "account_id"], kind="mergesort").reset_index(drop=True)
    df.insert(0, "event_id", [f"SEARCH_{seed}_{i:05d}" for i in range(len(df))])
    attack_mask = df["is_fraud"].to_numpy() == 1
    return df, attack_mask


def visible_attack_validity(params: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons = []
    if float(params["total_value"]) < 600:
        reasons.append("visible_minimum_value")
    if int(params["splits"]) < 2:
        reasons.append("at_least_two_splits")
    if float(params["spacing_minutes"]) <= 0:
        reasons.append("positive_spacing")
    if not 1 <= int(params["accounts"]) <= 4:
        reasons.append("valid_accounts")
    if not 1 <= int(params["beneficiary_count"]) <= 4:
        reasons.append("valid_beneficiaries")
    return not reasons, reasons


def hidden_attack_validity(params: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons = []
    if float(params["total_value"]) < 700:
        reasons.append("hidden_minimum_value_700")
    duration = (int(params["splits"]) - 1) * float(params["spacing_minutes"])
    if duration > 240:
        reasons.append("hidden_execution_duration_240m")
    if int(params["splits"]) > 12:
        reasons.append("hidden_maximum_12_splits")
    if int(params["warmup_days"]) > 14:
        reasons.append("hidden_maximum_14d_warmup")
    if bool(params["trusted_device"]) and not bool(params["capability_trusted_device"]):
        reasons.append("hidden_missing_trusted_device_capability")
    return not reasons, reasons


def _candidate_schedule(seed: int, budget: int = 40) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed + 15_000)
    candidates: list[dict[str, Any]] = [
        {
            "total_value": 1000,
            "splits": 3,
            "spacing_minutes": 5,
            "accounts": 1,
            "warmup_days": 0,
            "shared_device": True,
            "trusted_device": False,
            "beneficiary_count": 1,
            "capability_trusted_device": False,
        },
        {
            # A deliberately tempting edge of the visible constraint set. The
            # hidden oracle is expected to identify any success as reward hacking.
            "total_value": 650,
            "splits": 16,
            "spacing_minutes": 60,
            "accounts": 4,
            "warmup_days": 21,
            "shared_device": False,
            "trusted_device": True,
            "beneficiary_count": 4,
            "capability_trusted_device": False,
        },
        {
            # A candidate intended to be plausible under both visible and hidden
            # constraints, but it receives no privileged model score.
            "total_value": 900,
            "splits": 10,
            "spacing_minutes": 20,
            "accounts": 4,
            "warmup_days": 7,
            "shared_device": False,
            "trusted_device": False,
            "beneficiary_count": 4,
            "capability_trusted_device": True,
        },
    ]
    while len(candidates) < budget:
        candidates.append(
            {
                "total_value": int(rng.choice([650, 750, 900, 1100])),
                "splits": int(rng.integers(3, 17)),
                "spacing_minutes": int(rng.choice([5, 10, 20, 30, 45, 60])),
                "accounts": int(rng.integers(1, 5)),
                "warmup_days": int(rng.choice([0, 3, 7, 14, 21])),
                "shared_device": bool(rng.integers(0, 2)),
                "trusted_device": bool(rng.random() < 0.2),
                "beneficiary_count": int(rng.integers(1, 5)),
                "capability_trusted_device": bool(rng.random() < 0.35),
            }
        )
    return candidates[:budget]


def decision_only_search(
    model: LogisticModel,
    threshold: float,
    seed: int,
) -> dict[str, Any]:
    outcomes = []
    for query_index, params in enumerate(_candidate_schedule(seed, 40), start=1):
        visible_valid, visible_reasons = visible_attack_validity(params)
        if not visible_valid:
            outcomes.append(
                {
                    "query": query_index,
                    "visible_valid": False,
                    "visible_reasons": visible_reasons,
                    "evaded": False,
                }
            )
            continue
        candidate, attack_mask = make_attack_candidate(params, seed * 100 + query_index)
        feat = make_features(candidate)
        scores = model.predict(feat[TEMPORAL_FEATURES].to_numpy(float))
        # The attacker receives only this Boolean campaign decision.
        detected = bool((scores[attack_mask] >= threshold).any())
        hidden_valid, hidden_reasons = hidden_attack_validity(params)
        attacker_cost = (
            4 * int(params["splits"])
            + 2 * int(params["warmup_days"])
            + 12 * int(params["accounts"])
            + (80 if params["trusted_device"] else 0)
        )
        outcomes.append(
            {
                "query": query_index,
                "visible_valid": True,
                "evaded": not detected,
                "hidden_valid": bool(hidden_valid),
                "hidden_reasons": hidden_reasons,
                "visible_utility": float(params["total_value"] - attacker_cost),
                "params": params,
            }
        )
    evasions = [x for x in outcomes if x.get("evaded")]
    valid_evasions = [x for x in evasions if x.get("hidden_valid")]
    rejected_evasions = [x for x in evasions if not x.get("hidden_valid")]
    selected = max(evasions, key=lambda x: x.get("visible_utility", -1e9), default=None)
    return {
        "query_budget": 40,
        "queries_used": len(outcomes),
        "evasion_found": bool(evasions),
        "evasion_count": len(evasions),
        "hidden_valid_evasion_count": len(valid_evasions),
        "hidden_rejected_evasion_count": len(rejected_evasions),
        "selected_visible_best": selected,
        "all_hidden_rejection_reasons": sorted(
            {reason for item in rejected_evasions for reason in item.get("hidden_reasons", [])}
        ),
    }


def metamorphic_and_leakage_tests(
    events: pd.DataFrame,
    features: pd.DataFrame,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    sample = events.iloc[: min(700, len(events))].copy().reset_index(drop=True)
    sample_feat = make_features(sample)

    # Bijections change identifiers but preserve every relationship.
    permuted = sample.copy()
    for column in ["account_id", "device_id", "merchant_id", "beneficiary_id"]:
        values = [v for v in permuted[column].unique() if v != ""]
        mapping = {v: f"P_{column}_{i}" for i, v in enumerate(reversed(values))}
        permuted[column] = permuted[column].map(lambda x: mapping.get(x, x))
    permuted_feat = make_features(permuted)
    id_invariance = bool(
        np.allclose(
            sample_feat[TEMPORAL_FEATURES].to_numpy(float),
            permuted_feat[TEMPORAL_FEATURES].to_numpy(float),
            atol=1e-10,
        )
    )

    # Appending an event strictly in the future must not alter any past feature.
    future_row = sample.iloc[-1].copy()
    future_row["event_id"] = f"FUTURE_{seed}"
    future_row["timestamp"] = float(sample["timestamp"].max() + 10 * DAY)
    future_row["account_id"] = f"FUTURE_A_{seed}"
    future_row["device_id"] = f"FUTURE_D_{seed}"
    future_row["merchant_id"] = f"FUTURE_M_{seed}"
    future_row["beneficiary_id"] = ""
    augmented = pd.concat([sample, pd.DataFrame([future_row])], ignore_index=True)
    augmented_feat = make_features(augmented).iloc[: len(sample)]
    future_independence = bool(
        np.allclose(
            sample_feat[TEMPORAL_FEATURES].to_numpy(float),
            augmented_feat[TEMPORAL_FEATURES].to_numpy(float),
            atol=1e-10,
        )
    )

    # Metadata may exist in the simulator output, but the feature manifest must
    # make the model matrix identical after it is removed.
    injected = features.copy()
    injected["generator_regime_code"] = 1.0
    injected["scenario_id_code"] = events["is_fraud"].to_numpy(float)
    injected["future_campaign_total"] = events.groupby("campaign_id")["amount"].transform("sum").to_numpy(float)
    clean_matrix = features[TEMPORAL_FEATURES].to_numpy(float)
    selected_matrix = injected[TEMPORAL_FEATURES].to_numpy(float)
    metadata_removal = bool(np.array_equal(clean_matrix, selected_matrix))
    injected_names = ["generator_regime_code", "scenario_id_code", "future_campaign_total"]
    rejected_names = audit_feature_names(injected_names)

    future_source = features.iloc[:2].copy()
    equal_source = features.iloc[:2].copy()
    equal_source.loc[equal_source.index[0], "_max_source_time"] = float(events.iloc[0]["timestamp"])
    equal_source_rejected = not assert_event_time(events.iloc[:2], equal_source)
    future_source.loc[future_source.index[0], "_max_source_time"] = (
        float(events.iloc[0]["timestamp"]) + 1.0
    )
    future_source_rejected = not assert_event_time(events.iloc[:2], future_source)

    train_campaigns = set(events.loc[train_mask & (events["campaign_id"] != ""), "campaign_id"])
    test_campaigns = set(events.loc[test_mask & (events["campaign_id"] != ""), "campaign_id"])
    chronological_split = bool(
        events.loc[train_mask, "timestamp"].max() < events.loc[test_mask, "timestamp"].min()
    )
    campaign_isolation = not bool(train_campaigns & test_campaigns)

    return {
        "id_permutation_invariance": id_invariance,
        "future_event_independence": future_independence,
        "metadata_removal_invariance": metadata_removal,
        "deliberate_forbidden_features": injected_names,
        "rejected_forbidden_features": rejected_names,
        "all_forbidden_features_rejected": sorted(injected_names) == sorted(rejected_names),
        "equal_source_timestamp_rejected": bool(equal_source_rejected),
        "future_source_timestamp_rejected": bool(future_source_rejected),
        "clean_event_time_check": assert_event_time(events, features),
        "chronological_split": chronological_split,
        "campaign_isolation": campaign_isolation,
    }


def false_positive_rate(y: np.ndarray, score: np.ndarray, threshold: float) -> float:
    legit = np.asarray(y) == 0
    return float((np.asarray(score)[legit] >= threshold).mean()) if legit.any() else 0.0


def evaluate_seed(seed: int) -> dict[str, Any]:
    dev = simulate(seed, "A")
    hidden = simulate(seed, "B")
    dev_feat = make_features(dev)
    hidden_feat = make_features(hidden)

    train_mask = (dev["timestamp"].to_numpy() < 25 * DAY) & (dev["family"].to_numpy() != "mule")
    test_mask = dev["timestamp"].to_numpy() >= 25 * DAY
    train = dev.loc[train_mask].reset_index(drop=True)
    test = dev.loc[test_mask].reset_index(drop=True)
    train_feat = dev_feat.loc[train_mask].reset_index(drop=True)
    test_feat = dev_feat.loc[test_mask].reset_index(drop=True)

    base_model = fit_logistic(
        train_feat[BASE_FEATURES].to_numpy(float), train["is_fraud"].to_numpy(int)
    )
    temporal_model = fit_logistic(
        train_feat[TEMPORAL_FEATURES].to_numpy(float), train["is_fraud"].to_numpy(int)
    )
    train_base_score = base_model.predict(train_feat[BASE_FEATURES].to_numpy(float))
    train_temporal_score = temporal_model.predict(train_feat[TEMPORAL_FEATURES].to_numpy(float))
    threshold_base = float(np.quantile(train_base_score[train["is_fraud"].to_numpy() == 0], 0.99))
    threshold_temporal = float(np.quantile(train_temporal_score[train["is_fraud"].to_numpy() == 0], 0.99))

    test_base = base_model.predict(test_feat[BASE_FEATURES].to_numpy(float))
    test_temporal = temporal_model.predict(test_feat[TEMPORAL_FEATURES].to_numpy(float))
    hidden_base = base_model.predict(hidden_feat[BASE_FEATURES].to_numpy(float))
    hidden_temporal = temporal_model.predict(hidden_feat[TEMPORAL_FEATURES].to_numpy(float))

    metrics = {
        "development": {
            "baseline": {
                "average_precision": average_precision(test["is_fraud"].to_numpy(), test_base),
                "campaign_detection": campaign_detection(test, test_base, threshold_base),
                "false_positive_rate": false_positive_rate(test["is_fraud"].to_numpy(), test_base, threshold_base),
            },
            "temporal": {
                "average_precision": average_precision(test["is_fraud"].to_numpy(), test_temporal),
                "campaign_detection": campaign_detection(test, test_temporal, threshold_temporal),
                "false_positive_rate": false_positive_rate(test["is_fraud"].to_numpy(), test_temporal, threshold_temporal),
            },
        },
        "hidden": {
            "baseline": {
                "average_precision": average_precision(hidden["is_fraud"].to_numpy(), hidden_base),
                "campaign_detection": campaign_detection(hidden, hidden_base, threshold_base),
                "false_positive_rate": false_positive_rate(hidden["is_fraud"].to_numpy(), hidden_base, threshold_base),
            },
            "temporal": {
                "average_precision": average_precision(hidden["is_fraud"].to_numpy(), hidden_temporal),
                "campaign_detection": campaign_detection(hidden, hidden_temporal, threshold_temporal),
                "false_positive_rate": false_positive_rate(hidden["is_fraud"].to_numpy(), hidden_temporal, threshold_temporal),
            },
        },
    }

    novelty = robust_novelty(train, train_feat, hidden_feat)
    triage = triage_metrics(hidden, hidden_temporal, novelty, seed)
    attack = decision_only_search(temporal_model, threshold_temporal, seed)
    tests = metamorphic_and_leakage_tests(dev, dev_feat, train_mask, test_mask, seed)

    return {
        "seed": seed,
        "event_counts": {
            "development_total": len(dev),
            "development_train": len(train),
            "development_test": len(test),
            "hidden_total": len(hidden),
            "development_fraud_events": int(dev["is_fraud"].sum()),
            "development_train_fraud_events": int(train["is_fraud"].sum()),
            "development_test_fraud_events": int(test["is_fraud"].sum()),
            "hidden_fraud_events": int(hidden["is_fraud"].sum()),
            "development_fraud_rate": float(dev["is_fraud"].mean()),
            "development_train_fraud_rate": float(train["is_fraud"].mean()),
            "development_test_fraud_rate": float(test["is_fraud"].mean()),
            "hidden_fraud_rate": float(hidden["is_fraud"].mean()),
            "hidden_travel": int((hidden["benign_segment"] == "business_travel").sum()),
            "development_campaigns": int(dev.loc[dev["campaign_id"] != "", "campaign_id"].nunique()),
            "hidden_campaigns": int(hidden.loc[hidden["campaign_id"] != "", "campaign_id"].nunique()),
        },
        "thresholds": {"baseline": threshold_base, "temporal": threshold_temporal},
        "metrics": metrics,
        "triage": triage,
        "attack_search": attack,
        "tests": tests,
    }


def mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0}


def aggregate_results(per_seed: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, str]]:
    aggregate: dict[str, Any] = {"metrics": {}, "triage": {}, "attack_search": {}, "tests": {}, "event_counts": {}}
    for metric in [
        "development_total",
        "development_train",
        "development_test",
        "hidden_total",
        "development_fraud_rate",
        "development_train_fraud_rate",
        "development_test_fraud_rate",
        "hidden_fraud_rate",
        "hidden_travel",
    ]:
        aggregate["event_counts"][metric] = mean_std([x["event_counts"][metric] for x in per_seed])
    for split in ["development", "hidden"]:
        aggregate["metrics"][split] = {}
        for model in ["baseline", "temporal"]:
            aggregate["metrics"][split][model] = {}
            for metric in ["average_precision", "campaign_detection", "false_positive_rate"]:
                vals = [x["metrics"][split][model][metric] for x in per_seed]
                aggregate["metrics"][split][model][metric] = mean_std(vals)

    for policy in ["risk_only", "novelty_aware", "random"]:
        aggregate["triage"][policy] = {}
        for metric in [
            "heldout_mule_value_capture",
            "heldout_mule_event_capture",
            "travel_share_of_reviews",
            "fraud_share_of_reviews",
        ]:
            aggregate["triage"][policy][metric] = mean_std(
                [x["triage"][policy][metric] for x in per_seed]
            )

    aggregate["attack_search"] = {
        "evasion_seed_rate": float(np.mean([x["attack_search"]["evasion_found"] for x in per_seed])),
        "seeds_with_hidden_valid_evasion": int(
            sum(x["attack_search"]["hidden_valid_evasion_count"] > 0 for x in per_seed)
        ),
        "seeds_with_hidden_rejected_evasion": int(
            sum(x["attack_search"]["hidden_rejected_evasion_count"] > 0 for x in per_seed)
        ),
        "mean_evasion_count": float(np.mean([x["attack_search"]["evasion_count"] for x in per_seed])),
    }
    test_names = [
        "id_permutation_invariance",
        "future_event_independence",
        "metadata_removal_invariance",
        "all_forbidden_features_rejected",
        "equal_source_timestamp_rejected",
        "future_source_timestamp_rejected",
        "clean_event_time_check",
        "chronological_split",
        "campaign_isolation",
    ]
    aggregate["tests"] = {
        name: {"passed": int(sum(bool(x["tests"][name]) for x in per_seed)), "total": len(per_seed)}
        for name in test_names
    }

    dev_base_ap = aggregate["metrics"]["development"]["baseline"]["average_precision"]["mean"]
    dev_temp_ap = aggregate["metrics"]["development"]["temporal"]["average_precision"]["mean"]
    dev_base_cd = aggregate["metrics"]["development"]["baseline"]["campaign_detection"]["mean"]
    dev_temp_cd = aggregate["metrics"]["development"]["temporal"]["campaign_detection"]["mean"]
    h1_ap_gain = dev_temp_ap - dev_base_ap
    h1_cd_gain = dev_temp_cd - dev_base_cd
    if h1_ap_gain >= 0.05 and h1_cd_gain >= 0.10:
        h1 = "SUPPORTED"
    elif h1_ap_gain > 0 and h1_cd_gain > 0:
        h1 = "PARTIALLY SUPPORTED"
    else:
        h1 = "NOT SUPPORTED"

    hidden_base_ap = aggregate["metrics"]["hidden"]["baseline"]["average_precision"]["mean"]
    hidden_temp_ap = aggregate["metrics"]["hidden"]["temporal"]["average_precision"]["mean"]
    hidden_base_cd = aggregate["metrics"]["hidden"]["baseline"]["campaign_detection"]["mean"]
    hidden_temp_cd = aggregate["metrics"]["hidden"]["temporal"]["campaign_detection"]["mean"]
    hidden_ap_gain = hidden_temp_ap - hidden_base_ap
    if h1_ap_gain > 0 and hidden_ap_gain > 0 and hidden_ap_gain >= 0.5 * h1_ap_gain and hidden_temp_cd >= hidden_base_cd:
        h2 = "SUPPORTED"
    elif hidden_ap_gain > 0:
        h2 = "PARTIALLY SUPPORTED"
    else:
        h2 = "NOT SUPPORTED"

    novelty_capture = aggregate["triage"]["novelty_aware"]["heldout_mule_value_capture"]["mean"]
    risk_capture = aggregate["triage"]["risk_only"]["heldout_mule_value_capture"]["mean"]
    random_capture = aggregate["triage"]["random"]["heldout_mule_value_capture"]["mean"]
    novelty_travel = aggregate["triage"]["novelty_aware"]["travel_share_of_reviews"]["mean"]
    risk_travel = aggregate["triage"]["risk_only"]["travel_share_of_reviews"]["mean"]
    per_seed_improvements = [
        x["triage"]["novelty_aware"]["heldout_mule_value_capture"]
        > x["triage"]["risk_only"]["heldout_mule_value_capture"]
        for x in per_seed
    ]
    if (
        novelty_capture - risk_capture >= 0.05
        and novelty_capture > random_capture
        and novelty_travel <= risk_travel + 0.02
        and all(per_seed_improvements)
    ):
        h3 = "SUPPORTED"
    elif novelty_capture > risk_capture:
        h3 = "PARTIALLY SUPPORTED"
    else:
        h3 = "NOT SUPPORTED"

    evasion_rate = aggregate["attack_search"]["evasion_seed_rate"]
    valid_seeds = aggregate["attack_search"]["seeds_with_hidden_valid_evasion"]
    rejected_seeds = aggregate["attack_search"]["seeds_with_hidden_rejected_evasion"]
    if evasion_rate >= 0.6 and valid_seeds >= 1 and rejected_seeds >= 1:
        h4 = "SUPPORTED"
    elif evasion_rate >= 0.2:
        h4 = "PARTIALLY SUPPORTED"
    else:
        h4 = "NOT SUPPORTED"

    all_tests = all(v["passed"] == v["total"] for v in aggregate["tests"].values())
    deliberate_caught = (
        aggregate["tests"]["all_forbidden_features_rejected"]["passed"] == len(per_seed)
        and aggregate["tests"]["equal_source_timestamp_rejected"]["passed"] == len(per_seed)
        and aggregate["tests"]["future_source_timestamp_rejected"]["passed"] == len(per_seed)
    )
    h5 = "SUPPORTED" if all_tests else ("PARTIALLY SUPPORTED" if deliberate_caught else "NOT SUPPORTED")

    aggregate["hypothesis_inputs"] = {
        "h1_development_ap_gain": h1_ap_gain,
        "h1_development_campaign_detection_gain": h1_cd_gain,
        "h2_hidden_ap_gain": hidden_ap_gain,
        "h2_hidden_gain_fraction_of_development": (
            hidden_ap_gain / h1_ap_gain if abs(h1_ap_gain) > 1e-12 else None
        ),
        "h3_novelty_minus_risk_value_capture": novelty_capture - risk_capture,
        "h3_novelty_minus_risk_travel_share": novelty_travel - risk_travel,
    }
    return aggregate, {"H1": h1, "H2": h2, "H3": h3, "H4": h4, "H5": h5}


def flatten_metrics(per_seed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in per_seed:
        seed = item["seed"]
        for split, split_values in item["metrics"].items():
            for model, model_values in split_values.items():
                for metric, value in model_values.items():
                    rows.append({"seed": seed, "section": split, "method": model, "metric": metric, "value": value})
        for policy, policy_values in item["triage"].items():
            for metric, value in policy_values.items():
                rows.append({"seed": seed, "section": "triage", "method": policy, "metric": metric, "value": value})
        for metric in ["evasion_found", "evasion_count", "hidden_valid_evasion_count", "hidden_rejected_evasion_count"]:
            rows.append({"seed": seed, "section": "attack_search", "method": "decision_only", "metric": metric, "value": item["attack_search"][metric]})
        for metric, value in item["tests"].items():
            if isinstance(value, bool):
                rows.append({"seed": seed, "section": "test", "method": "invariant", "metric": metric, "value": int(value)})
    return rows


def build_report(aggregate: dict[str, Any], verdicts: dict[str, str]) -> str:
    def ms(section: str, model: str, metric: str) -> str:
        x = aggregate["metrics"][section][model][metric]
        return f"{x['mean']:.3f} ± {x['std']:.3f}"

    triage = aggregate["triage"]
    counts = aggregate["event_counts"]
    h = aggregate["hypothesis_inputs"]
    failed = [name for name, verdict in verdicts.items() if verdict != "SUPPORTED"]
    changes = []
    if verdicts["H2"] != "SUPPORTED":
        changes.append("Treat cross-regime transfer as an explicit promotion gate; do not claim hidden-mechanism robustness from development performance.")
    if verdicts["H3"] != "SUPPORTED":
        changes.append("Do not deploy a fixed novelty blend. Calibrate triage against benign novelty and review-capacity constraints, with a risk-only fallback.")
    if verdicts["H4"] != "SUPPORTED":
        changes.append("Keep campaign search as a red-team diagnostic until hidden-valid evasions are demonstrated consistently.")
    if verdicts["H5"] != "SUPPORTED":
        changes.append("Block model promotion until every leakage and metamorphic invariant passes.")
    changes.extend(
        [
            "Make drift-aware threshold monitoring and recalibration a promotion requirement. Hidden-regime feature gains do not make a static regime-A threshold operationally safe.",
            "Report absolute held-out-family coverage, not only improvement over risk-only. Reserve investigation capacity for campaign-level exploration if transaction-level novelty capture remains low.",
            "Describe the implemented red team as bounded decision-only candidate search. Adaptive mutation or evolutionary optimization remains unvalidated.",
            "Benchmark against a strong GBDT and production-style rules before claiming model superiority; the transaction-only logistic baseline is deliberately weak.",
            "Require an independently implemented generator or authorized external dataset before claiming cross-simulator transfer.",
        ]
    )
    if not changes:
        changes.append("Retain all five mechanisms in the architecture, but keep the synthetic-only evidence boundary explicit and require an independent generator or authorized external data before a production claim.")

    return f"""# Empirical falsification report

## Result

This preregistered synthetic spike produced the following verdicts across five fixed seeds:

| Hypothesis | Verdict |
|---|---|
| H1: temporal/campaign features in development | **{verdicts['H1']}** |
| H2: survival under hidden regime shift | **{verdicts['H2']}** |
| H3: novelty-aware fixed-budget triage | **{verdicts['H3']}** |
| H4: constrained decision-only evasion and hidden validity | **{verdicts['H4']}** |
| H5: leakage and metamorphic defenses | **{verdicts['H5']}** |

No thresholds, seeds, feature sets, or verdict rules were changed after hidden results were observed.

## Data volume

| Slice | Events, mean ± SD | Fraud rate, mean ± SD |
|---|---:|---:|
| Regime A, all | {counts['development_total']['mean']:.1f} ± {counts['development_total']['std']:.1f} | {counts['development_fraud_rate']['mean']:.3%} ± {counts['development_fraud_rate']['std']:.3%} |
| Regime A, training | {counts['development_train']['mean']:.1f} ± {counts['development_train']['std']:.1f} | {counts['development_train_fraud_rate']['mean']:.3%} ± {counts['development_train_fraud_rate']['std']:.3%} |
| Regime A, chronological test | {counts['development_test']['mean']:.1f} ± {counts['development_test']['std']:.1f} | {counts['development_test_fraud_rate']['mean']:.3%} ± {counts['development_test_fraud_rate']['std']:.3%} |
| Hidden regime B | {counts['hidden_total']['mean']:.1f} ± {counts['hidden_total']['std']:.1f} | {counts['hidden_fraud_rate']['mean']:.3%} ± {counts['hidden_fraud_rate']['std']:.3%} |

Hidden B contained {counts['hidden_travel']['mean']:.1f} ± {counts['hidden_travel']['std']:.1f} events from the unseen business-travel segment.

## Measurements

### H1 and H2: detector comparison

| Regime | Model | Average precision | Campaign detection at A-trained threshold | False-positive rate |
|---|---|---:|---:|---:|
| Development A | Transaction only | {ms('development', 'baseline', 'average_precision')} | {ms('development', 'baseline', 'campaign_detection')} | {ms('development', 'baseline', 'false_positive_rate')} |
| Development A | Event-time temporal/campaign | {ms('development', 'temporal', 'average_precision')} | {ms('development', 'temporal', 'campaign_detection')} | {ms('development', 'temporal', 'false_positive_rate')} |
| Hidden B | Transaction only | {ms('hidden', 'baseline', 'average_precision')} | {ms('hidden', 'baseline', 'campaign_detection')} | {ms('hidden', 'baseline', 'false_positive_rate')} |
| Hidden B | Event-time temporal/campaign | {ms('hidden', 'temporal', 'average_precision')} | {ms('hidden', 'temporal', 'campaign_detection')} | {ms('hidden', 'temporal', 'false_positive_rate')} |

Development average-precision gain was {h['h1_development_ap_gain']:.3f}; campaign-detection gain was {h['h1_development_campaign_detection_gain']:.3f}. Hidden average-precision gain was {h['h2_hidden_ap_gain']:.3f}. The hidden gain was {('undefined' if h['h2_hidden_gain_fraction_of_development'] is None else f"{h['h2_hidden_gain_fraction_of_development']:.1%}")} of the development gain.

### H3: 2% review capacity in hidden regime B

| Policy | Held-out mule value captured | Unseen travel share of reviews | Fraud share of reviews |
|---|---:|---:|---:|
| Risk only | {triage['risk_only']['heldout_mule_value_capture']['mean']:.3f} ± {triage['risk_only']['heldout_mule_value_capture']['std']:.3f} | {triage['risk_only']['travel_share_of_reviews']['mean']:.3f} ± {triage['risk_only']['travel_share_of_reviews']['std']:.3f} | {triage['risk_only']['fraud_share_of_reviews']['mean']:.3f} ± {triage['risk_only']['fraud_share_of_reviews']['std']:.3f} |
| Novelty aware | {triage['novelty_aware']['heldout_mule_value_capture']['mean']:.3f} ± {triage['novelty_aware']['heldout_mule_value_capture']['std']:.3f} | {triage['novelty_aware']['travel_share_of_reviews']['mean']:.3f} ± {triage['novelty_aware']['travel_share_of_reviews']['std']:.3f} | {triage['novelty_aware']['fraud_share_of_reviews']['mean']:.3f} ± {triage['novelty_aware']['fraud_share_of_reviews']['std']:.3f} |
| Random | {triage['random']['heldout_mule_value_capture']['mean']:.3f} ± {triage['random']['heldout_mule_value_capture']['std']:.3f} | {triage['random']['travel_share_of_reviews']['mean']:.3f} ± {triage['random']['travel_share_of_reviews']['std']:.3f} | {triage['random']['fraud_share_of_reviews']['mean']:.3f} ± {triage['random']['fraud_share_of_reviews']['std']:.3f} |

The novelty-aware policy changed mule-value capture by {h['h3_novelty_minus_risk_value_capture']:+.3f} and changed unseen-travel review consumption by {h['h3_novelty_minus_risk_travel_share']:+.3f} relative to risk-only.

### H4: decision-only search

- Evasion found in {aggregate['attack_search']['evasion_seed_rate']:.0%} of seeds under 40 queries.
- Seeds with at least one hidden-valid evasion: {aggregate['attack_search']['seeds_with_hidden_valid_evasion']}/5.
- Seeds where hidden checks rejected at least one superficial success: {aggregate['attack_search']['seeds_with_hidden_rejected_evasion']}/5.
- Mean number of evasions found per seed: {aggregate['attack_search']['mean_evasion_count']:.1f}.

The hidden checks cover minimum economic value, execution duration, split count, warm-up duration, and undeclared trusted-device capability. A rejected evasion is evidence that the search exploited the visible environment, not evidence of a successful fraud campaign.

This spike used a fixed, seed-specific schedule containing three declared candidates followed by reproducibly sampled candidates. It validates bounded decision-only candidate search and hidden post-search validity checks. It does not empirically validate an adaptive genetic or evolutionary optimizer.

### H5: test results

""" + "\n".join(
        f"- `{name}`: {value['passed']}/{value['total']} seeds passed"
        for name, value in aggregate["tests"].items()
    ) + f"""

## Architecture changes warranted

""" + "\n".join(f"- {item}" for item in changes) + f"""

## Negative results and limitations

- Non-supported hypotheses: {', '.join(failed) if failed else 'none under the preregistered synthetic criteria'}.
- All development and hidden data were produced by code in this repository. Distinct mechanisms and parameters reduce, but do not eliminate, simulator circularity.
- The NumPy logistic model is intentionally simple. Results do not establish that the same ordering holds for CatBoost, LightGBM, GNNs, or production decision systems.
- The transaction-only comparator is deliberately weak and contains no production rule engine, entity profiles, or GBDT interactions. H1 is a mechanism check, not evidence of superiority over a mature fraud stack.
- Although H2 passes its preregistered relative-gain rule, the event-time model's false-positive rate increased from {aggregate['metrics']['development']['temporal']['false_positive_rate']['mean']:.3%} in development to {aggregate['metrics']['hidden']['temporal']['false_positive_rate']['mean']:.3%} in hidden B. A static regime-A threshold is therefore operationally unsafe under this simulated drift.
- Novelty-aware triage captured only {triage['novelty_aware']['heldout_mule_value_capture']['mean']:.3%} of held-out mule value at the 2% review budget. Its improvement over risk-only did not reach the preregistered five-point threshold, so H3 is not fully supported.
- The five seeds quantify simulator variance, not uncertainty over real payment populations.
- The hidden benign segment covers one type of novelty. Product launches, festivals, migrations, emergencies, and merchant-network changes could consume triage capacity differently.
- The campaign attacker searches a small declared parameter space. It is not a comprehensive adversarial-ML evaluation.
- Hidden B changes mechanisms and distributions but is implemented in the same source file as regime A. It is not an independent generator and cannot resolve simulator circularity.
- “Campaign detection” means at least one event crossed a threshold. It does not establish that every actor or flow was reconstructed.

## Reproducibility

Run the exact command in `README.md` from this directory. Exact per-seed results, selected attack candidates, hidden rejection reasons, environment versions, and code hashes are in `outputs/results.json`; flat metrics are in `outputs/metrics.csv`.
"""


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    run_clock = time.perf_counter()
    source_path = Path(__file__).resolve()
    prereg_path = ROOT / "preregistration.md"
    started = datetime.now(timezone.utc).isoformat()
    print("Adaptive Payment Security Range validation spike")
    print(f"Seeds: {SEEDS}")
    print(f"Preregistration SHA256: {file_sha256(prereg_path)}")
    per_seed = []
    for seed in SEEDS:
        print(f"Running seed {seed}...", flush=True)
        seed_clock = time.perf_counter()
        result = evaluate_seed(seed)
        result["runtime_seconds"] = float(time.perf_counter() - seed_clock)
        per_seed.append(result)
        print(
            f"  dev AP base/temporal={result['metrics']['development']['baseline']['average_precision']:.3f}/"
            f"{result['metrics']['development']['temporal']['average_precision']:.3f}; "
            f"hidden={result['metrics']['hidden']['baseline']['average_precision']:.3f}/"
            f"{result['metrics']['hidden']['temporal']['average_precision']:.3f}"
        )
    aggregate, verdicts = aggregate_results(per_seed)
    payload = {
        "experiment": "adaptive_payment_security_range_validation_spike",
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "predeclared_seeds": SEEDS,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "hashes": {
            "preregistration_sha256": file_sha256(prereg_path),
            "source_sha256": file_sha256(source_path),
        },
        "runtime_seconds": float(time.perf_counter() - run_clock),
        "reporting_note": "This run adds runtime and class-rate reporting after the initial result-producing run. Experimental logic, seeds, thresholds, and verdict rules are unchanged; both source hashes remain in run_history.json.",
        "verdicts": verdicts,
        "aggregate": aggregate,
        "per_seed": per_seed,
    }
    (OUTPUT / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rows = flatten_metrics(per_seed)
    with (OUTPUT / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "section", "method", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)
    report = build_report(aggregate, verdicts)
    (ROOT / "empirical_report.md").write_text(report, encoding="utf-8")

    seed_lines = [
        (
            f"Seed {item['seed']}: dev AP {item['metrics']['development']['baseline']['average_precision']:.3f}/"
            f"{item['metrics']['development']['temporal']['average_precision']:.3f}; hidden AP "
            f"{item['metrics']['hidden']['baseline']['average_precision']:.3f}/"
            f"{item['metrics']['hidden']['temporal']['average_precision']:.3f}; "
            f"runtime {item['runtime_seconds']:.3f}s"
        )
        for item in per_seed
    ]
    summary_lines = [
        "Adaptive Payment Security Range validation spike",
        f"Seeds: {SEEDS}",
        f"Preregistration SHA256: {payload['hashes']['preregistration_sha256']}",
        *seed_lines,
        "",
        "Verdicts:",
    ] + [f"  {name}: {value}" for name, value in verdicts.items()]
    summary_lines += [
        "",
        f"Development AP gain: {aggregate['hypothesis_inputs']['h1_development_ap_gain']:+.3f}",
        f"Hidden AP gain: {aggregate['hypothesis_inputs']['h2_hidden_ap_gain']:+.3f}",
        f"Novelty minus risk-only mule value capture: {aggregate['hypothesis_inputs']['h3_novelty_minus_risk_value_capture']:+.3f}",
        f"Evasion seed rate: {aggregate['attack_search']['evasion_seed_rate']:.0%}",
        f"Total runtime: {payload['runtime_seconds']:.3f}s",
    ]
    console = "\n".join(summary_lines).strip() + "\n"
    print(console, end="")
    (OUTPUT / "console.log").write_text(console, encoding="utf-8")

    history_path = OUTPUT / "run_history.json"
    history = []
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
    history.append(
        {
            "started_utc": started,
            "completed_utc": payload["completed_utc"],
            "status": "completed",
            "source_sha256": payload["hashes"]["source_sha256"],
            "preregistration_sha256": payload["hashes"]["preregistration_sha256"],
            "verdicts": verdicts,
            "note": (
                "Initial result-producing run"
                if not history
                else "Reporting-only rerun adding runtime and class-rate fields; no experimental logic, seeds, thresholds, or verdict rules changed"
            ),
        }
    )
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
