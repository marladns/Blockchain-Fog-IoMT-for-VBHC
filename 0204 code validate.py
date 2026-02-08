from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
import zlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _csv_escape_cell(value: Any) -> str:
    if value is None:
        text = ""
    else:
        text = str(value)
    if any(ch in text for ch in [",", "\"", "\n", "\r"]):
        text = text.replace('"', '""')
        return f'"{text}"'
    return text


def _write_csv_rows(*, out_csv: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(str(x) for x in fieldnames) + "\n")
        for row in rows:
            f.write(",".join(_csv_escape_cell(row.get(fn, "")) for fn in fieldnames) + "\n")


def _parse_csv_line(line: str, *, delimiter: str = ",") -> List[str]:
    cells: List[str] = []
    buf: List[str] = []
    in_quotes = False
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        if in_quotes:
            if ch == '"':
                # Escaped quote
                if (i + 1) < n and line[i + 1] == '"':
                    buf.append('"')
                    i += 1
                else:
                    in_quotes = False
            else:
                buf.append(ch)
        else:
            if ch == delimiter:
                cells.append("".join(buf))
                buf = []
            elif ch == '"':
                in_quotes = True
            else:
                buf.append(ch)
        i += 1
    cells.append("".join(buf))
    return cells


def _iter_csv_rows(path: Path, *, delimiter: str = ",") -> Tuple[List[str], Iterable[List[str]]]:
    # Minimal CSV reader (quote-aware) to avoid dependency on stdlib csv.
    # Assumes a header row.
    f = path.open("r", encoding="utf-8-sig", errors="replace", newline="")

    def _rows() -> Iterable[List[str]]:
        with f:
            header_line = f.readline()
            if not header_line:
                return
            # header already read by outer scope
            for line in f:
                yield _parse_csv_line(line.rstrip("\n\r"), delimiter=delimiter)

    header_line = f.readline()
    if not header_line:
        return ([], [])
    header = [h.strip() for h in _parse_csv_line(header_line.rstrip("\n\r"), delimiter=delimiter)]
    return (header, _rows())


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    m = len(s) // 2
    if len(s) % 2 == 1:
        return float(s[m])
    return float((s[m - 1] + s[m]) / 2.0)


def _run_iomt_sampling_checks_no_pandas(
    *,
    sampling_csv: Path,
    sampling_limit: int,
    sampling_chunksize: int,
    time_col: str,
    metric_col: str,
    all_cols: bool,
    threshold_auto: bool,
    threshold: float,
    sigma: int,
    epsilon: float,
    alpha: float,
    contract_attainment_min: float,
    incentive_attainment_min: float,
    gamma: float,
    proof_mode: str,
    required_sigs: int,
    fog_smooth_window: int,
    fog_store_dir: Optional[Path],
    fog_store_max_items: int,
    fog_agg_window: int,
    fog_agg_func: str,
    fog_cache_ttl_seconds: float,
    fog_offline_rate: float,
    fog_byzantine_rate: float,
    fog_byzantine_noise_std: float,
    fog_consensus_max_rel_range: float,
    fog_enforce_consensus: bool,
    seed: int,
) -> List[Check]:
    checks: List[Check] = []

    header, rows_iter = _iter_csv_rows(sampling_csv)
    if not header:
        return [_warn("IoMT sampling CSV header readable", "empty file or missing header")]

    cols_lower = {str(c).strip().lower(): i for i, c in enumerate(header)}
    time_key = str(time_col).strip().lower()
    metric_key = str(metric_col).strip().lower()

    id_cols = {"sample_id", "resample_pos"}
    if all_cols:
        candidates = [c for c in header if str(c).strip().lower() not in id_cols]
    else:
        if metric_key not in cols_lower:
            return [_warn("IoMT metric column exists", f"missing column: {metric_col}")]
        candidates = [header[int(cols_lower[metric_key])]]

    checks.append(_ok("IoMT columns selected", f"count={len(candidates)}"))

    # Minimal authorized fog nodes / oracle for verification.
    authorized_nodes = {"fog_a": "secret_a", "fog_b": "secret_b", "fog_c": "secret_c"}
    oracle_secrets = {"oracle_1": "oracle_secret"}

    fog = FogCluster.from_secrets(
        node_secrets=authorized_nodes,
        smooth_window=int(fog_smooth_window),
        agg_window=int(fog_agg_window),
        agg_func=str(fog_agg_func),
        store_dir=fog_store_dir,
        store_max_items=int(fog_store_max_items),
        cache_ttl_seconds=float(fog_cache_ttl_seconds),
    )

    # Prepare indices
    time_idx = cols_lower.get(time_key) if time_key in cols_lower else None
    cand_indices: Dict[str, int] = {}
    for c in candidates:
        k = str(c).strip().lower()
        if k in cols_lower:
            cand_indices[str(c)] = int(cols_lower[k])

    if not cand_indices:
        return [_warn("IoMT columns selected", "no candidate columns found in header")]

    # Collect in-memory (default limit is 5000 so acceptable).
    ts_all: List[Optional[float]] = []
    values_by_col: Dict[str, List[Optional[float]]] = {c: [] for c in candidates}

    row_index = 0
    for cells in rows_iter:
        if sampling_limit is not None and int(sampling_limit) > 0 and row_index >= int(sampling_limit):
            break
        # pad
        if len(cells) < len(header):
            cells = list(cells) + [""] * (len(header) - len(cells))

        # timestamp
        if time_idx is None:
            ts = float(row_index)
        else:
            raw_ts = str(cells[int(time_idx)]).strip()
            try:
                ts = float(raw_ts) if raw_ts != "" else float("nan")
            except Exception:
                ts = float("nan")
        ts_all.append(ts if not (isinstance(ts, float) and math.isnan(ts)) else None)

        for col_name, col_idx in cand_indices.items():
            raw = str(cells[int(col_idx)]).strip()
            if raw == "":
                values_by_col[col_name].append(None)
                continue
            try:
                v = float(raw)
                if math.isnan(v):
                    values_by_col[col_name].append(None)
                else:
                    values_by_col[col_name].append(v)
            except Exception:
                values_by_col[col_name].append(None)

        row_index += 1

    ts_valid = [t for t in ts_all if isinstance(t, (int, float))]
    if not ts_valid:
        return [_warn("IoMT timestamp column valid", f"no valid timestamps in: {time_col}")]
    now_ts = float(max(ts_valid))
    min_ts = float(min(ts_valid))
    delta_t_max = max(0.0, (now_ts - min_ts) + 1e-9)

    contract_settled = 0
    contract_pending = 0
    contract_failed = 0
    incentive_full = 0
    incentive_partial = 0
    incentive_none = 0

    for col_name in candidates:
        series = values_by_col.get(col_name, [])
        total_n = len(series)
        miss_n = sum(1 for v in series if v is None)
        missing_rate = (float(miss_n) / float(total_n)) if total_n > 0 else 1.0
        checks.append(_ok(f"IoMT col {col_name} missing-rate", f"missing={missing_rate:.2%}"))

        datapoints: List[DataPoint] = []
        values: List[float] = []
        for t, v in zip(ts_all, series):
            if t is None or v is None:
                continue
            datapoints.append(DataPoint(timestamp=float(t), type=str(col_name).strip().lower(), value=float(v)))
            values.append(float(v))

        if not datapoints:
            checks.append(_warn(f"IoMT col {col_name} has data", "all values missing after coercion"))
            continue

        if threshold_auto:
            thr = float(_median(values))
        else:
            thr = float(threshold)
        rules = RuleSet(threshold=thr)
        weights = {datapoints[0].type: 1.0}

        try:
            processed_by_node = fog.process(
                metric=str(col_name),
                datapoints=datapoints,
                now_ts=now_ts,
                offline_rate=float(fog_offline_rate),
                byzantine_rate=float(fog_byzantine_rate),
                byzantine_noise_std=float(fog_byzantine_noise_std),
                rng_seed=int(seed),
            )
            if len(processed_by_node) == 0:
                checks.append(_warn(f"Fog processes {col_name}", "no available fog nodes (all offline)"))
                contract_failed += 1
                incentive_none += 1
                continue
            checks.append(
                _ok(
                    f"Fog processes {col_name}",
                    f"nodes={len(processed_by_node)} smooth_window={int(fog_smooth_window)} agg_window={int(fog_agg_window)} agg_func={str(fog_agg_func)} ttl={float(fog_cache_ttl_seconds)}s offline_rate={float(fog_offline_rate):.2f} byz_rate={float(fog_byzantine_rate):.2f}",
                )
            )
        except Exception as e:
            checks.append(_warn(f"Fog processes {col_name}", f"error: {e}"))
            continue

        try:
            scores = fog.compute_scores(
                metric=str(col_name),
                processed=processed_by_node,
                rules=rules,
                sigma=int(sigma),
                now_ts=now_ts,
                delta_t_max=delta_t_max,
                weights=weights,
            )

            score_values = sorted(int(v) for v in scores.values())
            v_score = int(score_values[len(score_values) // 2])
            all_same = all(v == score_values[0] for v in score_values)
            if len(score_values) >= 2:
                rel_range = (max(score_values) - min(score_values)) / float(max(1, abs(v_score)))
            else:
                rel_range = 0.0

            consistency_msg = f"all_same={all_same} rel_range={rel_range:.4f} scores={score_values}"
            if float(fog_consensus_max_rel_range) > 0.0 and rel_range > float(fog_consensus_max_rel_range):
                checks.append(
                    _warn(f"Fog score consistency {col_name}", consistency_msg)
                    if bool(fog_enforce_consensus)
                    else _ok(f"Fog score consistency {col_name}", consistency_msg)
                )
                if bool(fog_enforce_consensus):
                    contract_failed += 1
                    incentive_none += 1
                    continue
            else:
                checks.append(_ok(f"Fog score consistency {col_name}", consistency_msg))

            checks.append(_ok(f"Alg2 (fog) runs on {col_name}", f"threshold={thr} v_score={v_score}"))
        except Exception as e:
            checks.append(_warn(f"Alg2 (fog) runs on {col_name}", f"error: {e}"))
            continue

        payload = _score_payload(metric=str(col_name), v_score=int(v_score), now_ts=now_ts)
        is_authorized = False
        mode = str(proof_mode).strip().lower()
        if mode == "none":
            is_authorized = True
        elif mode == "oracle":
            oracle_id = "oracle_1"
            proof = OracleProof(oracle_id=oracle_id, signature=_sign_payload(oracle_secrets[oracle_id], payload))
            is_authorized = VERIFY_ORACLE(payload=payload, proof=proof, oracle_secrets=oracle_secrets)
        else:
            proof = fog.multisig_proof(metric=str(col_name), v_score=int(v_score), now_ts=now_ts, required=int(required_sigs))
            is_authorized = VERIFY_MULTISIG(payload=payload, proof=proof, authorized_node_secrets=authorized_nodes)

        if is_authorized:
            checks.append(_ok(f"Contract score authorized for {col_name}", f"mode={mode}"))
        else:
            checks.append(_warn(f"Contract score authorized for {col_name}", f"mode={mode} verification failed"))
            contract_failed += 1
            incentive_none += 1
            continue

        try:
            st = algorithm_3_value_delivery_execution(float(epsilon), float(v_score), float(alpha))
            checks.append(_ok(f"Alg3 runs on {col_name}", f"status={st}"))
            if st == "SETTLED":
                contract_settled += 1
            elif st == "PENDING_AUDIT":
                contract_pending += 1
            else:
                contract_failed += 1
        except Exception as e:
            checks.append(_warn(f"Alg3 runs on {col_name}", f"error: {e}"))

        try:
            i_status = algorithm_5_adaptive_value_incentives(
                tau=10.0,
                gamma=float(gamma),
                epsilon=float(epsilon),
                v_score=float(v_score),
            )
            checks.append(_ok(f"Alg5 runs on {col_name}", f"status={i_status}"))
            if i_status.startswith("REWARDED:"):
                incentive_full += 1
            elif i_status.startswith("REWARDED_PARTIAL:"):
                incentive_partial += 1
            else:
                incentive_none += 1
        except Exception as e:
            checks.append(_warn(f"Alg5 runs on {col_name}", f"error: {e}"))
            incentive_none += 1

    attempted = contract_settled + contract_pending + contract_failed
    if attempted <= 0:
        checks.append(_warn("Smart contract attainment rate", "no contract attempts were executed (Alg3 never ran)"))
        return checks

    attainment_rate = contract_settled / float(attempted)
    msg = (
        f"rate={attainment_rate:.2%} settled={contract_settled} "
        f"pending={contract_pending} failed={contract_failed} attempted={attempted} "
        f"min_required={float(contract_attainment_min):.2%}"
    )
    checks.append(
        _ok("Smart contract attainment rate", msg)
        if attainment_rate >= float(contract_attainment_min)
        else _warn("Smart contract attainment rate", msg)
    )

    incentive_attempted = incentive_full + incentive_partial + incentive_none
    if incentive_attempted > 0:
        incentive_rate = (incentive_full + incentive_partial) / float(incentive_attempted)
        msg2 = (
            f"rate={incentive_rate:.2%} full={incentive_full} partial={incentive_partial} "
            f"none={incentive_none} attempted={incentive_attempted} "
            f"min_required={float(incentive_attainment_min):.2%}"
        )
        checks.append(
            _ok("Smart contract incentive trigger rate", msg2)
            if incentive_rate >= float(incentive_attainment_min)
            else _warn("Smart contract incentive trigger rate", msg2)
        )

    if int(sampling_chunksize) > 0:
        checks.append(_ok("IoMT sampling batch mode", "pandas unavailable; used non-batch fallback"))

    return checks


def _load_kpi_inputs_no_pandas(
    *,
    sampling_csv: Path,
    sampling_limit: Optional[int],
    time_col: str,
    sampling_all_cols: bool,
    sampling_metric_col: str,
) -> Tuple[List[float], Dict[str, List[float]]]:
    header, rows_iter = _iter_csv_rows(sampling_csv)
    if not header:
        return ([], {})

    cols_lower = {str(c).strip().lower(): i for i, c in enumerate(header)}
    time_key = str(time_col).strip().lower()
    time_idx = cols_lower.get(time_key) if time_key in cols_lower else None

    id_cols = {"sample_id", "resample_pos"}
    if bool(sampling_all_cols):
        kpi_cols = [str(c).strip() for c in header if str(c).strip().lower() not in id_cols]
    else:
        mk = str(sampling_metric_col).strip().lower()
        if mk not in cols_lower:
            return ([], {})
        kpi_cols = [str(header[int(cols_lower[mk])]).strip()]

    col_indices: Dict[str, int] = {}
    for c in kpi_cols:
        ck = str(c).strip().lower()
        if ck in cols_lower:
            col_indices[str(c)] = int(cols_lower[ck])

    ts: List[float] = []
    values_by_col: Dict[str, List[float]] = {c: [] for c in col_indices}

    row_i = 0
    for cells in rows_iter:
        if sampling_limit is not None and int(sampling_limit) > 0 and row_i >= int(sampling_limit):
            break

        if len(cells) < len(header):
            cells = list(cells) + [""] * (len(header) - len(cells))

        # timestamp
        if time_idx is None:
            cur_ts = float(row_i)
        else:
            raw_ts = str(cells[int(time_idx)]).strip()
            try:
                cur_ts = float(raw_ts) if raw_ts != "" else float("nan")
            except Exception:
                cur_ts = float("nan")

        if isinstance(cur_ts, float) and math.isnan(cur_ts):
            # skip row if timestamp invalid; keeps alignment simple
            row_i += 1
            continue

        ts.append(float(cur_ts))

        for col_name, idx in col_indices.items():
            raw = str(cells[int(idx)]).strip()
            try:
                v = float(raw) if raw != "" else float("nan")
            except Exception:
                v = float("nan")
            values_by_col[col_name].append(v)

        row_i += 1

    # Filter out NaNs per-column later; keep NaNs in arrays for now.
    return (ts, values_by_col)


# -----------------------------
# Minimal executable reference
# -----------------------------


@dataclass(frozen=True)
class DataPoint:
    timestamp: float
    type: str
    value: float


@dataclass(frozen=True)
class RuleSet:
    threshold: float


@dataclass(frozen=True)
class Contract:
    contract_id: str
    active: bool = True


@dataclass(frozen=True)
class ConsentProof:
    subject_id: str
    signature: str


class UnauthorizedError(Exception):
    pass


@dataclass(frozen=True)
class MultiSigProof:
    """Minimal multisignature proof for score submission.

    In a real system, this would use asymmetric signatures. Here we use HMAC-like
    hashing with pre-shared secrets to keep the validator executable.
    """

    signatures: Dict[str, str]  # node_id -> signature hex
    required: int


@dataclass(frozen=True)
class OracleProof:
    oracle_id: str
    signature: str


def GET_TIMESTAMP() -> float:
    return datetime.now().timestamp()


def _stable_hash32(text: str) -> int:
    """Stable 32-bit hash for reproducible RNG seeding.

    Python's built-in hash() is randomized per process by default, which would
    otherwise make KPI simulations drift across runs even under a fixed --seed.
    """

    return int(zlib.crc32(str(text).encode("utf-8")) & 0xFFFFFFFF)


def _score_payload(*, metric: str, v_score: int, now_ts: float) -> bytes:
    # Canonical payload to prevent ambiguity.
    blob = json.dumps(
        {"metric": str(metric), "v_score": int(v_score), "now_ts": float(now_ts)},
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return blob


def _sign_payload(secret: str, payload: bytes) -> str:
    # Deterministic, keyed hash signature (minimal executable stand-in for real signatures).
    return hashlib.sha256(secret.encode("utf-8") + b"|" + payload).hexdigest()


def VERIFY_MULTISIG(
    *,
    payload: bytes,
    proof: MultiSigProof,
    authorized_node_secrets: Dict[str, str],
) -> bool:
    if proof.required <= 0:
        return False
    valid = 0
    for node_id, sig in proof.signatures.items():
        if node_id not in authorized_node_secrets:
            continue
        expected = _sign_payload(authorized_node_secrets[node_id], payload)
        if sig == expected:
            valid += 1
        if valid >= proof.required:
            return True
    return False


def VERIFY_ORACLE(
    *,
    payload: bytes,
    proof: OracleProof,
    oracle_secrets: Dict[str, str],
) -> bool:
    if proof.oracle_id not in oracle_secrets:
        return False
    expected = _sign_payload(oracle_secrets[proof.oracle_id], payload)
    return proof.signature == expected


# ---- Algorithm 1 dependencies ----

def GET_OBJECTIVES(stakeholder: str, stakeholder_objectives: Dict[str, List[str]]) -> List[str]:
    return list(stakeholder_objectives.get(stakeholder, []))


def GET_SERVICES(objective: str, objective_services: Dict[str, List[str]]) -> List[str]:
    return list(objective_services.get(objective, []))


def EXTRACT_CRITERIA(criteria_repo: Dict[str, List[str]], service: str) -> List[str]:
    return list(criteria_repo.get(service, []))


def algorithm_1_value_identification(
    stakeholder: str,
    criteria_repo: Dict[str, List[str]],
    stakeholder_objectives: Dict[str, List[str]],
    objective_services: Dict[str, List[str]],
) -> List[Tuple[str, str, str]]:
    """Executable reference for Algorithm 1.

    Returns a list of (objective, service, criterion) tuples.
    """
    objectives = GET_OBJECTIVES(stakeholder, stakeholder_objectives)
    vc: List[Tuple[str, str, str]] = []
    for obj in objectives:
        services = GET_SERVICES(obj, objective_services)
        for svc in services:
            criteria = EXTRACT_CRITERIA(criteria_repo, svc)
            for c in criteria:
                vc.append((obj, svc, c))
    # Deduplicate while preserving order
    seen = set()
    out: List[Tuple[str, str, str]] = []
    for t in vc:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


# ---- Algorithm 2 dependencies ----

def GET_WEIGHT(insight_type: str, weights: Dict[str, float]) -> float:
    return float(weights.get(insight_type, 1.0))


def EXTRACT_INSIGHT(d: DataPoint) -> DataPoint:
    # In this minimal reference, the data point is already an "insight".
    return d


def algorithm_2_value_quantification(
    v_info: Sequence[DataPoint],
    rules: RuleSet,
    sigma: int,
    *,
    now_ts: float,
    delta_t_max: float,
    weights: Dict[str, float],
) -> int:
    """Executable reference for Algorithm 2.

    Uses integer scaling to avoid precision loss: adds (w * value * sigma) for each qualifying insight.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if delta_t_max < 0:
        raise ValueError("delta_t_max must be non-negative")

    v_score = 0
    for d in v_info:
        if (now_ts - d.timestamp) > delta_t_max:
            continue
        insight = EXTRACT_INSIGHT(d)
        if insight.value >= rules.threshold:
            w = GET_WEIGHT(insight.type, weights)
            # Integer scaling: use rounding to nearest integer token.
            v_score += int(round(w * float(insight.value) * sigma))
    return int(v_score)


# ---- Algorithm 3 dependencies ----


def TRIGGER_SETTLEMENT() -> str:
    return "SETTLED"


def LOG_BLOCKCHAIN(status: str, message: str, log: List[Tuple[str, str]]) -> None:
    log.append((status, message))


def algorithm_3_value_delivery_execution(
    epsilon: float,
    v_score: float,
    alpha: float,
    *,
    log: Optional[List[Tuple[str, str]]] = None,
) -> str:
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1]")

    local_log: List[Tuple[str, str]] = [] if log is None else log

    if v_score >= epsilon:
        st = TRIGGER_SETTLEMENT()
        LOG_BLOCKCHAIN(st, "Success", local_log)
        return st
    if v_score >= (alpha * epsilon):
        return "PENDING_AUDIT"

    st = "FAILED"
    LOG_BLOCKCHAIN(st, "Under-performance", local_log)
    return st


# ---- Algorithm 4 dependencies ----


def VERIFY_CONSENT(proof: ConsentProof) -> bool:
    # Minimal: signature must be non-empty and include subject_id
    return bool(proof.signature) and proof.subject_id in proof.signature


def CHECK_CONTRACT(contract: Contract) -> bool:
    return bool(contract.contract_id) and contract.active


def ANONYMIZE(iomt_data: Dict[str, Any]) -> Dict[str, Any]:
    # Remove direct identifiers if present.
    redacted = dict(iomt_data)
    for key in ("name", "patient_name", "patient_id", "ssn", "email", "phone"):
        redacted.pop(key, None)
    return redacted


def COMPUTE_HASH(data: Dict[str, Any]) -> str:
    blob = json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def ENCRYPTED_CLOUD_STORE(data: Dict[str, Any]) -> str:
    # Minimal: return an opaque pointer; do NOT embed data.
    token = hashlib.sha256(os.urandom(16)).hexdigest()[:24]
    return f"cloudptr:{token}"


def RECORD_BLOCKCHAIN(h_root: str, pointer: str) -> str:
    # Minimal: deterministic transaction hash from inputs
    blob = (h_root + "|" + pointer).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def algorithm_4_value_capture_privacy_anchoring(
    iomt_data: Dict[str, Any],
    consent_proof: ConsentProof,
    contract: Contract,
) -> str:
    if not (VERIFY_CONSENT(consent_proof) and CHECK_CONTRACT(contract)):
        raise UnauthorizedError("ERR_UNAUTHORIZED")

    d_anon = ANONYMIZE(iomt_data)
    h_root = COMPUTE_HASH(d_anon)
    p_ext = ENCRYPTED_CLOUD_STORE(d_anon)
    tx = RECORD_BLOCKCHAIN(h_root, p_ext)
    return tx


# ---- Algorithm 5 dependencies ----


def CALC_BONUS_RATIO(v_cum: float) -> float:
    # Minimal monotonic bonus: saturates at 2.0
    return min(2.0, 1.0 + max(0.0, v_cum) / 100.0)


def EXECUTE_REWARD(amount: float) -> str:
    return f"REWARDED:{amount:.2f}"


def GENERATE_FEEDBACK(v_cum: float, gamma: float) -> str:
    gap = max(0.0, gamma - v_cum)
    return f"Performance below target by {gap:.2f}. Recommend follow-up and adjustments."


def NOTIFY_PROVIDER(advice: str) -> str:
    return f"NOTIFIED:{hashlib.sha1(advice.encode('utf-8')).hexdigest()[:10]}"


def algorithm_5_adaptive_value_incentives(
    tau: float,
    gamma: float,
    epsilon: float,
    v_score: float,
) -> str:
    if tau < 0:
        raise ValueError("tau must be non-negative")
    if gamma < 0:
        raise ValueError("gamma must be non-negative")

    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if gamma > epsilon and epsilon > 0:
        raise ValueError("gamma must be <= epsilon")

    v = float(v_score)

    # Full settlement band
    if v >= float(epsilon):
        beta = CALC_BONUS_RATIO(v)
        return EXECUTE_REWARD(tau * beta)

    # Partial incentive band: gamma <= V_score < epsilon
    if v >= float(gamma):
        if float(epsilon) == float(gamma):
            ratio = 1.0
        else:
            ratio = max(0.0, min(1.0, (v - float(gamma)) / (float(epsilon) - float(gamma))))
        # Partial reward scales with proximity to epsilon.
        amount = tau * ratio
        return f"REWARDED_PARTIAL:{amount:.2f}"

    advice = GENERATE_FEEDBACK(v, gamma)
    return NOTIFY_PROVIDER(advice)


# -----------------------------
# IoMT data connection interface
# -----------------------------


class IoMTDataConnector:
    """Minimal connector for IoMT sampling CSV.

    Converts a wide CSV (rows=samples, cols=metrics) into a stream of DataPoint.
    Default metric uses SpO2-style semantics from column `blood_oxygen`.
    """

    def __init__(
        self,
        csv_path: Path,
        *,
        time_col: str = "resample_pos",
        metric_col: str = "blood_oxygen",
        metric_type: str = "spo2",
    ) -> None:
        self.csv_path = csv_path
        self.time_col = time_col
        self.metric_col = metric_col
        self.metric_type = metric_type

    def load_datapoints(self, *, limit: Optional[int] = None) -> List[DataPoint]:
        try:
            import pandas as pd  # type: ignore
        except Exception as e:  # pragma: no cover
            raise SystemExit("Missing dependency pandas. Install: pip install pandas") from e

        if not self.csv_path.exists():
            raise FileNotFoundError(str(self.csv_path))

        usecols = [self.metric_col]
        if self.time_col:
            usecols.append(self.time_col)

        df = pd.read_csv(self.csv_path, usecols=lambda c: c in set(usecols))

        if self.time_col and self.time_col in df.columns:
            ts = pd.to_numeric(df[self.time_col], errors="coerce")
        else:
            ts = pd.Series(range(len(df)), dtype="float64")

        vals = pd.to_numeric(df[self.metric_col], errors="coerce")
        mask = (~ts.isna()) & (~vals.isna())
        ts = ts[mask]
        vals = vals[mask]

        if limit is not None:
            ts = ts.iloc[:limit]
            vals = vals.iloc[:limit]

        out: List[DataPoint] = []
        for t, v in zip(ts.to_numpy(), vals.to_numpy()):
            out.append(DataPoint(timestamp=float(t), type=self.metric_type, value=float(v)))
        return out


# -----------------------------
# Fog storage & processing framework
# -----------------------------


@dataclass(frozen=True)
class FogRecord:
    ts: float
    metric: str
    kind: str  # e.g., "raw", "processed", "v_score"
    payload: Dict[str, Any]


class FogStorage:
    """Minimal fog-side storage.

    - In-memory ring buffer for recent records
    - Optional append-only JSONL persistence for auditability
    """

    def __init__(
        self,
        *,
        max_items: int = 10_000,
        persist_path: Optional[Path] = None,
        cache_ttl_seconds: float = 0.0,
    ) -> None:
        self.max_items = int(max_items)
        self.persist_path = persist_path
        self.cache_ttl_seconds = float(cache_ttl_seconds)
        self._buf: List[FogRecord] = []

        # Metric-partitioned cache for processed time-series (fog-side feature store)
        self._metric_cache: Dict[str, Tuple[float, List[DataPoint]]] = {}

        if self.persist_path is not None:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, rec: FogRecord) -> None:
        self._buf.append(rec)
        if len(self._buf) > self.max_items:
            self._buf = self._buf[-self.max_items :]

        if self.persist_path is not None:
            line = json.dumps(
                {
                    "ts": rec.ts,
                    "metric": rec.metric,
                    "kind": rec.kind,
                    "payload": rec.payload,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            self.persist_path.open("a", encoding="utf-8").write(line + "\n")

    def snapshot(self) -> List[FogRecord]:
        return list(self._buf)

    def put_processed_series(self, *, metric: str, now_ts: float, datapoints: Sequence[DataPoint]) -> None:
        self._metric_cache[str(metric)] = (float(now_ts), list(datapoints))

    def get_processed_series(self, *, metric: str, now_ts: float) -> Optional[List[DataPoint]]:
        key = str(metric)
        if key not in self._metric_cache:
            return None
        cached_ts, cached = self._metric_cache[key]
        ttl = float(self.cache_ttl_seconds)
        if ttl > 0.0 and (float(now_ts) - float(cached_ts)) > ttl:
            return None
        return list(cached)


class FogProcessor:
    """Minimal fog-side processing.

    Currently supports optional moving-average smoothing by window size.
    """

    def __init__(
        self,
        *,
        smooth_window: int = 1,
        agg_window: int = 1,
        agg_func: str = "mean",
    ) -> None:
        self.smooth_window = max(1, int(smooth_window))
        self.agg_window = max(1, int(agg_window))
        self.agg_func = str(agg_func).strip().lower()

    def _aggregate(self, values: Sequence[float]) -> float:
        if not values:
            return float("nan")
        if self.agg_func == "min":
            return float(min(values))
        if self.agg_func == "max":
            return float(max(values))
        if self.agg_func == "median":
            s = sorted(float(v) for v in values)
            return float(s[len(s) // 2])
        # default mean
        return float(sum(float(v) for v in values) / float(len(values)))

    def process(self, datapoints: Sequence[DataPoint]) -> List[DataPoint]:
        # Stage 1: smoothing (moving average)
        smoothed: List[DataPoint] = []
        if self.smooth_window <= 1:
            smoothed = list(datapoints)
        else:
            window: List[float] = []
            for d in datapoints:
                window.append(float(d.value))
                if len(window) > self.smooth_window:
                    window.pop(0)
                avg = sum(window) / float(len(window))
                smoothed.append(DataPoint(timestamp=float(d.timestamp), type=d.type, value=float(avg)))

        # Stage 2: aggregation window (feature reduction)
        if self.agg_window <= 1:
            return smoothed

        out: List[DataPoint] = []
        bucket: List[float] = []
        bucket_ts: Optional[float] = None
        for d in smoothed:
            if bucket_ts is None:
                bucket_ts = float(d.timestamp)
            bucket.append(float(d.value))
            if len(bucket) >= self.agg_window:
                v = self._aggregate(bucket)
                out.append(DataPoint(timestamp=float(bucket_ts), type=d.type, value=float(v)))
                bucket = []
                bucket_ts = None
        if bucket:
            v = self._aggregate(bucket)
            out.append(DataPoint(timestamp=float(bucket_ts if bucket_ts is not None else smoothed[-1].timestamp), type=smoothed[-1].type, value=float(v)))
        return out


class FogNode:
    def __init__(
        self,
        *,
        node_id: str,
        secret: str,
        storage: FogStorage,
        processor: FogProcessor,
    ) -> None:
        self.node_id = str(node_id)
        self.secret = str(secret)
        self.storage = storage
        self.processor = processor

    def process(self, metric: str, datapoints: Sequence[DataPoint], *, now_ts: float) -> List[DataPoint]:
        cached = self.storage.get_processed_series(metric=str(metric), now_ts=float(now_ts))
        if cached is not None:
            self.storage.append(
                FogRecord(
                    ts=float(now_ts),
                    metric=str(metric),
                    kind="cache_hit",
                    payload={"count": int(len(cached)), "ttl": float(self.storage.cache_ttl_seconds)},
                )
            )
            return cached

        self.storage.append(
            FogRecord(
                ts=float(now_ts),
                metric=str(metric),
                kind="raw",
                payload={"count": int(len(datapoints))},
            )
        )
        processed = self.processor.process(datapoints)
        self.storage.append(
            FogRecord(
                ts=float(now_ts),
                metric=str(metric),
                kind="processed",
                payload={
                    "count": int(len(processed)),
                    "smooth_window": int(self.processor.smooth_window),
                    "agg_window": int(self.processor.agg_window),
                    "agg_func": str(self.processor.agg_func),
                },
            )
        )

        self.storage.put_processed_series(metric=str(metric), now_ts=float(now_ts), datapoints=processed)
        return processed

    def compute_v_score(
        self,
        *,
        metric: str,
        datapoints: Sequence[DataPoint],
        rules: RuleSet,
        sigma: int,
        now_ts: float,
        delta_t_max: float,
        weights: Dict[str, float],
    ) -> int:
        v = algorithm_2_value_quantification(
            datapoints,
            rules,
            int(sigma),
            now_ts=float(now_ts),
            delta_t_max=float(delta_t_max),
            weights=weights,
        )
        self.storage.append(
            FogRecord(
                ts=float(now_ts),
                metric=str(metric),
                kind="v_score",
                payload={"v_score": int(v), "sigma": int(sigma), "threshold": float(rules.threshold)},
            )
        )
        return int(v)

    def sign_score(self, *, metric: str, v_score: int, now_ts: float) -> str:
        payload = _score_payload(metric=str(metric), v_score=int(v_score), now_ts=float(now_ts))
        return _sign_payload(self.secret, payload)


class FogCluster:
    def __init__(self, nodes: Sequence[FogNode]) -> None:
        self.nodes = list(nodes)

    @staticmethod
    def from_secrets(
        *,
        node_secrets: Dict[str, str],
        smooth_window: int,
        agg_window: int,
        agg_func: str,
        store_dir: Optional[Path],
        store_max_items: int,
        cache_ttl_seconds: float,
    ) -> "FogCluster":
        nodes: List[FogNode] = []
        for node_id, secret in node_secrets.items():
            persist_path = None
            if store_dir is not None:
                persist_path = store_dir / f"fog_{node_id}.jsonl"
            storage = FogStorage(
                max_items=int(store_max_items),
                persist_path=persist_path,
                cache_ttl_seconds=float(cache_ttl_seconds),
            )
            processor = FogProcessor(
                smooth_window=int(smooth_window),
                agg_window=int(agg_window),
                agg_func=str(agg_func),
            )
            nodes.append(FogNode(node_id=node_id, secret=secret, storage=storage, processor=processor))
        return FogCluster(nodes)

    def process(
        self,
        metric: str,
        datapoints: Sequence[DataPoint],
        *,
        now_ts: float,
        offline_rate: float = 0.0,
        byzantine_rate: float = 0.0,
        byzantine_noise_std: float = 0.0,
        rng_seed: int = 0,
    ) -> Dict[str, List[DataPoint]]:
        out: Dict[str, List[DataPoint]] = {}

        rr = random.Random(int(rng_seed) ^ _stable_hash32(str(metric)))
        for n in self.nodes:
            if float(offline_rate) > 0.0 and rr.random() < float(offline_rate):
                continue
            processed = n.process(metric, datapoints, now_ts=float(now_ts))

            # Byzantine simulation: perturb processed values.
            if float(byzantine_rate) > 0.0 and rr.random() < float(byzantine_rate):
                noisy: List[DataPoint] = []
                for d in processed:
                    noise = 0.0
                    if float(byzantine_noise_std) > 0.0:
                        noise = rr.gauss(0.0, float(byzantine_noise_std))
                    noisy.append(DataPoint(timestamp=d.timestamp, type=d.type, value=float(d.value) + float(noise)))
                processed = noisy
            out[n.node_id] = processed
        return out

    def compute_scores(self, *, metric: str, processed: Dict[str, List[DataPoint]], rules: RuleSet, sigma: int, now_ts: float, delta_t_max: float, weights: Dict[str, float]) -> Dict[str, int]:
        scores: Dict[str, int] = {}
        for n in self.nodes:
            scores[n.node_id] = n.compute_v_score(
                metric=str(metric),
                datapoints=processed[n.node_id],
                rules=rules,
                sigma=int(sigma),
                now_ts=float(now_ts),
                delta_t_max=float(delta_t_max),
                weights=weights,
            )
        return scores

    def multisig_proof(self, *, metric: str, v_score: int, now_ts: float, required: int) -> MultiSigProof:
        req = max(1, min(int(required), len(self.nodes)))
        sigs: Dict[str, str] = {}
        for n in self.nodes[:req]:
            sigs[n.node_id] = n.sign_score(metric=str(metric), v_score=int(v_score), now_ts=float(now_ts))
        return MultiSigProof(signatures=sigs, required=req)


# -----------------------------
# System KPI computation model
# -----------------------------


@dataclass(frozen=True)
class SystemKPIInputs:
    # Workload / control variables
    delta_percent: float
    batch_size: int
    confirm_prob: float

    # Timing model (ms) for network + confirmation
    t_tx_ms: float
    t_tx_jitter_ms: float
    t_conf_ms: float
    t_conf_jitter_ms: float

    # Cloud-only baseline processing (ms) per submitted batch
    t_cloud_ms: float
    t_cloud_jitter_ms: float

    # Unit cost model
    c_device_unit: float
    c_fog_unit: float
    c_cloud_unit: float
    c_sc_unit: float
    c_conf_unit: float

    # Bytes per data point for throughput unit conversion.
    # Th_sys is reported in MB/s using this factor.
    bytes_per_point: float = 8.0


def _chunk(seq: Sequence[DataPoint], batch_size: int) -> List[List[DataPoint]]:
    bs = max(1, int(batch_size))
    out: List[List[DataPoint]] = []
    for i in range(0, len(seq), bs):
        out.append(list(seq[i : i + bs]))
    return out


def _relative_change_percent(curr: float, prev: float) -> float:
    denom = abs(float(prev))
    if denom <= 1e-12:
        return 0.0 if abs(float(curr)) <= 1e-12 else 100.0
    return (abs(float(curr) - float(prev)) / denom) * 100.0


def _jittered_ms(rng: random.Random, base_ms: float, jitter_ms: float) -> float:
    j = float(jitter_ms)
    if j <= 0.0:
        return float(base_ms)
    return max(0.0, float(base_ms) + rng.uniform(-j, j))


def compute_system_kpis_for_metric(
    *,
    metric: str,
    datapoints: Sequence[DataPoint],
    fog: FogCluster,
    fog_offline_rate: float,
    fog_byzantine_rate: float,
    fog_byzantine_noise_std: float,
    fog_consensus_max_rel_range: float,
    fog_enforce_consensus: bool,
    seed: int,
    # Smart contract model params
    proof_mode: str,
    required_sigs: int,
    epsilon: float,
    alpha: float,
    gamma: float,
    sigma: int,
    threshold: float,
    threshold_auto: bool,
    # KPI model
    kpi: SystemKPIInputs,
    cloud_only_baseline: bool = False,
) -> Dict[str, Any]:
    """Compute key system validation KPIs for a single metric stream.

    Implements the paper-style fog forwarding filter using delta on batch mean,
    then simulates smart contract submissions and confirmations.
    """

    dps = sorted(list(datapoints), key=lambda d: float(d.timestamp))
    n_points = len(dps)
    batches = _chunk(dps, int(kpi.batch_size))
    f_tx = len(batches)

    # Threshold for Alg2
    try:
        import statistics

        values = [float(d.value) for d in dps if not math.isnan(float(d.value))]
        median_v = float(statistics.median(values)) if values else float(threshold)
    except Exception:
        median_v = float(threshold)

    thr = float(median_v) if bool(threshold_auto) else float(threshold)
    rules = RuleSet(threshold=thr)
    weights = {str(metric).strip().lower(): 1.0}

    # Oracle secrets kept consistent with sampling checks
    oracle_secrets = {"oracle_1": "oracle_secret"}
    authorized_node_secrets = {n.node_id: n.secret for n in fog.nodes}

    # Counters
    f_sc = 0
    f_submitted = 0
    f_confirmed = 0
    contract_settled = 0
    contract_pending = 0
    contract_failed = 0

    incentive_full = 0
    incentive_partial = 0
    incentive_none = 0

    # Timings (ms)
    total_t_tx_ms = 0.0
    total_t_fog_ms = 0.0
    total_t_sc_ms = 0.0
    total_t_conf_ms = 0.0

    rng = random.Random(int(seed) ^ _stable_hash32(str(metric)))
    prev_mu: Optional[float] = None

    for batch in batches:
        if not batch:
            continue

        mu_t = sum(float(d.value) for d in batch) / float(len(batch))
        if prev_mu is None:
            delta_t = 100.0
        else:
            delta_t = _relative_change_percent(mu_t, prev_mu)
        prev_mu = mu_t

        trigger = delta_t > float(kpi.delta_percent)

        # Fog processing time includes preprocess + scoring.
        t0 = time.perf_counter()
        processed_by_node = fog.process(
            metric=str(metric),
            datapoints=batch,
            now_ts=float(batch[-1].timestamp),
            offline_rate=float(fog_offline_rate),
            byzantine_rate=float(fog_byzantine_rate),
            byzantine_noise_std=float(fog_byzantine_noise_std),
            rng_seed=int(seed),
        )
        if len(processed_by_node) == 0:
            total_t_fog_ms += (time.perf_counter() - t0) * 1000.0
            # No fog nodes available; cannot submit.
            continue

        scores = fog.compute_scores(
            metric=str(metric),
            processed=processed_by_node,
            rules=rules,
            sigma=int(sigma),
            now_ts=float(batch[-1].timestamp),
            delta_t_max=float("inf"),
            weights=weights,
        )
        score_values = sorted(int(v) for v in scores.values())
        v_score = int(score_values[len(score_values) // 2])
        if len(score_values) >= 2:
            rel_range = (max(score_values) - min(score_values)) / float(max(1, abs(v_score)))
        else:
            rel_range = 0.0
        total_t_fog_ms += (time.perf_counter() - t0) * 1000.0

        if float(fog_consensus_max_rel_range) > 0.0 and rel_range > float(fog_consensus_max_rel_range) and bool(fog_enforce_consensus):
            # Consensus violation prevents submission.
            contract_failed += 1
            incentive_none += 1
            continue

        if not trigger:
            continue

        f_sc += 1

        # Contract authorization
        now_ts = float(batch[-1].timestamp)
        payload = _score_payload(metric=str(metric), v_score=int(v_score), now_ts=now_ts)
        mode = str(proof_mode).strip().lower()
        if mode == "none":
            is_authorized = True
        elif mode == "oracle":
            oracle_id = "oracle_1"
            proof = OracleProof(oracle_id=oracle_id, signature=_sign_payload(oracle_secrets[oracle_id], payload))
            is_authorized = VERIFY_ORACLE(payload=payload, proof=proof, oracle_secrets=oracle_secrets)
        else:
            proof = fog.multisig_proof(metric=str(metric), v_score=int(v_score), now_ts=now_ts, required=int(required_sigs))
            is_authorized = VERIFY_MULTISIG(payload=payload, proof=proof, authorized_node_secrets=authorized_node_secrets)

        if not is_authorized:
            contract_failed += 1
            incentive_none += 1
            continue

        # Network transmission (simulated)
        total_t_tx_ms += _jittered_ms(rng, float(kpi.t_tx_ms), float(kpi.t_tx_jitter_ms))

        # Smart contract execution (measured)
        t1 = time.perf_counter()
        st = algorithm_3_value_delivery_execution(float(epsilon), float(v_score), float(alpha))
        if st == "SETTLED":
            contract_settled += 1
        elif st == "PENDING_AUDIT":
            contract_pending += 1
        else:
            contract_failed += 1

        i_status = algorithm_5_adaptive_value_incentives(
            tau=10.0,
            gamma=float(gamma),
            epsilon=float(epsilon),
            v_score=float(v_score),
        )
        if i_status.startswith("REWARDED:"):
            incentive_full += 1
        elif i_status.startswith("REWARDED_PARTIAL:"):
            incentive_partial += 1
        else:
            incentive_none += 1
        total_t_sc_ms += (time.perf_counter() - t1) * 1000.0

        # Submit + confirm (simulated)
        f_submitted += 1
        ok = rng.random() < float(kpi.confirm_prob)
        total_t_conf_ms += _jittered_ms(rng, float(kpi.t_conf_ms), float(kpi.t_conf_jitter_ms))
        if ok:
            f_confirmed += 1

    # Aggregate KPIs using definitions in verification script markdown.
    l_total_ms = total_t_tx_ms + total_t_fog_ms + total_t_sc_ms + total_t_conf_ms
    denom_time_s = max(1e-12, l_total_ms / 1000.0)

    l_e2e_mean_ms = (l_total_ms / float(f_submitted)) if f_submitted > 0 else 0.0
    th_sys_points_per_s = (float(n_points) / denom_time_s) if n_points > 0 else 0.0

    bpp = max(0.0, float(kpi.bytes_per_point))
    th_sys_mb_per_s = ((float(n_points) * bpp) / 1_000_000.0) / denom_time_s

    r_reliability = (float(f_confirmed) / float(f_submitted)) if f_submitted > 0 else 1.0
    e_eff = (1.0 - (float(f_sc) / float(f_tx))) if f_tx > 0 else 0.0

    c_devices = float(f_tx) * float(kpi.c_device_unit)
    c_fog = float(f_tx) * float(kpi.c_fog_unit)
    c_sc = float(f_sc) * float(kpi.c_sc_unit)
    c_conf = float(f_confirmed) * float(kpi.c_conf_unit)
    tc_total = c_devices + c_fog + c_sc + c_conf
    tc_unit = (tc_total / float(n_points)) if n_points > 0 else 0.0

    denom_p_s = max(1e-12, (total_t_sc_ms + total_t_conf_ms) / 1000.0)
    p_perf = float(f_confirmed) / denom_p_s

    denom_ro = float(n_points + f_sc)
    ro = ((c_fog + c_sc) / denom_ro) if denom_ro > 0 else 0.0

    # Symbols-aligned aliases (see Symbols definitions.csv)
    eta_percent = float(e_eff) * 100.0
    r_percent = float(r_reliability) * 100.0
    ttrans_mean_ms = (float(total_t_tx_ms) / float(f_submitted)) if f_submitted > 0 else 0.0
    tcons_mean_ms = (float(total_t_conf_ms) / float(f_submitted)) if f_submitted > 0 else 0.0

    return {
        "metric": str(metric),
        "params": {
            "delta_percent": float(kpi.delta_percent),
            "batch_size": int(kpi.batch_size),
            "confirm_prob": float(kpi.confirm_prob),
            "threshold": float(thr),
            "sigma": int(sigma),
            "epsilon": float(epsilon),
            "alpha": float(alpha),
            "gamma": float(gamma),
            "bytes_per_point": float(kpi.bytes_per_point),
        },
        "counts": {
            "N": int(n_points),
            "B": int(kpi.batch_size),
            "F_tx": int(f_tx),
            "F_sc": int(f_sc),
            "F_submitted": int(f_submitted),
            "F_confirmed": int(f_confirmed),
            "k_submitted": int(f_submitted),
        },
        "timing_ms": {
            "T_tx_total": round(float(total_t_tx_ms), 4),
            "T_fog_total": round(float(total_t_fog_ms), 4),
            "T_sc_total": round(float(total_t_sc_ms), 4),
            "T_conf_total": round(float(total_t_conf_ms), 4),
            "L_e2e_total": round(float(l_total_ms), 4),
            "L_e2e_mean": round(float(l_e2e_mean_ms), 6),
            "T_trans_mean": round(float(ttrans_mean_ms), 6),
            "T_cons_mean": round(float(tcons_mean_ms), 6),
        },
        "cost": {
            "C_devices": round(float(c_devices), 6),
            "C_fog": round(float(c_fog), 6),
            "C_sc": round(float(c_sc), 6),
            "C_conf": round(float(c_conf), 6),
            "T_c": round(float(tc_total), 6),
            "Tc_unit": round(float(tc_unit), 9),
        },
        "metrics": {
            "Th_sys_points_per_s": round(float(th_sys_points_per_s), 6),
            "Th_sys_MB_per_s": round(float(th_sys_mb_per_s), 6),
            "Thsys_points_per_s": round(float(th_sys_points_per_s), 6),
            "Thsys_MB_per_s": round(float(th_sys_mb_per_s), 6),
            "Reliability_r": round(float(r_reliability), 6),
            "Reliability_R_percent": round(float(r_percent), 6),
            "Efficiency_e": round(float(e_eff), 6),
            "Efficiency_eta_percent": round(float(eta_percent), 6),
            "Performance_p": round(float(p_perf), 6),
            "Resource_Optimization_Ro": round(float(ro), 9),
            "Resource_Optimization_Z": round(float(ro), 9),
        },
        "symbols": {
            "Le2e": round(float(l_total_ms), 4),
            "Le2e_ms_total": round(float(l_total_ms), 4),
            "Thsys": round(float(th_sys_mb_per_s), 6),
            "Thsys_points_per_s": round(float(th_sys_points_per_s), 6),
            "Thsys_MB_per_s": round(float(th_sys_mb_per_s), 6),
            "P": round(float(p_perf), 6),
            "Tc": round(float(tc_total), 6),
            "Z": round(float(ro), 9),
            "δ": float(kpi.delta_percent),
            "delta_percent": float(kpi.delta_percent),
            "η": round(float(eta_percent), 6),
            "eta_percent": round(float(eta_percent), 6),
            "R": round(float(r_percent), 6),
            "R_percent": round(float(r_percent), 6),
            "N": int(n_points),
            "B": int(kpi.batch_size),
            "Ttrans_ms_mean": round(float(ttrans_mean_ms), 6),
            "Tcons_ms_mean": round(float(tcons_mean_ms), 6),
            "ksub": int(f_submitted),
        },
        "contract": {
            "settled": int(contract_settled),
            "pending_audit": int(contract_pending),
            "failed": int(contract_failed),
            "attainment_rate": round((float(contract_settled) / float(max(1, contract_settled + contract_pending + contract_failed))), 6),
            "incentive_full": int(incentive_full),
            "incentive_partial": int(incentive_partial),
            "incentive_none": int(incentive_none),
        },
        "baseline_cloud": (
            _compute_cloud_only_baseline_for_metric(
                metric=str(metric),
                datapoints=dps,
                seed=int(seed),
                proof_mode=str(proof_mode),
                required_sigs=int(required_sigs),
                epsilon=float(epsilon),
                alpha=float(alpha),
                gamma=float(gamma),
                sigma=int(sigma),
                threshold=float(threshold),
                threshold_auto=bool(threshold_auto),
                kpi=kpi,
            )
            if bool(cloud_only_baseline)
            else None
        ),
    }


def _compute_cloud_only_baseline_for_metric(
    *,
    metric: str,
    datapoints: Sequence[DataPoint],
    seed: int,
    # Smart contract model params
    proof_mode: str,
    required_sigs: int,
    epsilon: float,
    alpha: float,
    gamma: float,
    sigma: int,
    threshold: float,
    threshold_auto: bool,
    # KPI model
    kpi: SystemKPIInputs,
) -> Dict[str, Any]:
    """Compute a cloud-only reference (no fog filtering; all batches submitted).

    Baseline assumptions:
    - Each batch is transmitted to the cloud and scored centrally (Alg2).
    - Threshold triggering is disabled: all batches submit to the contract.
    - Efficiency becomes e=1-F_sc/F_tx=0 by definition (F_sc=F_tx).
    - Timing includes tx + cloud processing + contract execution + confirmation.
    """

    dps = sorted(list(datapoints), key=lambda d: float(d.timestamp))
    n_points = len(dps)
    batches = _chunk(dps, int(kpi.batch_size))
    f_tx = len(batches)

    # Threshold for Alg2
    try:
        import statistics

        values = [float(d.value) for d in dps if not math.isnan(float(d.value))]
        median_v = float(statistics.median(values)) if values else float(threshold)
    except Exception:
        median_v = float(threshold)
    thr = float(median_v) if bool(threshold_auto) else float(threshold)
    rules = RuleSet(threshold=thr)
    weights = {str(metric).strip().lower(): 1.0}

    oracle_secrets = {"oracle_1": "oracle_secret"}
    authorized_node_secrets = {"cloud": "cloud_secret"}

    f_sc = int(f_tx)
    f_submitted = 0
    f_confirmed = 0
    contract_settled = 0
    contract_pending = 0
    contract_failed = 0

    total_t_tx_ms = 0.0
    total_t_cloud_ms = 0.0
    total_t_sc_ms = 0.0
    total_t_conf_ms = 0.0

    rng = random.Random(int(seed) ^ _stable_hash32(str(metric)) ^ 0xC10A0D0)  # deterministic offset

    for batch in batches:
        if not batch:
            continue

        now_ts = float(batch[-1].timestamp)

        # Transmission (simulated): always happens in cloud-only baseline.
        total_t_tx_ms += _jittered_ms(rng, float(kpi.t_tx_ms), float(kpi.t_tx_jitter_ms))

        # Cloud processing: (Alg2 scoring) + optional constant delay.
        t0 = time.perf_counter()
        v_score = algorithm_2_value_quantification(
            batch,
            rules,
            int(sigma),
            now_ts=now_ts,
            delta_t_max=float("inf"),
            weights=weights,
        )
        total_t_cloud_ms += (time.perf_counter() - t0) * 1000.0
        total_t_cloud_ms += _jittered_ms(rng, float(kpi.t_cloud_ms), float(kpi.t_cloud_jitter_ms))

        # Contract authorization
        payload = _score_payload(metric=str(metric), v_score=int(v_score), now_ts=now_ts)
        mode = str(proof_mode).strip().lower()
        if mode == "none":
            is_authorized = True
        elif mode == "oracle":
            oracle_id = "oracle_1"
            proof = OracleProof(oracle_id=oracle_id, signature=_sign_payload(oracle_secrets[oracle_id], payload))
            is_authorized = VERIFY_ORACLE(payload=payload, proof=proof, oracle_secrets=oracle_secrets)
        else:
            # Minimal baseline multisig: single "cloud" signer.
            proof = MultiSigProof(
                signatures={"cloud": _sign_payload(authorized_node_secrets["cloud"], payload)},
                required=1,
            )
            is_authorized = VERIFY_MULTISIG(payload=payload, proof=proof, authorized_node_secrets=authorized_node_secrets)

        if not is_authorized:
            contract_failed += 1
            continue

        # Smart contract execution
        t1 = time.perf_counter()
        st = algorithm_3_value_delivery_execution(float(epsilon), float(v_score), float(alpha))
        if st == "SETTLED":
            contract_settled += 1
        elif st == "PENDING_AUDIT":
            contract_pending += 1
        else:
            contract_failed += 1

        # Incentive (executed but not used in core A–E; kept for completeness)
        _ = algorithm_5_adaptive_value_incentives(
            tau=10.0,
            gamma=float(gamma),
            epsilon=float(epsilon),
            v_score=float(v_score),
        )
        total_t_sc_ms += (time.perf_counter() - t1) * 1000.0

        # Submit + confirm (simulated)
        f_submitted += 1
        ok = rng.random() < float(kpi.confirm_prob)
        total_t_conf_ms += _jittered_ms(rng, float(kpi.t_conf_ms), float(kpi.t_conf_jitter_ms))
        if ok:
            f_confirmed += 1

    l_total_ms = total_t_tx_ms + total_t_cloud_ms + total_t_sc_ms + total_t_conf_ms
    denom_time_s = max(1e-12, l_total_ms / 1000.0)

    l_e2e_mean_ms = (l_total_ms / float(f_submitted)) if f_submitted > 0 else 0.0
    th_sys_points_per_s = (float(n_points) / denom_time_s) if n_points > 0 else 0.0

    bpp = max(0.0, float(kpi.bytes_per_point))
    th_sys_mb_per_s = ((float(n_points) * bpp) / 1_000_000.0) / denom_time_s

    r_reliability = (float(f_confirmed) / float(f_submitted)) if f_submitted > 0 else 1.0
    e_eff = (1.0 - (float(f_sc) / float(f_tx))) if f_tx > 0 else 0.0

    c_devices = float(f_tx) * float(kpi.c_device_unit)
    c_cloud = float(f_tx) * float(kpi.c_cloud_unit)
    c_sc = float(f_sc) * float(kpi.c_sc_unit)
    c_conf = float(f_confirmed) * float(kpi.c_conf_unit)
    tc_total = c_devices + c_cloud + c_sc + c_conf
    tc_unit = (tc_total / float(n_points)) if n_points > 0 else 0.0

    denom_p_s = max(1e-12, (total_t_sc_ms + total_t_conf_ms) / 1000.0)
    p_perf = float(f_confirmed) / denom_p_s

    denom_ro = float(n_points + f_sc)
    ro = ((c_cloud + c_sc) / denom_ro) if denom_ro > 0 else 0.0

    eta_percent = float(e_eff) * 100.0
    r_percent = float(r_reliability) * 100.0
    ttrans_mean_ms = (float(total_t_tx_ms) / float(f_submitted)) if f_submitted > 0 else 0.0
    tcons_mean_ms = (float(total_t_conf_ms) / float(f_submitted)) if f_submitted > 0 else 0.0

    return {
        "params": {
            "batch_size": int(kpi.batch_size),
            "confirm_prob": float(kpi.confirm_prob),
            "threshold": float(thr),
            "sigma": int(sigma),
            "epsilon": float(epsilon),
            "alpha": float(alpha),
            "gamma": float(gamma),
            "bytes_per_point": float(kpi.bytes_per_point),
        },
        "counts": {
            "N": int(n_points),
            "B": int(kpi.batch_size),
            "F_tx": int(f_tx),
            "F_sc": int(f_sc),
            "F_submitted": int(f_submitted),
            "F_confirmed": int(f_confirmed),
            "k_submitted": int(f_submitted),
        },
        "timing_ms": {
            "T_tx_total": round(float(total_t_tx_ms), 4),
            "T_cloud_total": round(float(total_t_cloud_ms), 4),
            "T_sc_total": round(float(total_t_sc_ms), 4),
            "T_conf_total": round(float(total_t_conf_ms), 4),
            "L_e2e_total": round(float(l_total_ms), 4),
            "L_e2e_mean": round(float(l_e2e_mean_ms), 6),
            "T_trans_mean": round(float(ttrans_mean_ms), 6),
            "T_cons_mean": round(float(tcons_mean_ms), 6),
        },
        "cost": {
            "C_devices": round(float(c_devices), 6),
            "C_cloud": round(float(c_cloud), 6),
            "C_sc": round(float(c_sc), 6),
            "C_conf": round(float(c_conf), 6),
            "T_c": round(float(tc_total), 6),
            "Tc_unit": round(float(tc_unit), 9),
        },
        "metrics": {
            "Th_sys_points_per_s": round(float(th_sys_points_per_s), 6),
            "Th_sys_MB_per_s": round(float(th_sys_mb_per_s), 6),
            "Thsys_points_per_s": round(float(th_sys_points_per_s), 6),
            "Thsys_MB_per_s": round(float(th_sys_mb_per_s), 6),
            "Reliability_r": round(float(r_reliability), 6),
            "Reliability_R_percent": round(float(r_percent), 6),
            "Efficiency_e": round(float(e_eff), 6),
            "Efficiency_eta_percent": round(float(eta_percent), 6),
            "Performance_p": round(float(p_perf), 6),
            "Resource_Optimization_Ro": round(float(ro), 9),
            "Resource_Optimization_Z": round(float(ro), 9),
        },
        "symbols": {
            "Le2e": round(float(l_total_ms), 4),
            "Le2e_ms_total": round(float(l_total_ms), 4),
            "Thsys": round(float(th_sys_mb_per_s), 6),
            "Thsys_points_per_s": round(float(th_sys_points_per_s), 6),
            "Thsys_MB_per_s": round(float(th_sys_mb_per_s), 6),
            "P": round(float(p_perf), 6),
            "Tc": round(float(tc_total), 6),
            "Z": round(float(ro), 9),
            "δ": None,
            "delta_percent": None,
            "η": round(float(eta_percent), 6),
            "eta_percent": round(float(eta_percent), 6),
            "R": round(float(r_percent), 6),
            "R_percent": round(float(r_percent), 6),
            "N": int(n_points),
            "B": int(kpi.batch_size),
            "Ttrans_ms_mean": round(float(ttrans_mean_ms), 6),
            "Tcons_ms_mean": round(float(tcons_mean_ms), 6),
            "ksub": int(f_submitted),
        },
        "contract": {
            "settled": int(contract_settled),
            "pending_audit": int(contract_pending),
            "failed": int(contract_failed),
            "attainment_rate": round((float(contract_settled) / float(max(1, contract_settled + contract_pending + contract_failed))), 6),
        },
    }


def _write_kpi_outputs(*, results: List[Dict[str, Any]], out_json: Path, out_csv: Optional[Path]) -> None:
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if out_csv is None:
        return

    # Flatten a minimal subset into CSV.
    fieldnames = [
        "metric",
        "delta_percent",
        "batch_size",
        "confirm_prob",
        "bytes_per_point",
        "N",
        "B",
        "F_tx",
        "F_sc",
        "F_submitted",
        "F_confirmed",
        "k_submitted",
        "L_e2e_mean_ms",
        "Le2e_ms_total",
        "Th_sys_points_per_s",
        "Th_sys_MB_per_s",
        "Thsys_points_per_s",
        "Thsys_MB_per_s",
        "Reliability_r",
        "Reliability_R_percent",
        "Efficiency_e",
        "Efficiency_eta_percent",
        "T_c",
        "Tc_unit",
        "Performance_p",
        "Resource_Optimization_Ro",
        "Resource_Optimization_Z",
        "contract_attainment_rate",
        "Ttrans_ms_mean",
        "Tcons_ms_mean",
        "ksub",
        "cloud_L_e2e_mean_ms",
        "cloud_Le2e_ms_total",
        "cloud_Th_sys_points_per_s",
        "cloud_Th_sys_MB_per_s",
        "cloud_Thsys_points_per_s",
        "cloud_Thsys_MB_per_s",
        "cloud_Reliability_r",
        "cloud_Reliability_R_percent",
        "cloud_Efficiency_e",
        "cloud_Efficiency_eta_percent",
        "cloud_T_c",
        "cloud_Resource_Optimization_Ro",
        "cloud_Resource_Optimization_Z",
    ]
    rows: List[Dict[str, Any]] = []
    for r in results:
        bc = r.get("baseline_cloud") if isinstance(r, dict) else None
        bc_timing = (bc.get("timing_ms") if isinstance(bc, dict) else {})
        bc_metrics = (bc.get("metrics") if isinstance(bc, dict) else {})
        bc_cost = (bc.get("cost") if isinstance(bc, dict) else {})
        sym = (r.get("symbols") if isinstance(r, dict) else {}) or {}
        bc_sym = (bc.get("symbols") if isinstance(bc, dict) else {}) if isinstance(bc, dict) else {}
        rows.append(
            {
                "metric": r.get("metric"),
                "delta_percent": r.get("params", {}).get("delta_percent"),
                "batch_size": r.get("params", {}).get("batch_size"),
                "confirm_prob": r.get("params", {}).get("confirm_prob"),
                "bytes_per_point": r.get("params", {}).get("bytes_per_point"),
                "N": r.get("counts", {}).get("N"),
                "B": r.get("counts", {}).get("B"),
                "F_tx": r.get("counts", {}).get("F_tx"),
                "F_sc": r.get("counts", {}).get("F_sc"),
                "F_submitted": r.get("counts", {}).get("F_submitted"),
                "F_confirmed": r.get("counts", {}).get("F_confirmed"),
                "k_submitted": r.get("counts", {}).get("k_submitted"),
                "L_e2e_mean_ms": r.get("timing_ms", {}).get("L_e2e_mean"),
                "Le2e_ms_total": sym.get("Le2e_ms_total"),
                "Th_sys_points_per_s": r.get("metrics", {}).get("Th_sys_points_per_s"),
                "Th_sys_MB_per_s": r.get("metrics", {}).get("Th_sys_MB_per_s"),
                "Thsys_points_per_s": r.get("metrics", {}).get("Thsys_points_per_s"),
                "Thsys_MB_per_s": r.get("metrics", {}).get("Thsys_MB_per_s"),
                "Reliability_r": r.get("metrics", {}).get("Reliability_r"),
                "Reliability_R_percent": r.get("metrics", {}).get("Reliability_R_percent"),
                "Efficiency_e": r.get("metrics", {}).get("Efficiency_e"),
                "Efficiency_eta_percent": r.get("metrics", {}).get("Efficiency_eta_percent"),
                "T_c": r.get("cost", {}).get("T_c"),
                "Tc_unit": r.get("cost", {}).get("Tc_unit"),
                "Performance_p": r.get("metrics", {}).get("Performance_p"),
                "Resource_Optimization_Ro": r.get("metrics", {}).get("Resource_Optimization_Ro"),
                "Resource_Optimization_Z": r.get("metrics", {}).get("Resource_Optimization_Z"),
                "contract_attainment_rate": r.get("contract", {}).get("attainment_rate"),
                "Ttrans_ms_mean": sym.get("Ttrans_ms_mean"),
                "Tcons_ms_mean": sym.get("Tcons_ms_mean"),
                "ksub": sym.get("ksub"),
                "cloud_L_e2e_mean_ms": bc_timing.get("L_e2e_mean"),
                "cloud_Le2e_ms_total": (bc_sym.get("Le2e_ms_total") if isinstance(bc_sym, dict) else None),
                "cloud_Th_sys_points_per_s": bc_metrics.get("Th_sys_points_per_s"),
                "cloud_Th_sys_MB_per_s": bc_metrics.get("Th_sys_MB_per_s"),
                "cloud_Thsys_points_per_s": bc_metrics.get("Thsys_points_per_s"),
                "cloud_Thsys_MB_per_s": bc_metrics.get("Thsys_MB_per_s"),
                "cloud_Reliability_r": bc_metrics.get("Reliability_r"),
                "cloud_Reliability_R_percent": bc_metrics.get("Reliability_R_percent"),
                "cloud_Efficiency_e": bc_metrics.get("Efficiency_e"),
                "cloud_Efficiency_eta_percent": bc_metrics.get("Efficiency_eta_percent"),
                "cloud_T_c": bc_cost.get("T_c"),
                "cloud_Resource_Optimization_Ro": bc_metrics.get("Resource_Optimization_Ro"),
                "cloud_Resource_Optimization_Z": bc_metrics.get("Resource_Optimization_Z"),
            }
        )
    _write_csv_rows(out_csv=out_csv, fieldnames=fieldnames, rows=rows)


def _write_kpi_aggregate_csv(*, results: List[Dict[str, Any]], out_csv: Path) -> None:
    """Write a compact, cross-metric KPI summary (one row).

    Aggregates per-metric KPI results across columns, typically used to populate
    a single outcomes table row for a given configuration.
    """

    try:
        import statistics
    except Exception:  # pragma: no cover
        statistics = None  # type: ignore

    def _pluck(path: Sequence[str]) -> List[float]:
        vals: List[float] = []
        for r in results:
            cur: Any = r
            ok = True
            for k in path:
                if not isinstance(cur, dict) or k not in cur:
                    ok = False
                    break
                cur = cur[k]
            if not ok:
                continue
            try:
                vals.append(float(cur))
            except Exception:
                continue
        return vals

    def _mean(xs: List[float]) -> float:
        return float(sum(xs) / float(len(xs))) if xs else 0.0

    def _median(xs: List[float]) -> float:
        if not xs:
            return 0.0
        if statistics is None:
            s = sorted(xs)
            return float(s[len(s) // 2])
        return float(statistics.median(xs))

    def _min(xs: List[float]) -> float:
        return float(min(xs)) if xs else 0.0

    def _max(xs: List[float]) -> float:
        return float(max(xs)) if xs else 0.0

    # Assume configuration params are shared; if not, just leave them as blank/mixed.
    delta_vals = _pluck(["params", "delta_percent"])
    b_vals = _pluck(["params", "batch_size"])
    cp_vals = _pluck(["params", "confirm_prob"])

    # Key KPIs
    l_mean = _pluck(["timing_ms", "L_e2e_mean"])
    th = _pluck(["metrics", "Th_sys_points_per_s"])
    th_mb = _pluck(["metrics", "Th_sys_MB_per_s"])
    rr = _pluck(["metrics", "Reliability_r"])
    ee = _pluck(["metrics", "Efficiency_e"])
    tc = _pluck(["cost", "T_c"])
    tcu = _pluck(["cost", "Tc_unit"])
    pp = _pluck(["metrics", "Performance_p"])
    ro = _pluck(["metrics", "Resource_Optimization_Ro"])
    attain = _pluck(["contract", "attainment_rate"])

    # Cloud-only baseline KPIs (optional)
    l_mean_cloud = _pluck(["baseline_cloud", "timing_ms", "L_e2e_mean"])
    th_cloud = _pluck(["baseline_cloud", "metrics", "Th_sys_points_per_s"])
    th_mb_cloud = _pluck(["baseline_cloud", "metrics", "Th_sys_MB_per_s"])
    rr_cloud = _pluck(["baseline_cloud", "metrics", "Reliability_r"])
    ee_cloud = _pluck(["baseline_cloud", "metrics", "Efficiency_e"])
    tc_cloud = _pluck(["baseline_cloud", "cost", "T_c"])
    ro_cloud = _pluck(["baseline_cloud", "metrics", "Resource_Optimization_Ro"])

    # Delta-triggered smart-contract totals across all metrics (weighted overall).
    contract_settled_total = 0
    contract_pending_total = 0
    contract_failed_total = 0
    incentive_full_total = 0
    incentive_partial_total = 0
    incentive_none_total = 0
    for r in results:
        c = r.get("contract", {}) if isinstance(r, dict) else {}
        try:
            contract_settled_total += int(c.get("settled", 0))
            contract_pending_total += int(c.get("pending_audit", 0))
            contract_failed_total += int(c.get("failed", 0))
            incentive_full_total += int(c.get("incentive_full", 0))
            incentive_partial_total += int(c.get("incentive_partial", 0))
            incentive_none_total += int(c.get("incentive_none", 0))
        except Exception:
            continue
    contract_attempted_total = int(contract_settled_total + contract_pending_total + contract_failed_total)
    contract_attainment_rate_overall = (
        (float(contract_settled_total) / float(contract_attempted_total)) if contract_attempted_total > 0 else 0.0
    )

    fieldnames = [
        "metrics_count",
        "delta_percent_mean",
        "batch_size_mean",
        "confirm_prob_mean",
        "L_e2e_mean_ms_mean",
        "L_e2e_mean_ms_min",
        "L_e2e_mean_ms_max",
        "Th_sys_points_per_s_mean",
        "Th_sys_points_per_s_min",
        "Th_sys_points_per_s_max",
        "Th_sys_MB_per_s_mean",
        "Th_sys_MB_per_s_min",
        "Th_sys_MB_per_s_max",
        "Thsys_points_per_s_mean",
        "Reliability_r_mean",
        "Reliability_r_min",
        "Reliability_r_max",
        "Reliability_R_percent_mean",
        "Efficiency_e_mean",
        "Efficiency_e_min",
        "Efficiency_e_max",
        "Efficiency_eta_percent_mean",
        "T_c_mean",
        "T_c_min",
        "T_c_max",
        "Tc_unit_mean",
        "Tc_unit_min",
        "Tc_unit_max",
        "Performance_p_mean",
        "Performance_p_min",
        "Performance_p_max",
        "Resource_Optimization_Ro_mean",
        "Resource_Optimization_Ro_min",
        "Resource_Optimization_Ro_max",
        "Resource_Optimization_Z_mean",
        "cloud_L_e2e_mean_ms_mean",
        "cloud_Th_sys_points_per_s_mean",
        "cloud_Th_sys_MB_per_s_mean",
        "cloud_Reliability_r_mean",
        "cloud_Efficiency_e_mean",
        "cloud_T_c_mean",
        "cloud_Resource_Optimization_Ro_mean",
        "L_e2e_ratio_vs_cloud",
        "Th_sys_ratio_vs_cloud",
        "T_c_ratio_vs_cloud",
        "R_o_ratio_vs_cloud",
        "contract_attainment_rate_mean",
        "contract_attainment_rate_median",
        "contract_attainment_rate_min",
        "contract_attainment_rate_max",
        "contract_attempted_total",
        "contract_settled_total",
        "contract_pending_total",
        "contract_failed_total",
        "contract_attainment_rate_overall",
        "contract_incentive_full_total",
        "contract_incentive_partial_total",
        "contract_incentive_none_total",
    ]

    row = {
        "metrics_count": int(len(results)),
        "delta_percent_mean": round(_mean(delta_vals), 6),
        "batch_size_mean": round(_mean(b_vals), 6),
        "confirm_prob_mean": round(_mean(cp_vals), 6),
        "L_e2e_mean_ms_mean": round(_mean(l_mean), 6),
        "L_e2e_mean_ms_min": round(_min(l_mean), 6),
        "L_e2e_mean_ms_max": round(_max(l_mean), 6),
        "Th_sys_points_per_s_mean": round(_mean(th), 6),
        "Th_sys_points_per_s_min": round(_min(th), 6),
        "Th_sys_points_per_s_max": round(_max(th), 6),
        "Th_sys_MB_per_s_mean": round(_mean(th_mb), 6),
        "Th_sys_MB_per_s_min": round(_min(th_mb), 6),
        "Th_sys_MB_per_s_max": round(_max(th_mb), 6),
        "Thsys_points_per_s_mean": round(_mean(th), 6),
        "Reliability_r_mean": round(_mean(rr), 6),
        "Reliability_r_min": round(_min(rr), 6),
        "Reliability_r_max": round(_max(rr), 6),
        "Reliability_R_percent_mean": round(_mean(rr) * 100.0, 6),
        "Efficiency_e_mean": round(_mean(ee), 6),
        "Efficiency_e_min": round(_min(ee), 6),
        "Efficiency_e_max": round(_max(ee), 6),
        "Efficiency_eta_percent_mean": round(_mean(ee) * 100.0, 6),
        "T_c_mean": round(_mean(tc), 9),
        "T_c_min": round(_min(tc), 9),
        "T_c_max": round(_max(tc), 9),
        "Tc_unit_mean": round(_mean(tcu), 12),
        "Tc_unit_min": round(_min(tcu), 12),
        "Tc_unit_max": round(_max(tcu), 12),
        "Performance_p_mean": round(_mean(pp), 6),
        "Performance_p_min": round(_min(pp), 6),
        "Performance_p_max": round(_max(pp), 6),
        "Resource_Optimization_Ro_mean": round(_mean(ro), 12),
        "Resource_Optimization_Ro_min": round(_min(ro), 12),
        "Resource_Optimization_Ro_max": round(_max(ro), 12),
        "Resource_Optimization_Z_mean": round(_mean(ro), 12),
        "cloud_L_e2e_mean_ms_mean": round(_mean(l_mean_cloud), 6),
        "cloud_Th_sys_points_per_s_mean": round(_mean(th_cloud), 6),
        "cloud_Th_sys_MB_per_s_mean": round(_mean(th_mb_cloud), 6),
        "cloud_Reliability_r_mean": round(_mean(rr_cloud), 6),
        "cloud_Efficiency_e_mean": round(_mean(ee_cloud), 6),
        "cloud_T_c_mean": round(_mean(tc_cloud), 6),
        "cloud_Resource_Optimization_Ro_mean": round(_mean(ro_cloud), 12),
        "L_e2e_ratio_vs_cloud": round((_mean(l_mean) / _mean(l_mean_cloud)) if _mean(l_mean_cloud) > 0 else 0.0, 6),
        "Th_sys_ratio_vs_cloud": round((_mean(th_mb) / _mean(th_mb_cloud)) if _mean(th_mb_cloud) > 0 else 0.0, 6),
        "T_c_ratio_vs_cloud": round((_mean(tc) / _mean(tc_cloud)) if _mean(tc_cloud) > 0 else 0.0, 6),
        "R_o_ratio_vs_cloud": round((_mean(ro) / _mean(ro_cloud)) if _mean(ro_cloud) > 0 else 0.0, 6),
        "contract_attainment_rate_mean": round(_mean(attain), 6),
        "contract_attainment_rate_median": round(_median(attain), 6),
        "contract_attainment_rate_min": round(_min(attain), 6),
        "contract_attainment_rate_max": round(_max(attain), 6),
        "contract_attempted_total": int(contract_attempted_total),
        "contract_settled_total": int(contract_settled_total),
        "contract_pending_total": int(contract_pending_total),
        "contract_failed_total": int(contract_failed_total),
        "contract_attainment_rate_overall": round(float(contract_attainment_rate_overall), 6),
        "contract_incentive_full_total": int(incentive_full_total),
        "contract_incentive_partial_total": int(incentive_partial_total),
        "contract_incentive_none_total": int(incentive_none_total),
    }

    _write_csv_rows(out_csv=out_csv, fieldnames=fieldnames, rows=[row])


# -----------------------------
# Validation harness
# -----------------------------


@dataclass
class Check:
    name: str
    ok: bool
    message: str


def _ok(name: str, msg: str = "OK") -> Check:
    return Check(name=name, ok=True, message=msg)


def _warn(name: str, msg: str) -> Check:
    return Check(name=name, ok=False, message=msg)


def _run_checks(seed: int) -> List[Check]:
    random.seed(seed)
    checks: List[Check] = []

    # --- Algorithm 1: completeness & determinism ---
    criteria_repo = {
        "svc_a": ["latency", "throughput"],
        "svc_b": ["reliability"],
    }
    stakeholder_objectives = {"hospital": ["reduce_wait", "improve_quality"]}
    objective_services = {
        "reduce_wait": ["svc_a"],
        "improve_quality": ["svc_a", "svc_b"],
    }
    vc1 = algorithm_1_value_identification(
        "hospital", criteria_repo, stakeholder_objectives, objective_services
    )
    vc2 = algorithm_1_value_identification(
        "hospital", criteria_repo, stakeholder_objectives, objective_services
    )
    checks.append(_ok("Alg1 deterministic") if vc1 == vc2 else _warn("Alg1 deterministic", "outputs differ"))

    expected_min = {
        ("reduce_wait", "svc_a", "latency"),
        ("reduce_wait", "svc_a", "throughput"),
        ("improve_quality", "svc_b", "reliability"),
    }
    missing = expected_min - set(vc1)
    checks.append(_ok("Alg1 covers objectives/services/criteria") if not missing else _warn("Alg1 covers objectives/services/criteria", f"missing={sorted(missing)}"))

    # --- Algorithm 2: temporal validity, thresholding, monotonicity ---
    now = 1_000_000.0
    delta_t_max = 60.0
    rules = RuleSet(threshold=95.0)
    weights = {"spo2": 1.5, "temp": 1.0}
    v_info = [
        DataPoint(timestamp=now - 10, type="spo2", value=96.0),  # include
        DataPoint(timestamp=now - 10, type="spo2", value=94.0),  # exclude
        DataPoint(timestamp=now - 1000, type="spo2", value=99.0),  # stale
    ]

    s1 = algorithm_2_value_quantification(v_info, rules, 100, now_ts=now, delta_t_max=delta_t_max, weights=weights)
    s2 = algorithm_2_value_quantification(v_info, rules, 200, now_ts=now, delta_t_max=delta_t_max, weights=weights)
    checks.append(_ok("Alg2 integer score") if isinstance(s1, int) else _warn("Alg2 integer score", f"type={type(s1)}"))
    checks.append(_ok("Alg2 monotonic in sigma") if s2 >= s1 else _warn("Alg2 monotonic in sigma", f"s1={s1} s2={s2}"))

    # Check stale exclusion
    v_only_stale = [DataPoint(timestamp=now - 9999, type="spo2", value=100.0)]
    s_stale = algorithm_2_value_quantification(v_only_stale, rules, 100, now_ts=now, delta_t_max=delta_t_max, weights=weights)
    checks.append(_ok("Alg2 excludes stale data") if s_stale == 0 else _warn("Alg2 excludes stale data", f"score={s_stale}"))

    # --- Algorithm 3: boundary correctness ---
    log: List[Tuple[str, str]] = []
    st_eq = algorithm_3_value_delivery_execution(100.0, 100.0, 0.8, log=log)
    checks.append(_ok("Alg3 settlement at v_score==epsilon") if st_eq == "SETTLED" else _warn("Alg3 settlement at v_score==epsilon", f"status={st_eq}"))

    st_pending = algorithm_3_value_delivery_execution(100.0, 80.0, 0.8)
    checks.append(_ok("Alg3 pending at v_score==alpha*epsilon") if st_pending == "PENDING_AUDIT" else _warn("Alg3 pending at v_score==alpha*epsilon", f"status={st_pending}"))

    st_fail = algorithm_3_value_delivery_execution(100.0, 10.0, 0.8)
    checks.append(_ok("Alg3 failed below threshold") if st_fail == "FAILED" else _warn("Alg3 failed below threshold", f"status={st_fail}"))

    # --- Algorithm 4: authorization & privacy anchoring sanity ---
    iomt = {
        "patient_id": "P-001",
        "name": "Alice",
        "blood_oxygen": 96,
        "temperature": 36.7,
    }
    proof_ok = ConsentProof(subject_id="P-001", signature="sig:P-001:ok")
    proof_bad = ConsentProof(subject_id="P-001", signature="")
    contract = Contract(contract_id="C-1", active=True)

    try:
        _ = algorithm_4_value_capture_privacy_anchoring(iomt, proof_bad, contract)
        checks.append(_warn("Alg4 unauthorized rejected", "expected UnauthorizedError"))
    except UnauthorizedError:
        checks.append(_ok("Alg4 unauthorized rejected"))

    tx1 = algorithm_4_value_capture_privacy_anchoring(iomt, proof_ok, contract)
    iomt_changed = dict(iomt)
    iomt_changed["blood_oxygen"] = 95
    tx2 = algorithm_4_value_capture_privacy_anchoring(iomt_changed, proof_ok, contract)
    checks.append(_ok("Alg4 tx hash changes with data") if tx1 != tx2 else _warn("Alg4 tx hash changes with data", "tx unchanged"))

    anon = ANONYMIZE(iomt)
    checks.append(_ok("Alg4 anonymize removes identifiers") if ("patient_id" not in anon and "name" not in anon) else _warn("Alg4 anonymize removes identifiers", f"keys={sorted(anon.keys())}"))

    # --- Algorithm 5: branching & monotonic reward amount ---
    r_low = algorithm_5_adaptive_value_incentives(tau=10.0, gamma=50.0, epsilon=100.0, v_score=40.0)
    r_partial = algorithm_5_adaptive_value_incentives(tau=10.0, gamma=50.0, epsilon=100.0, v_score=60.0)
    r_full = algorithm_5_adaptive_value_incentives(tau=10.0, gamma=50.0, epsilon=100.0, v_score=120.0)
    checks.append(_ok("Alg5 below gamma notifies") if r_low.startswith("NOTIFIED:") else _warn("Alg5 below gamma notifies", r_low))
    checks.append(_ok("Alg5 partial band rewards") if r_partial.startswith("REWARDED_PARTIAL:") else _warn("Alg5 partial band rewards", r_partial))
    checks.append(_ok("Alg5 full band rewards") if r_full.startswith("REWARDED:") else _warn("Alg5 full band rewards", r_full))

    # Parse rewarded amount
    try:
        amt = float(r_full.split(":", 1)[1])
        checks.append(_ok("Alg5 reward amount feasible") if amt >= 10.0 else _warn("Alg5 reward amount feasible", f"amount={amt}"))
    except Exception as e:
        checks.append(_warn("Alg5 reward parse", str(e)))

    # --- Smart contract authorization: multisig/oracle proof sanity ---
    authorized_nodes = {"fog_a": "secret_a", "fog_b": "secret_b", "fog_c": "secret_c"}
    oracle_secrets = {"oracle_1": "oracle_secret"}
    payload = _score_payload(metric="spo2", v_score=12345, now_ts=now)

    proof_ok = MultiSigProof(
        signatures={
            "fog_a": _sign_payload(authorized_nodes["fog_a"], payload),
            "fog_b": _sign_payload(authorized_nodes["fog_b"], payload),
        },
        required=2,
    )
    proof_bad = MultiSigProof(signatures={"fog_x": "deadbeef"}, required=1)
    checks.append(_ok("Contract multisig verifies") if VERIFY_MULTISIG(payload=payload, proof=proof_ok, authorized_node_secrets=authorized_nodes) else _warn("Contract multisig verifies", "verification failed"))
    checks.append(_ok("Contract multisig rejects unauthorized") if not VERIFY_MULTISIG(payload=payload, proof=proof_bad, authorized_node_secrets=authorized_nodes) else _warn("Contract multisig rejects unauthorized", "unexpectedly accepted"))

    oracle_ok = OracleProof(oracle_id="oracle_1", signature=_sign_payload(oracle_secrets["oracle_1"], payload))
    oracle_bad = OracleProof(oracle_id="oracle_1", signature="bad")
    checks.append(_ok("Contract oracle verifies") if VERIFY_ORACLE(payload=payload, proof=oracle_ok, oracle_secrets=oracle_secrets) else _warn("Contract oracle verifies", "verification failed"))
    checks.append(_ok("Contract oracle rejects bad sig") if not VERIFY_ORACLE(payload=payload, proof=oracle_bad, oracle_secrets=oracle_secrets) else _warn("Contract oracle rejects bad sig", "unexpectedly accepted"))

    return checks


def _run_iomt_sampling_checks(
    *,
    sampling_csv: Path,
    sampling_limit: int,
    sampling_chunksize: int,
    time_col: str,
    metric_col: str,
    all_cols: bool,
    threshold_auto: bool,
    threshold: float,
    sigma: int,
    epsilon: float,
    alpha: float,
    contract_attainment_min: float,
    incentive_attainment_min: float,
    gamma: float,
    proof_mode: str,
    required_sigs: int,
    fog_smooth_window: int,
    fog_store_dir: Optional[Path],
    fog_store_max_items: int,
    fog_agg_window: int,
    fog_agg_func: str,
    fog_cache_ttl_seconds: float,
    fog_offline_rate: float,
    fog_byzantine_rate: float,
    fog_byzantine_noise_std: float,
    fog_consensus_max_rel_range: float,
    fog_enforce_consensus: bool,
    seed: int,
) -> List[Check]:
    checks: List[Check] = []

    if not sampling_csv.exists():
        return [_warn("IoMT sampling CSV available", f"missing: {sampling_csv}")]

    try:
        import pandas as pd  # type: ignore
    except Exception as e:  # pragma: no cover
        # Fallback: pure-Python CSV parsing (works even if stdlib csv is broken).
        checks.append(_ok("IoMT sampling requires pandas", "pandas unavailable; used pure-Python CSV fallback"))
        checks.extend(
            _run_iomt_sampling_checks_no_pandas(
                sampling_csv=sampling_csv,
                sampling_limit=sampling_limit,
                sampling_chunksize=sampling_chunksize,
                time_col=time_col,
                metric_col=metric_col,
                all_cols=all_cols,
                threshold_auto=threshold_auto,
                threshold=threshold,
                sigma=sigma,
                epsilon=epsilon,
                alpha=alpha,
                contract_attainment_min=contract_attainment_min,
                incentive_attainment_min=incentive_attainment_min,
                gamma=gamma,
                proof_mode=proof_mode,
                required_sigs=required_sigs,
                fog_smooth_window=fog_smooth_window,
                fog_store_dir=fog_store_dir,
                fog_store_max_items=fog_store_max_items,
                fog_agg_window=fog_agg_window,
                fog_agg_func=fog_agg_func,
                fog_cache_ttl_seconds=fog_cache_ttl_seconds,
                fog_offline_rate=fog_offline_rate,
                fog_byzantine_rate=fog_byzantine_rate,
                fog_byzantine_noise_std=fog_byzantine_noise_std,
                fog_consensus_max_rel_range=fog_consensus_max_rel_range,
                fog_enforce_consensus=fog_enforce_consensus,
                seed=seed,
            )
        )
        return checks

    try:
        df: Any
        if sampling_chunksize is not None and int(sampling_chunksize) > 0:
            # Batch mode: avoid loading entire CSV into memory.
            df = None
        else:
            df = pd.read_csv(sampling_csv)
    except Exception as e:
        return [_warn("IoMT sampling CSV readable", f"error: {e}")]

    # Resolve columns without loading full CSV (works for both modes).
    try:
        cols_df = pd.read_csv(sampling_csv, nrows=0)
        cols = list(cols_df.columns)
    except Exception as e:
        return [_warn("IoMT sampling CSV header readable", f"error: {e}")]

    if not cols:
        return [_warn("IoMT sampling CSV non-empty", "no columns")]

    cols_lower = {str(c).strip().lower(): c for c in cols}
    time_key = time_col.strip().lower()
    metric_key = metric_col.strip().lower()

    # Determine which columns to validate
    id_cols = {"sample_id", "resample_pos"}
    if all_cols:
        candidates = [str(c).strip() for c in cols if str(c).strip().lower() not in id_cols]
    else:
        if metric_key not in cols_lower:
            return [_warn("IoMT metric column exists", f"missing column: {metric_col}")]
        candidates = [str(cols_lower[metric_key])]

    checks.append(_ok("IoMT columns selected", f"count={len(candidates)}"))

    # Smart-contract attainment is measured at the column-level integration test:
    # each validated metric column represents one contract execution attempt
    # resulting in SETTLED / PENDING_AUDIT / FAILED (Algorithm 3).
    contract_settled = 0
    contract_pending = 0
    contract_failed = 0

    incentive_full = 0
    incentive_partial = 0
    incentive_none = 0

    # Minimal authorized fog nodes / oracle for verification.
    authorized_nodes = {"fog_a": "secret_a", "fog_b": "secret_b", "fog_c": "secret_c"}
    oracle_secrets = {"oracle_1": "oracle_secret"}

    fog = FogCluster.from_secrets(
        node_secrets=authorized_nodes,
        smooth_window=int(fog_smooth_window),
        agg_window=int(fog_agg_window),
        agg_func=str(fog_agg_func),
        store_dir=fog_store_dir,
        store_max_items=int(fog_store_max_items),
        cache_ttl_seconds=float(fog_cache_ttl_seconds),
    )

    # Two execution modes:
    # - non-batch: keep previous behaviour (single in-memory dataframe)
    # - batch: stream CSV in chunks, compute per-column summaries across all rows
    if sampling_chunksize is None or int(sampling_chunksize) <= 0:
        if df is None or getattr(df, "empty", True):
            return [_warn("IoMT sampling CSV non-empty", "empty dataframe")]

        if time_key and time_key in cols_lower:
            ts = pd.to_numeric(df[cols_lower[time_key]], errors="coerce")
        else:
            ts = pd.Series(range(len(df)), dtype="float64")

        if sampling_limit is not None and sampling_limit > 0:
            df = df.iloc[:sampling_limit].copy()
            ts = ts.iloc[:sampling_limit]

        # Shared delta_t_max across all columns
        ts_valid = ts.dropna()
        if ts_valid.empty:
            return [_warn("IoMT timestamp column valid", f"no valid timestamps in: {time_col}")]
        now_ts = float(ts_valid.max())
        min_ts = float(ts_valid.min())
        delta_t_max = max(0.0, (now_ts - min_ts) + 1e-9)

        for col_name in candidates:
            col_series = pd.to_numeric(df[col_name], errors="coerce")
            valid_mask = (~ts.isna()) & (~col_series.isna())
            if valid_mask.sum() == 0:
                checks.append(_warn(f"IoMT col {col_name} has data", "all values missing after coercion"))
                continue

            missing_rate = float(col_series.isna().mean())
            checks.append(_ok(f"IoMT col {col_name} missing-rate", f"missing={missing_rate:.2%}"))

            ts_use = ts[valid_mask]
            v_use = col_series[valid_mask]
            datapoints = [
                DataPoint(timestamp=float(t), type=str(col_name).strip().lower(), value=float(v))
                for t, v in zip(ts_use.to_numpy(), v_use.to_numpy())
            ]

            # Choose threshold
            if threshold_auto:
                thr = float(v_use.median())
            else:
                thr = float(threshold)
            rules = RuleSet(threshold=thr)
            weights = {datapoints[0].type: 1.0}

            # Fog processing stage (per node)
            try:
                processed_by_node = fog.process(
                    metric=str(col_name),
                    datapoints=datapoints,
                    now_ts=now_ts,
                    offline_rate=float(fog_offline_rate),
                    byzantine_rate=float(fog_byzantine_rate),
                    byzantine_noise_std=float(fog_byzantine_noise_std),
                    rng_seed=int(seed),
                )
                if len(processed_by_node) == 0:
                    checks.append(_warn(f"Fog processes {col_name}", "no available fog nodes (all offline)") )
                    contract_failed += 1
                    incentive_none += 1
                    continue
                checks.append(
                    _ok(
                        f"Fog processes {col_name}",
                        f"nodes={len(processed_by_node)} smooth_window={int(fog_smooth_window)} agg_window={int(fog_agg_window)} agg_func={str(fog_agg_func)} ttl={float(fog_cache_ttl_seconds)}s offline_rate={float(fog_offline_rate):.2f} byz_rate={float(fog_byzantine_rate):.2f}",
                    )
                )
            except Exception as e:
                checks.append(_warn(f"Fog processes {col_name}", f"error: {e}"))
                continue

            try:
                scores = fog.compute_scores(
                    metric=str(col_name),
                    processed=processed_by_node,
                    rules=rules,
                    sigma=int(sigma),
                    now_ts=now_ts,
                    delta_t_max=delta_t_max,
                    weights=weights,
                )

                # Consensus rule: median score across fog nodes (robust to outliers)
                score_values = sorted(int(v) for v in scores.values())
                v_score = int(score_values[len(score_values) // 2])

                all_same = all(v == score_values[0] for v in score_values)

                # Relative range: (max-min)/max(1, |median|)
                if len(score_values) >= 2:
                    rel_range = (max(score_values) - min(score_values)) / float(max(1, abs(v_score)))
                else:
                    rel_range = 0.0

                consistency_msg = f"all_same={all_same} rel_range={rel_range:.4f} scores={score_values}"
                if float(fog_consensus_max_rel_range) > 0.0 and rel_range > float(fog_consensus_max_rel_range):
                    checks.append(
                        _warn(f"Fog score consistency {col_name}", consistency_msg)
                        if bool(fog_enforce_consensus)
                        else _ok(f"Fog score consistency {col_name}", consistency_msg)
                    )
                    if bool(fog_enforce_consensus):
                        # Treat as a contract failure: consensus not reached.
                        contract_failed += 1
                        incentive_none += 1
                        continue
                else:
                    checks.append(_ok(f"Fog score consistency {col_name}", consistency_msg))

                checks.append(_ok(f"Alg2 (fog) runs on {col_name}", f"threshold={thr} v_score={v_score}"))
            except Exception as e:
                checks.append(_warn(f"Alg2 (fog) runs on {col_name}", f"error: {e}"))
                continue

            # Contract must verify the score was submitted by authorized fog nodes (multisig) or oracle.
            payload = _score_payload(metric=str(col_name), v_score=int(v_score), now_ts=now_ts)
            is_authorized = False
            mode = str(proof_mode).strip().lower()
            if mode == "none":
                is_authorized = True
            elif mode == "oracle":
                oracle_id = "oracle_1"
                proof = OracleProof(oracle_id=oracle_id, signature=_sign_payload(oracle_secrets[oracle_id], payload))
                is_authorized = VERIFY_ORACLE(payload=payload, proof=proof, oracle_secrets=oracle_secrets)
            else:
                # Default: multisig
                proof = fog.multisig_proof(metric=str(col_name), v_score=int(v_score), now_ts=now_ts, required=int(required_sigs))
                is_authorized = VERIFY_MULTISIG(payload=payload, proof=proof, authorized_node_secrets=authorized_nodes)

            if is_authorized:
                checks.append(_ok(f"Contract score authorized for {col_name}", f"mode={mode}"))
            else:
                checks.append(_warn(f"Contract score authorized for {col_name}", f"mode={mode} verification failed"))
                # Unauthorized score cannot trigger settlement/incentives.
                contract_failed += 1
                incentive_none += 1
                continue

            try:
                st = algorithm_3_value_delivery_execution(float(epsilon), float(v_score), float(alpha))
                checks.append(_ok(f"Alg3 runs on {col_name}", f"status={st}"))
                if st == "SETTLED":
                    contract_settled += 1
                elif st == "PENDING_AUDIT":
                    contract_pending += 1
                else:
                    contract_failed += 1
            except Exception as e:
                checks.append(_warn(f"Alg3 runs on {col_name}", f"error: {e}"))

            try:
                i_status = algorithm_5_adaptive_value_incentives(
                    tau=10.0,
                    gamma=float(gamma),
                    epsilon=float(epsilon),
                    v_score=float(v_score),
                )
                checks.append(_ok(f"Alg5 runs on {col_name}", f"status={i_status}"))
                if i_status.startswith("REWARDED:"):
                    incentive_full += 1
                elif i_status.startswith("REWARDED_PARTIAL:"):
                    incentive_partial += 1
                else:
                    incentive_none += 1
            except Exception as e:
                checks.append(_warn(f"Alg5 runs on {col_name}", f"error: {e}"))
                incentive_none += 1
    else:
        # Batch mode summaries per column (streaming across all rows).
        chunksize = int(sampling_chunksize)

        total_rows_by_col: Dict[str, int] = {c: 0 for c in candidates}
        missing_rows_by_col: Dict[str, int] = {c: 0 for c in candidates}
        # Per-column contract + incentive counters across all chunks.
        col_contract: Dict[str, Dict[str, int]] = {
            c: {
                "settled": 0,
                "pending": 0,
                "failed": 0,
                "incentive_full": 0,
                "incentive_partial": 0,
                "incentive_none": 0,
                "fog_offline_fail": 0,
                "consensus_fail": 0,
                "unauthorized_fail": 0,
                "batches": 0,
            }
            for c in candidates
        }

        processed_rows = 0
        offset = 0
        time_col_actual = cols_lower.get(time_key) if (time_key and time_key in cols_lower) else None

        # Read only required columns for efficiency.
        usecols = list(dict.fromkeys(([time_col_actual] if time_col_actual else []) + candidates))

        try:
            it = pd.read_csv(sampling_csv, usecols=usecols, chunksize=chunksize)
        except Exception as e:
            return [_warn("IoMT sampling CSV readable (batch)", f"error: {e}")]

        for chunk in it:
            if chunk is None or chunk.empty:
                continue

            if sampling_limit is not None and int(sampling_limit) > 0 and processed_rows >= int(sampling_limit):
                break

            if sampling_limit is not None and int(sampling_limit) > 0:
                remaining = int(sampling_limit) - processed_rows
                if remaining <= 0:
                    break
                if len(chunk) > remaining:
                    chunk = chunk.iloc[:remaining].copy()

            processed_rows += int(len(chunk))

            if time_col_actual is not None and time_col_actual in chunk.columns:
                ts = pd.to_numeric(chunk[time_col_actual], errors="coerce")
            else:
                ts = pd.Series(range(offset, offset + len(chunk)), dtype="float64")
            offset += int(len(chunk))

            # Process each candidate column in this chunk.
            for col_name in candidates:
                if col_name not in chunk.columns:
                    continue

                col_series = pd.to_numeric(chunk[col_name], errors="coerce")
                total_rows_by_col[col_name] += int(len(col_series))
                missing_rows_by_col[col_name] += int(col_series.isna().sum())

                valid_mask = (~ts.isna()) & (~col_series.isna())
                if int(valid_mask.sum()) == 0:
                    continue

                ts_use = ts[valid_mask]
                v_use = col_series[valid_mask]
                datapoints = [
                    DataPoint(timestamp=float(t), type=str(col_name).strip().lower(), value=float(v))
                    for t, v in zip(ts_use.to_numpy(), v_use.to_numpy())
                ]
                if not datapoints:
                    continue

                col_contract[col_name]["batches"] += 1
                now_ts = float(ts_use.max())
                delta_t_max = float("inf")

                # Choose threshold per-batch.
                if threshold_auto:
                    thr = float(v_use.median())
                else:
                    thr = float(threshold)
                rules = RuleSet(threshold=thr)
                weights = {datapoints[0].type: 1.0}

                try:
                    processed_by_node = fog.process(
                        metric=str(col_name),
                        datapoints=datapoints,
                        now_ts=now_ts,
                        offline_rate=float(fog_offline_rate),
                        byzantine_rate=float(fog_byzantine_rate),
                        byzantine_noise_std=float(fog_byzantine_noise_std),
                        rng_seed=int(seed),
                    )
                    if len(processed_by_node) == 0:
                        col_contract[col_name]["fog_offline_fail"] += 1
                        col_contract[col_name]["failed"] += 1
                        col_contract[col_name]["incentive_none"] += 1
                        continue
                except Exception:
                    col_contract[col_name]["failed"] += 1
                    col_contract[col_name]["incentive_none"] += 1
                    continue

                try:
                    scores = fog.compute_scores(
                        metric=str(col_name),
                        processed=processed_by_node,
                        rules=rules,
                        sigma=int(sigma),
                        now_ts=now_ts,
                        delta_t_max=delta_t_max,
                        weights=weights,
                    )
                    score_values = sorted(int(v) for v in scores.values())
                    v_score = int(score_values[len(score_values) // 2])
                    if len(score_values) >= 2:
                        rel_range = (max(score_values) - min(score_values)) / float(max(1, abs(v_score)))
                    else:
                        rel_range = 0.0
                    if float(fog_consensus_max_rel_range) > 0.0 and rel_range > float(fog_consensus_max_rel_range) and bool(fog_enforce_consensus):
                        col_contract[col_name]["consensus_fail"] += 1
                        col_contract[col_name]["failed"] += 1
                        col_contract[col_name]["incentive_none"] += 1
                        continue
                except Exception:
                    col_contract[col_name]["failed"] += 1
                    col_contract[col_name]["incentive_none"] += 1
                    continue

                payload = _score_payload(metric=str(col_name), v_score=int(v_score), now_ts=now_ts)
                mode = str(proof_mode).strip().lower()
                if mode == "none":
                    is_authorized = True
                elif mode == "oracle":
                    oracle_id = "oracle_1"
                    proof = OracleProof(oracle_id=oracle_id, signature=_sign_payload(oracle_secrets[oracle_id], payload))
                    is_authorized = VERIFY_ORACLE(payload=payload, proof=proof, oracle_secrets=oracle_secrets)
                else:
                    proof = fog.multisig_proof(metric=str(col_name), v_score=int(v_score), now_ts=now_ts, required=int(required_sigs))
                    is_authorized = VERIFY_MULTISIG(payload=payload, proof=proof, authorized_node_secrets=authorized_nodes)

                if not is_authorized:
                    col_contract[col_name]["unauthorized_fail"] += 1
                    col_contract[col_name]["failed"] += 1
                    col_contract[col_name]["incentive_none"] += 1
                    continue

                st = algorithm_3_value_delivery_execution(float(epsilon), float(v_score), float(alpha))
                if st == "SETTLED":
                    col_contract[col_name]["settled"] += 1
                elif st == "PENDING_AUDIT":
                    col_contract[col_name]["pending"] += 1
                else:
                    col_contract[col_name]["failed"] += 1

                i_status = algorithm_5_adaptive_value_incentives(
                    tau=10.0,
                    gamma=float(gamma),
                    epsilon=float(epsilon),
                    v_score=float(v_score),
                )
                if i_status.startswith("REWARDED:"):
                    col_contract[col_name]["incentive_full"] += 1
                elif i_status.startswith("REWARDED_PARTIAL:"):
                    col_contract[col_name]["incentive_partial"] += 1
                else:
                    col_contract[col_name]["incentive_none"] += 1

        # Emit per-column summaries.
        for col_name in candidates:
            total_n = int(total_rows_by_col.get(col_name, 0))
            miss_n = int(missing_rows_by_col.get(col_name, 0))
            missing_rate = (float(miss_n) / float(total_n)) if total_n > 0 else 1.0
            checks.append(_ok(f"IoMT col {col_name} missing-rate", f"missing={missing_rate:.2%} rows={total_n}"))

            c = col_contract[col_name]
            attempted = int(c["settled"] + c["pending"] + c["failed"])
            attainment = (float(c["settled"]) / float(attempted)) if attempted > 0 else 0.0
            msg = (
                f"batches={int(c['batches'])} attempted={attempted} settled={int(c['settled'])} "
                f"pending={int(c['pending'])} failed={int(c['failed'])} rate={attainment:.2%} "
                f"offline_fail={int(c['fog_offline_fail'])} consensus_fail={int(c['consensus_fail'])} unauthorized_fail={int(c['unauthorized_fail'])}"
            )
            checks.append(_ok(f"Smart contract attainment (batch) {col_name}", msg))

            inc_attempted = int(c["incentive_full"] + c["incentive_partial"] + c["incentive_none"])
            inc_rate = (float(c["incentive_full"] + c["incentive_partial"]) / float(inc_attempted)) if inc_attempted > 0 else 0.0
            msg2 = (
                f"attempted={inc_attempted} full={int(c['incentive_full'])} partial={int(c['incentive_partial'])} "
                f"none={int(c['incentive_none'])} rate={inc_rate:.2%}"
            )
            checks.append(_ok(f"Smart contract incentives (batch) {col_name}", msg2))

        # Roll up totals across all columns in batch mode.
        for col_name in candidates:
            c = col_contract[col_name]
            contract_settled += int(c["settled"])
            contract_pending += int(c["pending"])
            contract_failed += int(c["failed"])
            incentive_full += int(c["incentive_full"])
            incentive_partial += int(c["incentive_partial"])
            incentive_none += int(c["incentive_none"])

    # Summarize contract attainment across all validated columns.
    attempted = contract_settled + contract_pending + contract_failed
    if attempted <= 0:
        checks.append(_warn("Smart contract attainment rate", "no contract attempts were executed (Alg3 never ran)"))
        return checks

    attainment_rate = contract_settled / float(attempted)
    msg = (
        f"rate={attainment_rate:.2%} settled={contract_settled} "
        f"pending={contract_pending} failed={contract_failed} attempted={attempted} "
        f"min_required={float(contract_attainment_min):.2%}"
    )
    checks.append(
        _ok("Smart contract attainment rate", msg)
        if attainment_rate >= float(contract_attainment_min)
        else _warn("Smart contract attainment rate", msg)
    )

    incentive_attempted = incentive_full + incentive_partial + incentive_none
    if incentive_attempted > 0:
        incentive_rate = (incentive_full + incentive_partial) / float(incentive_attempted)
        msg2 = (
            f"rate={incentive_rate:.2%} full={incentive_full} partial={incentive_partial} "
            f"none={incentive_none} attempted={incentive_attempted} "
            f"min_required={float(incentive_attainment_min):.2%}"
        )
        checks.append(
            _ok("Smart contract incentive trigger rate", msg2)
            if incentive_rate >= float(incentive_attainment_min)
            else _warn("Smart contract incentive trigger rate", msg2)
        )

    return checks


def _render_report(checks: Sequence[Check]) -> str:
    ok = sum(1 for c in checks if c.ok)
    total = len(checks)
    status = "PASS" if ok == total else "CHECK"
    lines = [
        "Algorithm Reliability & Feasibility Validation Report",
        f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"status: {status} ({ok}/{total} checks OK)",
        "",
    ]
    for c in checks:
        tag = "[OK]" if c.ok else "[WARN]"
        lines.append(f"{tag} {c.name} :: {c.message}")
    lines.append("")
    lines.append("Scope:")
    lines.append("- Validates internal consistency, boundary behaviour, determinism, and basic plausibility of executable references mirroring the pseudocode.")
    lines.append("- Does not prove real-world blockchain/network reliability; it ensures the algorithmic logic is implementable and testable.")
    return "\n".join(lines)


def _run_kpi_model(args: argparse.Namespace) -> List[Check]:
    if (not bool(args.kpi)) or bool(args.no_sampling):
        return []

    checks: List[Check] = []

    def _build_fog_and_inputs() -> Tuple["FogCluster", "SystemKPIInputs"]:
        authorized_nodes = {"fog_a": "secret_a", "fog_b": "secret_b", "fog_c": "secret_c"}
        fog = FogCluster.from_secrets(
            node_secrets=authorized_nodes,
            smooth_window=int(args.fog_smooth_window),
            agg_window=int(args.fog_agg_window),
            agg_func=str(args.fog_agg_func),
            store_dir=args.fog_store_dir,
            store_max_items=int(args.fog_store_max_items),
            cache_ttl_seconds=float(args.fog_cache_ttl_seconds),
        )
        kpi_inputs = SystemKPIInputs(
            delta_percent=float(args.kpi_delta_percent),
            batch_size=int(args.kpi_batch_size),
            confirm_prob=float(args.kpi_confirm_prob),
            t_tx_ms=float(args.kpi_tx_ms),
            t_tx_jitter_ms=float(args.kpi_tx_jitter_ms),
            t_conf_ms=float(args.kpi_conf_ms),
            t_conf_jitter_ms=float(args.kpi_conf_jitter_ms),
            t_cloud_ms=float(args.kpi_cloud_ms),
            t_cloud_jitter_ms=float(args.kpi_cloud_jitter_ms),
            c_device_unit=float(args.kpi_c_device_unit),
            c_fog_unit=float(args.kpi_c_fog_unit),
            c_cloud_unit=float(args.kpi_c_cloud_unit) if args.kpi_c_cloud_unit is not None else float(args.kpi_c_fog_unit),
            c_sc_unit=float(args.kpi_c_sc_unit),
            c_conf_unit=float(args.kpi_c_conf_unit),
            bytes_per_point=float(args.kpi_bytes_per_point),
        )
        return fog, kpi_inputs

    def _write_outputs(results: List[Dict[str, Any]]) -> None:
        if not results:
            checks.append(_warn("System KPI model computed", "no valid metric columns for KPI model"))
            return

        if args.kpi_out_json is None:
            ts2 = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_json = Path(f"kpi_summary_{ts2}.json")
        else:
            out_json = Path(args.kpi_out_json)

        out_csv = Path(args.kpi_out_csv) if args.kpi_out_csv is not None else None
        _write_kpi_outputs(results=results, out_json=out_json, out_csv=out_csv)

        if bool(args.kpi_aggregate):
            if args.kpi_aggregate_out_csv is None:
                ts3 = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_agg = Path(f"kpi_aggregate_{ts3}.csv")
            else:
                out_agg = Path(args.kpi_aggregate_out_csv)
            _write_kpi_aggregate_csv(results=results, out_csv=out_agg)
            checks.append(_ok("System KPI aggregate CSV", f"out_csv={out_agg}"))

        checks.append(_ok("System KPI model computed", f"metrics={len(results)} out_json={out_json}"))

    # Prefer pandas for speed; fall back to pure-Python loader if pandas cannot be imported.
    try:
        import pandas as pd  # type: ignore

        df = pd.read_csv(args.sampling_csv)
        if args.sampling_limit is not None and args.sampling_limit > 0:
            df = df.iloc[: args.sampling_limit].copy()

        cols_lower = {str(c).strip().lower(): c for c in df.columns}
        time_key = str(args.sampling_time_col).strip().lower()
        if time_key and time_key in cols_lower:
            ts = pd.to_numeric(df[cols_lower[time_key]], errors="coerce")
        else:
            ts = pd.Series(range(len(df)), dtype="float64")

        id_cols = {"sample_id", "resample_pos"}
        if bool(args.sampling_all_cols):
            kpi_cols = [str(c).strip() for c in df.columns if str(c).strip().lower() not in id_cols]
        else:
            key = str(args.sampling_metric_col).strip().lower()
            if key not in cols_lower:
                _write_outputs([])
                return checks
            kpi_cols = [str(cols_lower[key])]

        fog, kpi_inputs = _build_fog_and_inputs()
        results: List[Dict[str, Any]] = []
        for col_name in kpi_cols:
            col_series = pd.to_numeric(df[col_name], errors="coerce")
            valid_mask = (~ts.isna()) & (~col_series.isna())
            if valid_mask.sum() == 0:
                continue
            ts_use = ts[valid_mask]
            v_use = col_series[valid_mask]
            dps = [
                DataPoint(timestamp=float(t), type=str(col_name).strip().lower(), value=float(v))
                for t, v in zip(ts_use.to_numpy(), v_use.to_numpy())
            ]
            results.append(
                compute_system_kpis_for_metric(
                    metric=str(col_name).strip().lower(),
                    datapoints=dps,
                    fog=fog,
                    fog_offline_rate=float(args.fog_offline_rate),
                    fog_byzantine_rate=float(args.fog_byzantine_rate),
                    fog_byzantine_noise_std=float(args.fog_byzantine_noise_std),
                    fog_consensus_max_rel_range=float(args.fog_consensus_max_rel_range),
                    fog_enforce_consensus=bool(args.fog_enforce_consensus),
                    seed=int(args.seed),
                    proof_mode=str(args.contract_proof_mode),
                    required_sigs=int(args.contract_required_sigs),
                    epsilon=float(args.sampling_epsilon),
                    alpha=float(args.sampling_alpha),
                    gamma=float(args.sampling_gamma),
                    sigma=int(args.sampling_sigma),
                    threshold=float(args.sampling_threshold),
                    threshold_auto=bool(args.sampling_threshold_auto) or bool(args.sampling_all_cols),
                    kpi=kpi_inputs,
                    cloud_only_baseline=bool(args.kpi_cloud_baseline),
                )
            )
        _write_outputs(results)
        return checks
    except Exception:
        # Fall back without pandas.
        try:
            ts_list, values_by_col = _load_kpi_inputs_no_pandas(
                sampling_csv=Path(args.sampling_csv),
                sampling_limit=args.sampling_limit,
                time_col=str(args.sampling_time_col),
                sampling_all_cols=bool(args.sampling_all_cols),
                sampling_metric_col=str(args.sampling_metric_col),
            )
            if not ts_list or not values_by_col:
                checks.append(_warn("System KPI model computed", "no valid rows/columns for KPI model (fallback)"))
                return checks

            fog, kpi_inputs = _build_fog_and_inputs()
            results2: List[Dict[str, Any]] = []
            for col_name, series in values_by_col.items():
                dps: List[DataPoint] = []
                for t, v in zip(ts_list, series):
                    if v is None or (isinstance(v, float) and math.isnan(v)):
                        continue
                    dps.append(DataPoint(timestamp=float(t), type=str(col_name).strip().lower(), value=float(v)))
                if not dps:
                    continue
                results2.append(
                    compute_system_kpis_for_metric(
                        metric=str(col_name).strip().lower(),
                        datapoints=dps,
                        fog=fog,
                        fog_offline_rate=float(args.fog_offline_rate),
                        fog_byzantine_rate=float(args.fog_byzantine_rate),
                        fog_byzantine_noise_std=float(args.fog_byzantine_noise_std),
                        fog_consensus_max_rel_range=float(args.fog_consensus_max_rel_range),
                        fog_enforce_consensus=bool(args.fog_enforce_consensus),
                        seed=int(args.seed),
                        proof_mode=str(args.contract_proof_mode),
                        required_sigs=int(args.contract_required_sigs),
                        epsilon=float(args.sampling_epsilon),
                        alpha=float(args.sampling_alpha),
                        gamma=float(args.sampling_gamma),
                        sigma=int(args.sampling_sigma),
                        threshold=float(args.sampling_threshold),
                        threshold_auto=bool(args.sampling_threshold_auto) or bool(args.sampling_all_cols),
                        kpi=kpi_inputs,
                        cloud_only_baseline=bool(args.kpi_cloud_baseline),
                    )
                )
            _write_outputs(results2)
            if results2:
                checks.append(_ok("System KPI model data source", "pure-Python CSV fallback"))
            return checks
        except Exception as e2:
            checks.append(_warn("System KPI model computed", f"error: {e2}"))
            return checks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate reliability and feasibility of Algorithms 1–5 pseudocode via executable references and invariant checks."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic checks")
    parser.add_argument(
        "--sampling-csv",
        type=Path,
        default=Path(__file__).with_name("IoMTPatient Sampling.csv"),
        help="IoMT sampling CSV for integration test (default: IoMTPatient Sampling.csv)",
    )
    parser.add_argument(
        "--sampling-limit",
        type=int,
        default=5000,
        help="Max rows to read from sampling CSV (0 or negative means no limit)",
    )
    parser.add_argument(
        "--sampling-chunksize",
        type=int,
        default=0,
        help="Read sampling CSV in batches of this size for full-data validation (0 disables batch mode)",
    )
    parser.add_argument("--sampling-time-col", type=str, default="resample_pos", help="Timestamp column in sampling CSV")
    parser.add_argument("--sampling-metric-col", type=str, default="blood_oxygen", help="Metric column to validate (default blood_oxygen)")
    parser.add_argument("--sampling-all-cols", action="store_true", help="Validate all non-id columns in the sampling CSV")
    parser.add_argument(
        "--sampling-threshold-auto",
        action="store_true",
        help="Auto threshold per column (median) instead of fixed --sampling-threshold",
    )
    parser.add_argument("--sampling-threshold", type=float, default=95.0, help="Threshold used in Algorithm 2 for sampling test")
    parser.add_argument("--sampling-sigma", type=int, default=100, help="Scaling factor sigma used in Algorithm 2 for sampling test")
    parser.add_argument("--sampling-epsilon", type=float, default=100.0, help="Target epsilon for Algorithm 3 in sampling test")
    parser.add_argument("--sampling-alpha", type=float, default=0.8, help="Tolerance alpha for Algorithm 3 in sampling test")
    parser.add_argument("--sampling-gamma", type=float, default=50.0, help="Gamma threshold for Algorithm 5 (partial incentive band starts at gamma)")
    parser.add_argument(
        "--sampling-contract-attainment-min",
        type=float,
        default=0.90,
        help="Minimum smart contract attainment rate required to pass (measured as SETTLED/attempted across validated columns)",
    )
    parser.add_argument(
        "--sampling-incentive-attainment-min",
        type=float,
        default=0.90,
        help="Minimum incentive trigger rate required to pass (measured as (FULL+PARTIAL)/attempted across validated columns)",
    )
    parser.add_argument(
        "--contract-proof-mode",
        type=str,
        default="multisig",
        choices=["multisig", "oracle", "none"],
        help="How the contract verifies V_score submissions: multisig, oracle, or none",
    )
    parser.add_argument(
        "--contract-required-sigs",
        type=int,
        default=2,
        help="Required number of fog-node signatures when --contract-proof-mode=multisig",
    )
    parser.add_argument(
        "--fog-smooth-window",
        type=int,
        default=1,
        help="Fog-side moving average smoothing window (1 disables smoothing)",
    )
    parser.add_argument(
        "--fog-agg-window",
        type=int,
        default=1,
        help="Fog-side aggregation window size (1 disables aggregation)",
    )
    parser.add_argument(
        "--fog-agg-func",
        type=str,
        default="mean",
        choices=["mean", "min", "max", "median"],
        help="Fog-side aggregation function",
    )
    parser.add_argument(
        "--fog-cache-ttl-seconds",
        type=float,
        default=0.0,
        help="Fog-side processed-series cache TTL in seconds (0 disables cache TTL)",
    )
    parser.add_argument(
        "--fog-offline-rate",
        type=float,
        default=0.0,
        help="Simulated probability that a fog node is offline per metric evaluation (0 disables)",
    )
    parser.add_argument(
        "--fog-byzantine-rate",
        type=float,
        default=0.0,
        help="Simulated probability that a fog node behaves byzantine per metric evaluation (0 disables)",
    )
    parser.add_argument(
        "--fog-byzantine-noise-std",
        type=float,
        default=0.0,
        help="Gaussian noise std added to processed values for byzantine nodes (0 disables)",
    )
    parser.add_argument(
        "--fog-consensus-max-rel-range",
        type=float,
        default=0.0,
        help="Relative score range threshold for consensus: (max-min)/|median|. 0 disables consensus range check.",
    )
    parser.add_argument(
        "--fog-enforce-consensus",
        action="store_true",
        help="If set, consensus violations fail the column (treated as contract failure)",
    )

    # --- System KPI computation model (optional) ---
    parser.add_argument(
        "--kpi",
        action="store_true",
        help="Compute key system validation KPIs (L_e2e, Th_sys, r, e, Tc, p, Ro) using a fog-filtering + confirmation model",
    )
    parser.add_argument("--kpi-batch-size", type=int, default=20, help="Batch size B used for fog forwarding filter in KPI model")
    parser.add_argument("--kpi-delta-percent", type=float, default=5.0, help="Fog variation threshold delta (percent) used in KPI model")
    parser.add_argument("--kpi-confirm-prob", type=float, default=0.98, help="Confirmation probability for submitted transactions")
    parser.add_argument("--kpi-tx-ms", type=float, default=20.0, help="Transmission delay per submitted tx (ms)")
    parser.add_argument("--kpi-tx-jitter-ms", type=float, default=0.0, help="Uniform jitter for tx delay (ms)")
    parser.add_argument("--kpi-conf-ms", type=float, default=50.0, help="Confirmation delay per submitted tx (ms)")
    parser.add_argument("--kpi-conf-jitter-ms", type=float, default=0.0, help="Uniform jitter for confirmation delay (ms)")
    parser.add_argument(
        "--kpi-cloud-baseline",
        action="store_true",
        help="Compute a cloud-only baseline reference (no fog filtering; all batches submitted) alongside KPI results",
    )
    parser.add_argument("--kpi-cloud-ms", type=float, default=0.0, help="Cloud processing delay per submitted tx in baseline (ms)")
    parser.add_argument("--kpi-cloud-jitter-ms", type=float, default=0.0, help="Uniform jitter for cloud processing delay in baseline (ms)")
    parser.add_argument("--kpi-c-device-unit", type=float, default=0.001, help="Unit cost per transmitted batch (devices)")
    parser.add_argument("--kpi-c-fog-unit", type=float, default=0.001, help="Unit cost per transmitted batch (fog)")
    parser.add_argument(
        "--kpi-c-cloud-unit",
        type=float,
        default=None,
        help="Unit cost per transmitted batch (cloud-only baseline). Defaults to --kpi-c-fog-unit if omitted.",
    )
    parser.add_argument("--kpi-c-sc-unit", type=float, default=0.02, help="Unit cost per smart contract execution")
    parser.add_argument("--kpi-c-conf-unit", type=float, default=0.01, help="Unit cost per confirmed transaction")
    parser.add_argument(
        "--kpi-bytes-per-point",
        type=float,
        default=8.0,
        help="Bytes per data point for throughput conversion; Th_sys is reported in MB/s using this factor (default: 8)",
    )
    parser.add_argument(
        "--kpi-out-json",
        type=Path,
        default=None,
        help="Write KPI model results to JSON (default: auto-named kpi_summary_*.json)",
    )
    parser.add_argument(
        "--kpi-out-csv",
        type=Path,
        default=None,
        help="Optional: also write flattened KPI summary to CSV",
    )
    parser.add_argument(
        "--kpi-aggregate",
        action="store_true",
        help="Also write a one-row aggregate KPI CSV across all validated metric columns",
    )
    parser.add_argument(
        "--kpi-aggregate-out-csv",
        type=Path,
        default=None,
        help="Output path for aggregate KPI CSV (default: auto-named kpi_aggregate_*.csv)",
    )
    parser.add_argument(
        "--fog-store-dir",
        type=Path,
        default=None,
        help="Optional directory to persist fog audit logs as JSONL per node",
    )
    parser.add_argument(
        "--fog-store-max-items",
        type=int,
        default=10_000,
        help="Max in-memory fog records per node (ring buffer)",
    )
    parser.add_argument("--no-sampling", action="store_true", help="Disable IoMT sampling integration test")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Write report to file (default: auto-named .txt in current directory)",
    )
    args = parser.parse_args()

    checks = _run_checks(int(args.seed))
    if not bool(args.no_sampling):
        checks.extend(
            _run_iomt_sampling_checks(
                sampling_csv=Path(args.sampling_csv),
                sampling_limit=int(args.sampling_limit),
                sampling_chunksize=int(args.sampling_chunksize),
                time_col=str(args.sampling_time_col),
                metric_col=str(args.sampling_metric_col),
                all_cols=bool(args.sampling_all_cols),
                threshold_auto=bool(args.sampling_threshold_auto) or bool(args.sampling_all_cols),
                threshold=float(args.sampling_threshold),
                sigma=int(args.sampling_sigma),
                epsilon=float(args.sampling_epsilon),
                alpha=float(args.sampling_alpha),
                contract_attainment_min=float(args.sampling_contract_attainment_min),
                incentive_attainment_min=float(args.sampling_incentive_attainment_min),
                gamma=float(args.sampling_gamma),
                proof_mode=str(args.contract_proof_mode),
                required_sigs=int(args.contract_required_sigs),
                fog_smooth_window=int(args.fog_smooth_window),
                fog_store_dir=args.fog_store_dir,
                fog_store_max_items=int(args.fog_store_max_items),
                fog_agg_window=int(args.fog_agg_window),
                fog_agg_func=str(args.fog_agg_func),
                fog_cache_ttl_seconds=float(args.fog_cache_ttl_seconds),
                fog_offline_rate=float(args.fog_offline_rate),
                fog_byzantine_rate=float(args.fog_byzantine_rate),
                fog_byzantine_noise_std=float(args.fog_byzantine_noise_std),
                fog_consensus_max_rel_range=float(args.fog_consensus_max_rel_range),
                fog_enforce_consensus=bool(args.fog_enforce_consensus),
                seed=int(args.seed),
            )
        )

    checks.extend(_run_kpi_model(args))
    report = _render_report(checks)

    out_path = args.out
    if out_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"algorithm_reliability_feasibility_report_{ts}.txt"

    Path(out_path).write_text(report, encoding="utf-8")
    print(report)

    return 0 if all(c.ok for c in checks) else 2


if __name__ == "__main__":
    raise SystemExit(main())
