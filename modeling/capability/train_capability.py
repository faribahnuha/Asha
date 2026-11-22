"""
==============================================================================
MODULE: train_capability.py              (offline training + evaluation, not a request stage)
------------------------------------------------------------------------------
- HOW THIS RELATES TO THE LAST FILE: uses capability_model.py's feature builder on
  dataset/gold/treatment_sites.csv and the labels in labels/capability_labels.csv; writes
  capability_model.json (read by capability_model.load_model), dataset/gold/capability_scores.csv,
  and modeling/capability/evaluation_report.{json,md}.
- WHAT THIS FILE DOES: fits one shared L2-regularised logistic regression across all 8 capabilities,
  picks the penalty by facility-grouped 5-fold cross-validation, reports out-of-fold accuracy
  metrics against two baselines, and tests the scores against two independent sources the model
  never saw (government blood bank directory links; LaQshya maternity certification).
- KNOWN LIMITATIONS / ASSUMPTIONS: 218 labels made by an AI from record text (not real-world truth);
  the penalty is chosen on the same folds used for reporting (slightly optimistic); population
  metrics reweight the stratified sample by stratum size; directory/LaQshya links only confirm
  "yes", so they test ranking (do confirmed places score higher?), not precision.
==============================================================================
"""

from __future__ import annotations

import csv
import datetime
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capability_model as cm  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
LAMBDA_GRID: tuple[float, ...] = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
N_FOLDS: int = 5
N_BOOTSTRAP: int = 500
MAX_NEWTON_ITERS: int = 50
SEED: int = 20261122
ECE_BINS: int = 5


def load_inputs() -> tuple[dict[str, dict[str, str]], list[dict[str, str]]]:
    """Gold sites keyed by id, and the decided (non-'unclear') labels."""
    sites = {r["facility_id"]: r for r in csv.DictReader(open(ROOT / "dataset/gold/treatment_sites.csv", encoding="utf-8"))}
    labels = [r for r in csv.DictReader(open(ROOT / "labels/capability_labels.csv", encoding="utf-8")) if r["label"] in ("0", "1")]
    assert len(sites) > 1000, "treatment_sites.csv looks empty"
    assert all(r["facility_id"] in sites for r in labels), "every label must refer to a gold site"
    return sites, labels


def build_dataset(sites: dict[str, dict[str, str]], labels: list[dict[str, str]]) -> dict[str, np.ndarray]:
    """Feature matrix, targets, facility groups, capability names and population weights."""
    assert len(labels) >= 50, "too few labels to train"
    x = np.vstack([cm.build_features(sites[r["facility_id"]], r["capability"]) for r in labels])
    y = np.array([int(r["label"]) for r in labels], dtype=float)
    counts: dict[tuple[str, str], int] = {}
    for r in labels:                                   # bounded by len(labels)
        counts[(r["capability"], r["stratum"])] = counts.get((r["capability"], r["stratum"]), 0) + 1
    w = np.array([int(r["stratum_size"]) / counts[(r["capability"], r["stratum"])] for r in labels])
    data = dict(x=x, y=y, w=w, groups=np.array([r["facility_id"] for r in labels]),
                caps=np.array([r["capability"] for r in labels]),
                level=np.array([int(sites[r["facility_id"]][f"cap_{r['capability']}_level"]) for r in labels]))
    assert x.shape[0] == y.shape[0] == w.shape[0], "one row per label"
    return data


def fit_logreg(x: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """L2-penalised logistic regression by Newton's method (intercept not penalised)."""
    assert x.shape[0] == y.shape[0] and lam > 0, "bad inputs"
    means, stds = x.mean(axis=0), x.std(axis=0)
    xs = np.hstack([np.ones((x.shape[0], 1)), cm.standardize(x, means, stds)])
    beta = np.zeros(xs.shape[1])
    penalty = np.full(xs.shape[1], lam)
    penalty[0] = 0.0
    for _ in range(MAX_NEWTON_ITERS):                  # fixed upper bound
        p = cm.sigmoid(xs @ beta)
        grad = xs.T @ (p - y) + penalty * beta
        hess = (xs * (p * (1 - p))[:, None]).T @ xs + np.diag(penalty) + 1e-9 * np.eye(len(beta))
        step = np.linalg.solve(hess, grad)
        beta -= step
        if np.abs(step).max() < 1e-8:
            break
    assert np.isfinite(beta).all(), "coefficients must be finite"
    return beta[1:], float(beta[0]), means, stds


def predict(x: np.ndarray, coef: np.ndarray, intercept: float, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    """Probabilities from fitted parameters."""
    assert x.shape[1] == len(coef), "feature count mismatch"
    p = cm.sigmoid(cm.standardize(x, means, stds) @ coef + intercept)
    assert p.shape == (x.shape[0],), "one probability per row"
    return p


def grouped_folds(groups: np.ndarray) -> np.ndarray:
    """Assign each facility (all its labels together) to one of N_FOLDS folds."""
    assert len(groups) > N_FOLDS, "need more rows than folds"
    uniq = np.unique(groups)
    rng = np.random.default_rng(SEED)
    fold_of = dict(zip(uniq, rng.permutation(len(uniq)) % N_FOLDS))
    folds = np.array([fold_of[g] for g in groups])
    assert set(folds.tolist()) == set(range(N_FOLDS)), "every fold must be used"
    return folds


def out_of_fold(data: dict[str, np.ndarray], folds: np.ndarray, lam: float) -> np.ndarray:
    """Out-of-fold P(capable): each row is predicted by a model that never saw its facility."""
    assert len(folds) == len(data["y"]), "fold per row"
    oof = np.zeros(len(data["y"]))
    for k in range(N_FOLDS):                           # fixed upper bound
        tr, te = folds != k, folds == k
        coef, b0, mu, sd = fit_logreg(data["x"][tr], data["y"][tr], lam)
        oof[te] = predict(data["x"][te], coef, b0, mu, sd)
    assert ((oof > 0) & (oof < 1)).all(), "probabilities must be strictly inside (0, 1)"
    return oof


def level_rate_baseline(data: dict[str, np.ndarray], folds: np.ndarray) -> np.ndarray:
    """Baseline: P = share of true claims at the same evidence level, learned on the other folds."""
    assert len(folds) == len(data["level"]), "fold per row"
    oof = np.zeros(len(data["y"]))
    for k in range(N_FOLDS):                           # fixed upper bound
        tr, te = folds != k, folds == k
        rates = {lv: (data["y"][tr][data["level"][tr] == lv].sum() + 1) / ((data["level"][tr] == lv).sum() + 2) for lv in range(4)}
        oof[te] = [rates[int(lv)] for lv in data["level"][te]]
    assert ((oof > 0) & (oof < 1)).all(), "smoothed rates are strictly inside (0, 1)"
    return oof


def auc(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None) -> float:
    """Weighted ROC AUC via the rank-sum identity (ties count half)."""
    assert len(y) == len(p), "length mismatch"
    w = np.ones(len(y)) if w is None else w
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    diff = p[pos][:, None] - p[neg][None, :]
    pair_w = w[pos][:, None] * w[neg][None, :]
    value = float(((diff > 0) + 0.5 * (diff == 0)).astype(float).__mul__(pair_w).sum() / pair_w.sum())
    assert 0.0 <= value <= 1.0, "AUC must be in [0, 1]"
    return value


def calibration(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> tuple[float, list[dict[str, float]]]:
    """Expected calibration error over equal-width bins, plus the reliability table."""
    assert len(y) == len(p) == len(w), "length mismatch"
    edges = np.linspace(0, 1, ECE_BINS + 1)
    table, ece = [], 0.0
    for i in range(ECE_BINS):                          # fixed upper bound
        m = (p >= edges[i]) & ((p < edges[i + 1]) if i < ECE_BINS - 1 else (p <= 1))
        if m.sum() == 0:
            continue
        pred, obs = float(np.average(p[m], weights=w[m])), float(np.average(y[m], weights=w[m]))
        ece += w[m].sum() / w.sum() * abs(pred - obs)
        table.append(dict(bin=f"{edges[i]:.1f}-{edges[i + 1]:.1f}", n=int(m.sum()), mean_predicted=round(pred, 3), observed=round(obs, 3)))
    assert 0.0 <= ece <= 1.0, "ECE must be in [0, 1]"
    return float(ece), table


def scores(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> dict[str, float]:
    """AUC, Brier, log loss, accuracy at 0.5 and calibration error (all weighted by w)."""
    assert len(y) == len(p) == len(w), "length mismatch"
    eps = 1e-12
    out = dict(auc=auc(y, p, w), brier=float(np.average((p - y) ** 2, weights=w)),
               log_loss=float(np.average(-(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)), weights=w)),
               accuracy_at_0_5=float(np.average((p >= 0.5) == (y == 1), weights=w)), ece=calibration(y, p, w)[0])
    assert all(np.isfinite(v) or k == "auc" for k, v in out.items()), "metrics must be finite"
    return {k: round(v, 3) for k, v in out.items()}


def bootstrap_auc(data: dict[str, np.ndarray], p: np.ndarray) -> tuple[float, float]:
    """95% interval for AUC, resampling facilities (not rows) with replacement."""
    assert len(p) == len(data["y"]), "length mismatch"
    rng = np.random.default_rng(SEED)
    uniq = np.unique(data["groups"])
    idx_of = {g: np.where(data["groups"] == g)[0] for g in uniq}
    vals = []
    for _ in range(N_BOOTSTRAP):                       # fixed upper bound
        rows = np.concatenate([idx_of[g] for g in rng.choice(uniq, size=len(uniq))])
        vals.append(auc(data["y"][rows], p[rows]))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    assert lo <= hi, "interval bounds out of order"
    return round(float(lo), 3), round(float(hi), 3)


def independent_checks(sites: dict[str, dict[str, str]], probs: dict[str, np.ndarray]) -> dict[str, object]:
    """Do sources the model never saw agree with its scores?"""
    assert "blood_service_claim" in probs, "blood probabilities required"
    ids = list(sites)
    linked = np.array([sites[i]["in_house_directory_level"] not in ("0", "") for i in ids])
    claim = np.array([sites[i]["cap_blood_service_claim_level"] != "0" for i in ids])
    pb = probs["blood_service_claim"]
    public = np.array([sites[i]["operator"] != "private" for i in ids])
    laq = np.array([sites[i]["laqshya_mot_ever"] == "True" or sites[i]["laqshya_lr_ever"] == "True" for i in ids])
    out = dict(
        blood_directory=dict(
            note="Hospitals linked to a licensed government blood bank vs not, among hospitals whose text claims blood",
            n_claiming=int(claim.sum()), n_claiming_and_linked=int((claim & linked).sum()),
            auc_linked_vs_not=round(auc(linked[claim].astype(float), pb[claim]), 3),
            mean_p_linked=round(float(pb[claim & linked].mean()), 3), mean_p_not_linked=round(float(pb[claim & ~linked].mean()), 3),
            linked_but_text_silent=int((linked & ~claim).sum())),
        laqshya=dict(
            note="LaQshya-certified public hospitals vs other public/unknown-ownership hospitals",
            n_public=int(public.sum()), n_certified=int((public & laq).sum()),
            auc_obstetric_care=round(auc(laq[public].astype(float), probs["obstetric_care"][public]), 3),
            mean_p_obstetric_certified=round(float(probs["obstetric_care"][public & laq].mean()), 3),
            mean_p_obstetric_other=round(float(probs["obstetric_care"][public & ~laq].mean()), 3),
            auc_surgical_capacity=round(auc(laq[public].astype(float), probs["surgical_capacity"][public]), 3)))
    assert out["blood_directory"]["n_claiming"] > 0, "no blood claims found"
    return out


def write_outputs(model: dict[str, object], report: dict[str, object], sites: dict[str, dict[str, str]], probs: dict[str, np.ndarray]) -> None:
    """Model JSON, per-site scores and the evaluation report."""
    assert len(probs) == len(cm.CAPABILITIES), "one probability vector per capability"
    assert all(len(v) == len(sites) for v in probs.values()), "one probability per site"
    cm.MODEL_PATH.write_text(json.dumps(model, indent=2))
    (Path(__file__).resolve().parent / "evaluation_report.json").write_text(json.dumps(report, indent=2))
    with open(ROOT / "dataset/gold/capability_scores.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["facility_id"] + [f"p_{c}" for c in cm.CAPABILITIES])
        for i, fid in enumerate(sites):                # bounded by len(sites)
            wr.writerow([fid] + [round(float(probs[c][i]), 4) for c in cm.CAPABILITIES])


def main() -> dict[str, object]:
    """Train, evaluate, score every site, write outputs."""
    sites, labels = load_inputs()
    data = build_dataset(sites, labels)
    folds = grouped_folds(data["groups"])
    oof_by_lam = {lam: out_of_fold(data, folds, lam) for lam in LAMBDA_GRID}
    ll = {lam: scores(data["y"], p, np.ones(len(p)))["log_loss"] for lam, p in oof_by_lam.items()}
    lam = min(ll, key=ll.get)
    oof, base = oof_by_lam[lam], level_rate_baseline(data, folds)
    keyword = (data["level"] > 0).astype(float) * 0.98 + 0.01      # "any mention = capable"
    coef, b0, mu, sd = fit_logreg(data["x"], data["y"], lam)
    probs = {c: predict(np.vstack([cm.build_features(s, c) for s in sites.values()]), coef, b0, mu, sd) for c in cm.CAPABILITIES}
    per_cap = {c: dict(n=int((data["caps"] == c).sum()), positives=int(data["y"][data["caps"] == c].sum()),
                       auc_model=round(auc(data["y"][data["caps"] == c], oof[data["caps"] == c]), 3),
                       auc_level_baseline=round(auc(data["y"][data["caps"] == c], base[data["caps"] == c]), 3)) for c in cm.CAPABILITIES}
    report = dict(trained_at=datetime.datetime.now().isoformat(timespec="seconds"), n_labels=len(labels),
                  n_unclear_excluded=sum(1 for r in csv.DictReader(open(ROOT / "labels/capability_labels.csv", encoding="utf-8")) if r["label"] == "unclear"),
                  lambda_chosen=lam, cv_log_loss_by_lambda=ll,
                  sample_metrics=dict(model=scores(data["y"], oof, np.ones(len(oof))), level_baseline=scores(data["y"], base, np.ones(len(oof))),
                                      keyword_baseline=scores(data["y"], keyword, np.ones(len(oof)))),
                  population_weighted_metrics=dict(model=scores(data["y"], oof, data["w"]), level_baseline=scores(data["y"], base, data["w"]),
                                                   keyword_baseline=scores(data["y"], keyword, data["w"])),
                  model_auc_95ci=bootstrap_auc(data, oof), reliability_table=calibration(data["y"], oof, np.ones(len(oof)))[1],
                  per_capability=per_cap, independent_checks=independent_checks(sites, probs),
                  coefficients=dict(sorted(zip(cm.FEATURE_NAMES, np.round(coef, 3).tolist()), key=lambda kv: -abs(kv[1]))))
    model = dict(feature_names=list(cm.FEATURE_NAMES), coef=coef.tolist(), intercept=b0, means=mu.tolist(), stds=sd.tolist(),
                 lambda_l2=lam, trained_at=report["trained_at"], n_labels=len(labels))
    write_outputs(model, report, sites, probs)
    assert report["sample_metrics"]["model"]["auc"] > 0.5, "model must beat chance out of fold"
    assert cm.load_model()["n_labels"] == len(labels), "saved model must round-trip"
    return report


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
