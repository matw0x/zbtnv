"""Static figures for the written report (light theme, print-safe, Russian captions)."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import precision_recall_curve, roc_curve

from meshguard.detection.datasets import LabeledRun
from meshguard.detection.detector import AnomalyDetector
from meshguard.detection.evaluation import Evaluation
from meshguard.detection.features import FEATURE_NAMES

SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
DASHES = ("-", "--", "-.", ":", (0, (5, 1, 1, 1)))
INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
NEUTRAL = "#f0efec"
DIVERGING = LinearSegmentedColormap.from_list("blue_red", ["#184f95", NEUTRAL, "#b52b2b"])
SEQUENTIAL = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#86b6ef", "#2a78d6", "#0d366b"])

SCENARIO_TITLES = {
    "ddos": "DDoS (HTTP flood)",
    "brute_force": "Подбор паролей",
    "software_fault": "Программный сбой",
    "data_exfiltration": "Утечка данных",
    "benign": "Легитимный всплеск",
    "ddos_residential": "DDoS с домашних IP",
    "brute_force_distributed": "Распределённый подбор",
    "software_fault_silent": "Тихая деградация",
    "data_exfiltration_relay": "Утечка через релей",
    "benign_campaign": "Рекламная кампания",
    "missed": "Не обнаружено",
}

CLASS_TITLES = {
    "ddos": "DDoS",
    "brute_force": "Подбор паролей",
    "data_exfiltration": "Утечка данных",
    "software_fault": "Программный сбой",
    "benign": "Легитимная аномалия",
    "missed": "Не обнаружено",
}

FEATURE_TITLES = {
    "requests": "Число запросов",
    "server_error_rate": "Доля 5xx",
    "client_error_rate": "Доля 4xx",
    "auth_failure_rate": "Доля 401/403",
    "latency_p50_ms": "Задержка p50",
    "latency_p95_ms": "Задержка p95",
    "bytes_per_response": "Размер ответа",
    "outbound_bytes": "Исходящий объём",
    "unique_clients": "Уникальные IP",
    "top_client_share": "Доля топ-IP",
    "users_per_client": "Пользователей на IP",
    "novel_route_ratio": "Доля новых маршрутов",
    "hour_sin": "Время суток (sin)",
    "hour_cos": "Время суток (cos)",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_MUTED,
            "axes.titlecolor": INK,
            "axes.titlesize": 10,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "legend.frameon": False,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


def roc_pr_curves(evaluation: Evaluation, path: Path) -> Path:
    _style()
    fig, (roc_ax, pr_ax) = plt.subplots(1, 2, figsize=(9, 3.8))
    for i, report in enumerate(evaluation.reports):
        y_true, y_score = evaluation.curves[report.name]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        style = {"color": SERIES[i], "linestyle": DASHES[i], "linewidth": 2}
        roc_ax.plot(fpr, tpr, label=f"{report.name} (AUC {report.roc_auc:.3f})", **style)
        pr_ax.plot(recall, precision, label=f"{report.name} (AP {report.pr_auc:.3f})", **style)
    roc_ax.plot([0, 1], [0, 1], color=GRID, linewidth=1)
    roc_ax.set(xlabel="Доля ложных срабатываний (FPR)", ylabel="Полнота (TPR)", title="ROC-кривые")
    pr_ax.set(xlabel="Полнота (recall)", ylabel="Точность (precision)", title="PR-кривые")
    for ax in (roc_ax, pr_ax):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(loc="lower right" if ax is roc_ax else "lower left", fontsize=8)
    return _save(fig, path)


def score_timelines(detector: AnomalyDetector, runs: Sequence[LabeledRun], path: Path) -> Path:
    _style()
    fig, axes = plt.subplots(len(runs), 1, figsize=(9, 1.7 * len(runs)), sharex=True)
    for ax, run in zip(np.atleast_1d(axes), runs, strict=True):
        windows = run.windows
        scores = detector.score_many(windows)
        start = windows[0].start
        minutes = np.array([(w.end - start).total_seconds() / 60 for w in windows])
        by_time: dict[float, float] = {}
        for t, s in zip(minutes, scores, strict=True):
            by_time[t] = max(by_time.get(t, -np.inf), float(s))
        t = np.array(sorted(by_time))
        ax.axvspan(
            (run.scenario.start - start).total_seconds() / 60,
            (run.scenario.end - start).total_seconds() / 60,
            color=NEUTRAL,
            zorder=0,
            label="Инцидент",
        )
        ax.plot(t, [by_time[x] for x in t], color=SERIES[0], linewidth=1.6, label="Макс. оценка")
        ax.axhline(detector.threshold, color=INK_MUTED, linestyle="--", linewidth=1, label="Порог")
        ax.set_title(SCENARIO_TITLES[run.scenario.name], loc="left")
        ax.set_ylabel("Оценка")
        ax.set_ylim(top=0.4)
    panels = np.atleast_1d(axes)
    panels[-1].set_xlabel("Время от начала прогона, мин")
    handles, labels = panels[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(hspace=0.6, top=0.93)
    return _save(fig, path)


def deviation_heatmap(
    detector: AnomalyDetector, runs: Sequence[LabeledRun], path: Path, clip: float = 10.0
) -> Path:
    _style()
    kinds = list(dict.fromkeys(run.scenario.name for run in runs))
    features = [f for f in FEATURE_NAMES if not f.startswith("hour_")]
    matrix = np.zeros((len(features), len(kinds)))
    for j, kind in enumerate(kinds):
        rows = [
            detector.profile.standardize(w)
            for run in runs
            if run.scenario.name == kind
            for w, positive in zip(run.windows, run.labels, strict=True)
            if positive
        ]
        mean = np.mean(rows, axis=0)
        matrix[:, j] = [mean[FEATURE_NAMES.index(f)] for f in features]
    shown = np.clip(matrix, -clip, clip)
    fig, ax = plt.subplots(figsize=(7, 5.2))
    ax.grid(False)
    image = ax.imshow(shown, cmap=DIVERGING, vmin=-clip, vmax=clip, aspect="auto")
    ax.set_xticks(range(len(kinds)), [SCENARIO_TITLES[k] for k in kinds], rotation=20, ha="right")
    ax.set_yticks(range(len(features)), [FEATURE_TITLES[f] for f in features])
    for i in range(len(features)):
        for j in range(len(kinds)):
            value = matrix[i, j]
            ax.text(
                j,
                i,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=7,
                color=SURFACE if abs(shown[i, j]) > clip * 0.6 else INK,
            )
    bar = fig.colorbar(image, ax=ax, shrink=0.8)
    bar.set_label(f"Средний робастный z-score (обрезан до ±{clip:g})")
    bar.outline.set_visible(False)
    return _save(fig, path)


def confusion_heatmap(summary: dict[str, Any], path: Path, title: str) -> Path:
    _style()
    labels: list[str] = summary["labels"]
    matrix = np.array(summary["confusion"])
    keep = [i for i, label in enumerate(labels) if label != "missed" or matrix[:, i].sum()]
    rows = [i for i in keep if labels[i] != "missed"]
    matrix = matrix[np.ix_(rows, keep)]
    fig, ax = plt.subplots(figsize=(6, 4.6))
    ax.grid(False)
    ax.imshow(matrix, cmap=SEQUENTIAL, vmin=0, vmax=max(1, matrix.max()))
    ax.set_xticks(
        range(len(keep)), [CLASS_TITLES[labels[i]] for i in keep], rotation=25, ha="right"
    )
    ax.set_yticks(range(len(rows)), [CLASS_TITLES[labels[i]] for i in rows])
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                fontsize=9,
                color=SURFACE if matrix[i, j] > matrix.max() * 0.55 else INK,
            )
    return _save(fig, path)


def _save(fig: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
