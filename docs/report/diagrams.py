"""Block diagrams for the report, drawn with matplotlib to stay dependency-free."""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

type Point = tuple[float, float]

INK, MUTED, LINE = "#0b0b0b", "#52514e", "#8a8984"
FILL = {"ml": "#cde2fb", "agent": "#fbe0d3", "act": "#d4f1e4", "data": "#f0efec"}


def _box(
    ax: Axes, x: float, y: float, w: float, h: float, text: str, kind: str = "data", size: float = 9
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=FILL[kind],
            edgecolor=LINE,
            linewidth=1,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK)


def _arrow(
    ax: Axes,
    start: Point,
    end: Point,
    text: str = "",
    offset: Point = (0, 0.12),
    style: str = "-|>",
    dashed: bool = False,
    curve: float = 0.0,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=12,
            color=MUTED,
            linewidth=1.1,
            linestyle="--" if dashed else "-",
            shrinkA=2,
            shrinkB=2,
            connectionstyle=f"arc3,rad={curve}",
        )
    )
    if text:
        mx, my = (start[0] + end[0]) / 2 + offset[0], (start[1] + end[1]) / 2 + offset[1]
        ax.text(mx, my, text, ha="center", va="bottom", fontsize=7.5, color=MUTED)


def _canvas(width: float, height: float) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, width)
    ax.set_ylim(-0.25, height)
    ax.axis("off")
    plt.rcParams["font.family"] = "DejaVu Sans"
    return fig, ax


def architecture(path: Path) -> Path:
    fig, ax = _canvas(10, 6.2)
    _box(ax, 0.2, 4.9, 1.9, 0.9, "Микросервисы\n(журналы\nвзаимодействия)")
    _box(ax, 2.6, 4.9, 1.9, 0.9, "Агрегатор окон\n10 с × сервис,\n14 признаков", "ml")
    _box(ax, 5.0, 4.9, 2.0, 0.9, "Детектор\nробастная норм. +\nOne-Class SVM", "ml")
    _box(ax, 7.5, 4.9, 2.3, 0.9, "Гейт сигналов\n2 из 3 окон,\nкорреляция инцидента", "ml")
    _box(ax, 0.2, 3.1, 1.9, 0.9, "Хранилище\nжурналов\n(1 ч)")
    _box(
        ax,
        3.4,
        2.8,
        3.2,
        1.5,
        "LLM-агент\nцикл tool calling:\nрасследование → классификация →\nсдерживание → отчёт",
        "agent",
        9.5,
    )
    _box(ax, 7.5, 3.1, 2.3, 0.9, "Сигнал аномалии\n+ объяснение\n(отклонения признаков)", "ml", 8)
    _box(
        ax,
        0.2,
        1.1,
        2.6,
        1.0,
        "Инструменты\nрасследования\nsearch_logs, top_clients,\nservice_health, call_graph",
        "agent",
        8,
    )
    _box(
        ax,
        3.4,
        0.9,
        3.2,
        1.2,
        "Политика-ограничитель\nматрица действий, уверенность,\n"
        "защищённые сети,\nкритичные сервисы",
        "act",
        8.5,
    )
    _box(ax, 7.2, 1.55, 2.6, 0.7, "Межсетевой экран\n(блокировка IP)", "act", 8.5)
    _box(ax, 7.2, 0.75, 2.6, 0.7, "Сервисная сеть\n(изоляция сервиса)", "act", 8.5)
    _box(ax, 7.2, -0.05, 2.6, 0.7, "Отчёт администратору", "act", 8.5)
    _arrow(ax, (2.1, 5.35), (2.6, 5.35))
    _arrow(ax, (4.5, 5.35), (5.0, 5.35))
    _arrow(ax, (7.0, 5.35), (7.5, 5.35))
    _arrow(ax, (8.65, 4.9), (8.65, 4.0))
    _arrow(ax, (7.5, 3.55), (6.6, 3.55))
    _arrow(ax, (1.15, 4.9), (1.15, 4.0))
    ax.text(1.25, 4.45, "запись", fontsize=7.5, color=MUTED, va="center")
    _arrow(ax, (1.15, 3.1), (1.15, 2.1), style="<|-|>")
    _arrow(ax, (3.4, 3.2), (2.8, 1.9), style="<|-|>")
    _arrow(ax, (5.0, 2.8), (5.0, 2.1))
    ax.text(
        5.1,
        2.4,
        "block_ips, isolate_service,\nnotify_admin",
        fontsize=7.5,
        color=MUTED,
        va="center",
    )
    _arrow(ax, (6.6, 1.6), (7.2, 1.9))
    _arrow(ax, (6.6, 1.4), (7.2, 1.1))
    _arrow(ax, (6.6, 1.2), (7.2, 0.3))
    _arrow(ax, (9.9, 1.9), (9.9, 4.9), dashed=True)
    ax.text(9.95, 2.9, "обратная\nсвязь", fontsize=7.5, color=MUTED, rotation=90, va="center")
    ax.text(0.2, 6.0, "Этап 1. Статистическое обнаружение", fontsize=9, color=MUTED)
    ax.text(3.4, 4.45, "Этап 2. Расследование", fontsize=9, color=MUTED)
    ax.text(0.2, 0.25, "Этап 3. Реагирование", fontsize=9, color=MUTED)
    return _save(fig, path)


def topology(path: Path) -> Path:
    fig, ax = _canvas(10.2, 4.6)
    nodes = {
        "Клиенты": (0.2, 2.0, "data"),
        "gateway": (1.9, 2.0, "ml"),
        "catalog": (3.9, 3.6, "ml"),
        "auth": (3.9, 2.5, "ml"),
        "users": (6.0, 2.5, "ml"),
        "orders": (3.9, 1.0, "ml"),
        "inventory": (6.0, 3.6, "ml"),
        "payments": (6.0, 1.0, "ml"),
        "notifications": (6.0, 0.0, "ml"),
        "ПСП\n198.51.100.40": (8.5, 1.0, "act"),
        "Почтовый релей\n198.51.100.25": (8.5, 0.0, "act"),
    }
    w, h = 1.45, 0.7
    for name, (x, y, kind) in nodes.items():
        _box(ax, x, y, w, h, name, kind, 8.5)

    def link(a: str, b: str) -> None:
        ax_, ay, _ = nodes[a]
        bx, by, _ = nodes[b]
        start = (ax_ + w, ay + h / 2) if bx > ax_ else (ax_ + w / 2, ay + (h if by > ay else 0))
        end = (bx, by + h / 2) if bx > ax_ else (bx + w / 2, by + (0 if by > ay else h))
        _arrow(ax, start, end)

    for a, b in [
        ("Клиенты", "gateway"),
        ("gateway", "catalog"),
        ("gateway", "auth"),
        ("gateway", "orders"),
        ("catalog", "inventory"),
        ("auth", "users"),
        ("orders", "inventory"),
        ("orders", "payments"),
        ("orders", "notifications"),
        ("payments", "ПСП\n198.51.100.40"),
        ("notifications", "Почтовый релей\n198.51.100.25"),
    ]:
        link(a, b)
    _arrow(ax, (3.35, 2.45), (6.0, 2.6))
    _arrow(ax, (6.0 + w, 0.45), (6.0 + w, 2.65), dashed=True, curve=0.45)
    ax.text(8.0, 1.85, "контакты", fontsize=7.5, color=MUTED)
    return _save(fig, path)


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
