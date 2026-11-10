"""Build the explanatory note (.docx) following GOST 7.32 and the department guidelines.

Usage: uv run --group report python docs/report/build.py
"""

import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from docx import Document
from docx.enum.section import WD_SECTION_START
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import (
    WD_ALIGN_PARAGRAPH,
    WD_BREAK,
    WD_LINE_SPACING,
    WD_TAB_ALIGNMENT,
    WD_TAB_LEADER,
)
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Mm, Pt, RGBColor
from docx.text.paragraph import Paragraph

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUILD = HERE / "build"
REPORTS = ROOT / "reports"
OUTPUT = HERE / "Пояснительная записка.docx"
sys.path[:0] = [str(HERE), str(ROOT / "src")]

import diagrams  # noqa: E402

from meshguard.agent.investigator import Investigator  # noqa: E402
from meshguard.domain import Severity  # noqa: E402
from meshguard.reporting.plots import SCENARIO_TITLES, confusion_heatmap  # noqa: E402
from meshguard.response.actions import AdminReport  # noqa: E402
from meshguard.response.policy import ResponsePolicy  # noqa: E402

FONT = "Times New Roman"
PARAGRAPH_INDENT = Cm(1.25)
TEXT_WIDTH = Mm(210 - 30 - 15)
STRUCTURAL = {"ВВЕДЕНИЕ", "ЗАКЛЮЧЕНИЕ", "СПИСОК ИСПОЛЬЗОВАННЫХ ИСТОЧНИКОВ", "СОДЕРЖАНИЕ"}
MATH = "http://schemas.openxmlformats.org/officeDocument/2006/math"
SUITE_TITLES = {"core": "базовые сценарии", "hard": "усложнённые сценарии"}
FAILURE_NOTES = {
    "brute_force": (
        "при подборе паролей класс определён верно, но правило блокирует только адреса с тремя "
        "и более отказами за две минуты, поэтому заблокировано лишь {coverage} адресов-нарушителей"
    ),
    "ddos_residential": (
        "DDoS с домашних адресов принят за легитимный всплеск: правило ищет новые подсети /16, "
        "а ботнет находится в тех же подсетях, что и клиенты"
    ),
    "brute_force_distributed": (
        "распределённый подбор классифицирован верно, но ни один из 300 адресов не набрал "
        "порогового числа отказов, и блокировка не выполнена"
    ),
    "data_exfiltration_relay": (
        "утечка через почтовый релей принята за DDoS из-за роста трафика к users; агент "
        "попытался заблокировать внутреннюю подсеть 10.0.0.0/16, и политика отклонила действие"
    ),
}


def num(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def pct(value: float) -> str:
    return f"{value * 100:.0f} %"


def load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def pipe_table(caption: str, header: list[str], widths: list[int], rows: list[list[str]]) -> str:
    lines = [f"Table: {caption}", "", "| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join("-" * w for w in widths) + "|")
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def detector_table(metrics: dict[str, Any]) -> str:
    rows = [
        [
            r["name"],
            num(r["roc_auc"]),
            num(r["pr_auc"]),
            num(r["precision"]),
            num(r["recall"]),
            num(r["f1"]),
            num(r["false_alarms_per_hour"], 2),
            num(r["fit_seconds"], 1),
            f"{r['score_us_per_window']:.0f}",
        ]
        for r in metrics["reports"]
    ]
    return pipe_table(
        "Таблица 4.1 – Качество и производительность моделей обнаружения",
        [
            "Модель",
            "ROC-AUC",
            "PR-AUC",
            "Точность",
            "Полнота",
            "F1",
            "Ложных тревог в час",
            "Обучение, с",
            "Оценка окна, мкс",
        ],
        [18, 8, 8, 9, 9, 7, 10, 9, 9],
        rows,
    )


def detector_text(metrics: dict[str, Any]) -> str:
    best = max(metrics["reports"], key=lambda r: r["f1"])
    return (
        f"Наибольшие значения F1 ({num(best['f1'])}) и PR-AUC ({num(best['pr_auc'])}) при "
        f"откалиброванном пороге показал {best['name']}; эта модель выбрана для первого этапа. "
        "Выбор сделан по валидационным прогонам и подтверждён на тестовых."
    )


def delay_table(metrics: dict[str, Any]) -> str:
    reports = metrics["reports"]
    scenarios = list(reports[0]["detection_delay_s"])
    rows = [
        [SCENARIO_TITLES[s], *(num(r["detection_delay_s"][s], 1) for r in reports)]
        for s in scenarios
    ]
    return pipe_table(
        "Таблица 4.2 – Средняя задержка обнаружения инцидента, с",
        ["Сценарий", *(r["name"] for r in reports)],
        [24, 14, 14, 14, 14],
        rows,
    )


def agent_rows(summary: dict[str, Any]) -> Iterator[list[str]]:
    runs: dict[str, list[dict[str, Any]]] = {}
    for run in summary["runs"]:
        runs.setdefault(run["scenario"], []).append(run)
    for name, stats in summary["per_scenario"].items():
        predicted = sorted({r["predicted"] or "не обнаружено" for r in runs[name]})
        yield [
            SCENARIO_TITLES[name],
            runs[name][0]["expected"],
            ", ".join(predicted),
            pct(stats["accuracy"]),
            pct(stats["containment_correct"]),
        ]


def agent_table(model: str) -> tuple[str, str]:
    rows: list[list[str]] = []
    totals: dict[str, dict[str, Any]] = {}
    for suite in ("core", "hard"):
        summary = load(REPORTS / "agent" / model / suite / "summary.json")
        if summary is None:
            continue
        totals[suite] = summary
        rows += list(agent_rows(summary))
        rows.append(
            [
                f"**Итого: {SUITE_TITLES[suite]}**",
                "",
                "",
                f"**{pct(summary['accuracy'])}**",
                f"**{pct(summary['containment_correct'])}**",
            ]
        )
    table = pipe_table(
        "Таблица 4.3 – Результаты эвристического агента (по 3 прогона на сценарий)",
        [
            "Сценарий",
            "Истинный класс",
            "Предсказанный класс",
            "Точность",
            "Корректность реагирования",
        ],
        [22, 16, 18, 10, 14],
        rows,
    )
    return table, agent_text(totals)


def agent_text(totals: dict[str, dict[str, Any]]) -> str:
    core, hard = totals["core"], totals["hard"]
    notes = []
    for suite in (core, hard):
        for name, stats in suite["per_scenario"].items():
            if name in FAILURE_NOTES and (
                stats["accuracy"] < 1 or stats["containment_correct"] < 1
            ):
                coverage = [r["offenders_blocked"] for r in suite["runs"] if r["scenario"] == name]
                valid = [c for c in coverage if c is not None]
                span = f"{pct(min(valid))}–{pct(max(valid))}" if valid else "0 %"
                notes.append(FAILURE_NOTES[name].format(coverage=span))
    failures = "; ".join(notes)
    return (
        f"На базовых сценариях эвристика верно классифицировала {pct(core['accuracy'])} "
        f"инцидентов (macro-F1 {num(core['macro_f1'])}), однако корректное реагирование выполнено "
        f"лишь в {pct(core['containment_correct'])} случаев. На усложнённых сценариях точность "
        f"снизилась до {pct(hard['accuracy'])} (macro-F1 {num(hard['macro_f1'])}), а доля "
        f"корректного реагирования – до {pct(hard['containment_correct'])}. Причины ошибок: "
        f"{failures}. Ни в одном прогоне не заблокирован легитимный клиент: ошибочные действия "
        f"либо не выполнялись, либо были отклонены политикой "
        f"({hard['denied_actions_total'] + core['denied_actions_total']} отклонений). "
        "Матрица ошибок на усложнённых сценариях приведена на рисунке 4.4.\n\n"
        "![Рисунок 4.4 – Матрица ошибок эвристического агента на усложнённых сценариях]"
        "(agent/heuristic-baseline/hard/confusion.png){width=12cm}\n\n"
        "Результат показывает принципиальное ограничение правил: они кодируют признаки известных "
        "сценариев, а не причинно-следственную логику расследования. Агенту на основе LLM "
        "передаются те же данные, но решение он принимает, рассуждая над ними: например, может "
        "заметить, что новые адреса без истории обращений составляют основную долю трафика даже "
        "внутри «домашних» подсетей, или что сервис уведомлений внезапно стал читать контакты "
        "пользователей с частотой, несоразмерной числу заказов."
    )


def llm_block() -> str:
    candidates = sorted(
        p
        for p in (REPORTS / "agent").glob("*/*/summary.json")
        if not p.parts[-3].startswith("heuristic")
    )
    if not candidates:
        return (
            "Количественная оценка агента на основе LLM выполняется той же процедурой и на тех же "
            "сценариях командой `meshguard benchmark --provider anthropic --suite all`, "
            "фиксирующей точность классификации, корректность реагирования, число ходов, расход "
            "токенов и время расследования. В рамках данной работы прогон с платным API не "
            "выполнялся. Корректность интеграции с Claude API и OpenAI-совместимыми серверами "
            "подтверждена автоматическими тестами на подменённых клиентах: проверяются формат "
            "запроса (строгие схемы инструментов, уровень рассуждений, кэширование, резервная "
            "модель), сохранение истории диалога, разбор вызовов инструментов и обработка отказа "
            "модели. Сборщик пояснительной записки автоматически включает таблицу результатов "
            "LLM-агента, если результаты прогона присутствуют в каталоге reports/agent."
        )
    rows = []
    for path in candidates:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            [
                data["model"],
                SUITE_TITLES.get(path.parent.name, path.parent.name),
                pct(data["accuracy"]),
                num(data["macro_f1"]),
                pct(data["containment_correct"]),
                num(data["mean_turns"], 1),
                f"{data['mean_input_tokens']:.0f}",
                f"{data['mean_output_tokens']:.0f}",
                num(data["mean_llm_seconds"], 1),
            ]
        )
    return "Результаты агента на основе LLM приведены в таблице 4.4.\n\n" + pipe_table(
        "Таблица 4.4 – Результаты агента на основе LLM",
        [
            "Модель",
            "Набор",
            "Точность",
            "Macro-F1",
            "Корректность реагирования",
            "Ходов",
            "Входных токенов",
            "Выходных токенов",
            "Время, с",
        ],
        [16, 12, 9, 8, 12, 7, 9, 9, 8],
        rows,
    )


def example_incident() -> tuple[str, str]:
    for path in sorted(
        (REPORTS / "agent" / "heuristic-baseline" / "core" / "incidents").glob("*.json")
    ):
        incident = json.loads(path.read_text(encoding="utf-8"))
        classification = incident["classification"]
        if classification and classification["incident_type"] == "data_exfiltration":
            break
    else:
        raise SystemExit("run `meshguard benchmark --provider heuristic` first")
    header = {
        **{k: incident[k] for k in ("incident_id", "model", "outcome", "turns", "signal")},
        "classification": {k: classification[k] for k in ("incident_type", "confidence")},
        "blocked": incident["blocked"],
        "isolated": incident["isolated"],
    }

    def short(value: Any, limit: int) -> str:
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        text = text.replace("|", "/")
        return text if len(text) <= limit else text[: limit - 1] + "…"

    rows = [
        [a["at"][11:19], a["tool"], short(a["arguments"], 70), short(a["result"], 90)]
        for a in incident["audit"]
    ]
    table = pipe_table(
        "Таблица Б.1 – Вызовы инструментов в ходе расследования",
        ["Время", "Инструмент", "Аргументы", "Результат (сокращён)"],
        [11, 16, 30, 43],
        rows,
    )
    notify = next(a for a in incident["audit"] if a["tool"] == "notify_admin")["arguments"]
    rendered = AdminReport(
        incident["incident_id"],
        Severity(notify["severity"]),
        notify["title"],
        notify["summary"],
        tuple(notify["recommendations"]),
        datetime.fromisoformat(incident["audit"][-1]["at"]),
    ).render()
    summary = json.dumps(header, ensure_ascii=False, indent=1)
    return f"```\n{summary}\n```\n\n{table}", f"```\n{rendered}```"


def source(obj: Callable[..., Any] | type) -> str:
    return textwrap.dedent(inspect.getsource(obj)).rstrip()


def listing(obj: Callable[..., Any] | type) -> str:
    return f"```\n{source(obj)}\n```"


def render_markdown() -> str:
    metrics = load(REPORTS / "detection" / "metrics.json")
    if metrics is None:
        raise SystemExit("run `meshguard evaluate` first")
    agent, agent_comment = agent_table("heuristic-baseline")
    audit, report = example_incident()
    investigator = "\n\n".join(source(m) for m in (Investigator.investigate, Investigator._loop))
    replacements = {
        "{{table:detectors}}": detector_table(metrics),
        "{{text:detectors}}": detector_text(metrics),
        "{{table:delays}}": delay_table(metrics),
        "{{table:agent}}": agent,
        "{{text:agent}}": agent_comment,
        "{{block:llm}}": llm_block(),
        "{{example:audit}}": audit,
        "{{example:report}}": report,
        "{{listing:investigator}}": f"```\n{investigator}\n```",
        "{{listing:policy}}": listing(ResponsePolicy),
    }
    text = (HERE / "report.md").read_text(encoding="utf-8")
    for key, value in replacements.items():
        text = text.replace(key, value)
    if leftover := re.findall(r"\{\{[^}]+\}\}", text):
        raise SystemExit(f"unresolved placeholders: {leftover}")
    return text


def find_style(doc: Any, name: str, *, create: str | None = None) -> Any:
    for style in doc.styles:
        if style.name.lower() == name.lower():
            return style
    if create is None:
        raise KeyError(name)
    style = doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
    style.element.set(qn("w:styleId"), create)
    style.base_style = find_style(doc, "Normal")
    return style


def configure(
    doc: Any,
    name: str,
    *,
    size: float = 14,
    bold: bool | None = None,
    align: Any = WD_ALIGN_PARAGRAPH.JUSTIFY,
    indent: Any = PARAGRAPH_INDENT,
    spacing: Any = WD_LINE_SPACING.ONE_POINT_FIVE,
    before: float = 0,
    after: float = 0,
    font: str = FONT,
    keep: bool = False,
    page_break: bool = False,
    hyphenate: bool = True,
    create: str | None = None,
) -> None:
    style = find_style(doc, name, create=create)
    style.font.name = font
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.italic = False
    style.font.color.rgb = RGBColor(0, 0, 0)
    rpr = style.element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.append(fonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        fonts.set(qn(attr), font)
    for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
        fonts.attrib.pop(qn(attr), None)
    fmt = style.paragraph_format
    fmt.alignment = align
    fmt.first_line_indent = indent
    fmt.left_indent = Cm(0)
    fmt.line_spacing_rule = spacing
    fmt.space_before = Pt(before)
    fmt.space_after = Pt(after)
    fmt.widow_control = True
    fmt.keep_with_next = keep
    fmt.page_break_before = page_break
    if not hyphenate:
        style.element.get_or_add_pPr().append(OxmlElement("w:suppressAutoHyphens"))


def style_reference(path: Path) -> Path:
    with path.open("wb") as sink:
        subprocess.run(
            ["pandoc", "--print-default-data-file", "reference.docx"], stdout=sink, check=True
        )
    doc = Document(str(path))
    for name in ("Normal", "Body Text", "First Paragraph"):
        configure(doc, name)
    heading = {"bold": True, "align": WD_ALIGN_PARAGRAPH.LEFT, "keep": True, "hyphenate": False}
    configure(doc, "Heading 1", after=21, page_break=True, **heading)
    configure(doc, "Heading 2", before=21, after=21, **heading)
    configure(doc, "Heading 3", before=14, after=14, **heading)
    single = {"indent": Cm(0), "spacing": WD_LINE_SPACING.SINGLE}
    configure(doc, "Compact", size=12, align=WD_ALIGN_PARAGRAPH.LEFT, **single)
    configure(
        doc, "Image Caption", align=WD_ALIGN_PARAGRAPH.CENTER, after=21, hyphenate=False, **single
    )
    configure(
        doc,
        "Table Caption",
        align=WD_ALIGN_PARAGRAPH.LEFT,
        before=21,
        keep=True,
        hyphenate=False,
        **single,
    )
    for name in ("Captioned Figure", "Figure"):
        configure(doc, name, align=WD_ALIGN_PARAGRAPH.CENTER, indent=Cm(0), before=14, keep=True)
    configure(
        doc, "TOC Heading", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, indent=Cm(0), after=21
    )
    for level in (1, 2):
        configure(
            doc, f"toc {level}", align=WD_ALIGN_PARAGRAPH.LEFT, indent=Cm(0), create=f"TOC{level}"
        )
        toc = find_style(doc, f"toc {level}").paragraph_format
        toc.left_indent = Cm(0.5 * (level - 1))
        toc.tab_stops.add_tab_stop(TEXT_WIDTH, WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
    verbatim = find_style(doc, "Verbatim Char").font
    verbatim.name = "Courier New"
    verbatim.size = Pt(12)
    hyperlink = find_style(doc, "Hyperlink").font
    hyperlink.color.rgb = RGBColor(0, 0, 0)
    hyperlink.underline = False
    defaults = doc.styles.element.find(qn("w:docDefaults"))
    rpr = defaults.find(f"{qn('w:rPrDefault')}/{qn('w:rPr')}") if defaults is not None else None
    if rpr is not None:
        lang = rpr.find(qn("w:lang"))
        if lang is None:
            lang = OxmlElement("w:lang")
            rpr.append(lang)
        lang.set(qn("w:val"), "ru-RU")
        lang.set(qn("w:eastAsia"), "ru-RU")
    doc.save(str(path))
    return path


def run_pandoc(markdown: Path, reference: Path, target: Path) -> None:
    subprocess.run(
        [
            "pandoc",
            str(markdown),
            "-o",
            str(target),
            "--from",
            "markdown-smart+pipe_tables+tex_math_dollars+implicit_figures+link_attributes",
            "--reference-doc",
            str(reference),
            "--toc",
            "--toc-depth=2",
            "-M",
            "toc-title=СОДЕРЖАНИЕ",
            "--resource-path",
            f"{BUILD}:{REPORTS}",
            "--no-highlight",
        ],
        check=True,
    )


def add_field(paragraph: Paragraph, instruction: str) -> None:
    for kind, text in (("begin", None), (None, instruction), ("separate", None), ("end", None)):
        run = paragraph.add_run()
        if kind:
            element = OxmlElement("w:fldChar")
            element.set(qn("w:fldCharType"), kind)
        else:
            element = OxmlElement("w:instrText")
            element.set(qn("xml:space"), "preserve")
            element.text = text
        run._r.append(element)


def setup_page(doc: Any) -> None:
    section = doc.sections[0]
    section.start_type = WD_SECTION_START.NEW_PAGE
    section.page_width, section.page_height = Mm(210), Mm(297)
    section.left_margin, section.right_margin = Mm(30), Mm(15)
    section.top_margin, section.bottom_margin = Mm(20), Mm(20)
    section.footer_distance = Mm(10)
    section.different_first_page_header_footer = True
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.paragraph_format.first_line_indent = Cm(0)
    footer.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    add_field(footer, "PAGE")
    for run in footer.runs:
        run.font.name = FONT
        run.font.size = Pt(12)


def title_page(doc: Any) -> None:
    body = doc.element.body
    anchor = body[0]
    lines: list[tuple[str, dict[str, Any]]] = [
        ("МИНОБРНАУКИ РОССИИ", {"size": 12}),
        (
            "Федеральное государственное бюджетное образовательное учреждение\nвысшего образования",
            {"size": 12},
        ),
        ("«МИРЭА – Российский технологический университет»", {"size": 12, "bold": True}),
        ("РТУ МИРЭА", {"size": 16, "bold": True, "after": 6}),
        ("Институт комплексной безопасности и специального приборостроения", {"size": 12}),
        ("Кафедра КБ-3 «Безопасность программных решений»", {"size": 12, "after": 90}),
        ("КУРСОВАЯ РАБОТА", {"size": 16, "bold": True, "after": 12}),
        ("по дисциплине", {}),
        ("«Применение машинного обучения для программных решений»", {"after": 18}),
        ("на тему:", {}),
        (
            "«Агент автоматического расследования аномалий межсервисного взаимодействия "
            "и реагирования на них»",
            {"bold": True, "after": 90},
        ),
        (
            "Выполнил студент группы ____________\t____________ / ____________ /",
            {"align": WD_ALIGN_PARAGRAPH.LEFT, "after": 18, "tab": True},
        ),
        (
            "Руководитель работы ____________\t____________ / ____________ /",
            {"align": WD_ALIGN_PARAGRAPH.LEFT, "after": 30, "tab": True},
        ),
        (
            "Работа представлена к защите «___» __________ 2026 г.",
            {"align": WD_ALIGN_PARAGRAPH.LEFT, "after": 6},
        ),
        (
            "Допущен к защите «___» __________ 2026 г.",
            {"align": WD_ALIGN_PARAGRAPH.LEFT, "after": 110},
        ),
        ("Москва 2026", {"page_break": True}),
    ]
    for text, options in lines:
        paragraph = doc.add_paragraph()
        fmt = paragraph.paragraph_format
        paragraph.alignment = options.get("align", WD_ALIGN_PARAGRAPH.CENTER)
        fmt.first_line_indent = Cm(0)
        fmt.line_spacing_rule = WD_LINE_SPACING.SINGLE
        fmt.space_after = Pt(options.get("after", 0))
        if options.get("tab"):
            fmt.tab_stops.add_tab_stop(TEXT_WIDTH, WD_TAB_ALIGNMENT.RIGHT)
        for i, part in enumerate(text.split("\n")):
            run = paragraph.add_run(part)
            run.font.name = FONT
            run.font.size = Pt(options.get("size", 14))
            run.bold = options.get("bold", False)
            if i < len(text.split("\n")) - 1:
                run.add_break()
        if options.get("page_break"):
            paragraph.add_run().add_break(WD_BREAK.PAGE)
        anchor.addprevious(paragraph._p)


def format_headings(doc: Any) -> None:
    for paragraph in doc.paragraphs:
        if not paragraph.style.name.startswith("Heading"):
            continue
        text = paragraph.text.strip()
        if text in STRUCTURAL or text.startswith("ПРИЛОЖЕНИЕ"):
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.first_line_indent = Cm(0)
        if "|" in text:
            label, title = (part.strip() for part in text.split("|", 1))
            for run in list(paragraph.runs):
                run._r.getparent().remove(run._r)
            first = paragraph.add_run(label)
            first.add_break()
            paragraph.add_run(title)


def number_formulas(doc: Any) -> None:
    counter = 0
    for paragraph in doc.paragraphs:
        block = paragraph._p.find(f"{{{MATH}}}oMathPara")
        if block is None:
            continue
        counter += 1
        equation = block.find(f"{{{MATH}}}oMath")
        position = list(paragraph._p).index(block)
        paragraph._p.remove(block)
        fmt = paragraph.paragraph_format
        fmt.first_line_indent = Cm(0)
        fmt.alignment = WD_ALIGN_PARAGRAPH.LEFT
        fmt.space_before = fmt.space_after = Pt(14)
        fmt.tab_stops.add_tab_stop(TEXT_WIDTH // 2, WD_TAB_ALIGNMENT.CENTER)
        fmt.tab_stops.add_tab_stop(TEXT_WIDTH, WD_TAB_ALIGNMENT.RIGHT)
        tab, after, number = (OxmlElement("w:r") for _ in range(3))
        tab.append(OxmlElement("w:tab"))
        after.append(OxmlElement("w:tab"))
        text = OxmlElement("w:t")
        text.text = f"({counter})"
        number.append(text)
        for offset, element in enumerate((tab, equation, after, number)):
            paragraph._p.insert(position + offset, element)


def format_tables(doc: Any) -> None:
    for table in doc.tables:
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        borders = OxmlElement("w:tblBorders")
        for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
            element = OxmlElement(f"w:{edge}")
            element.set(qn("w:val"), "single")
            element.set(qn("w:sz"), "4")
            element.set(qn("w:color"), "000000")
            borders.append(element)
        properties = table._tbl.tblPr
        for existing in properties.findall(qn("w:tblBorders")):
            properties.remove(existing)
        properties.append(borders)
        caption = table._tbl.getprevious()
        appendix = caption is not None and "Таблица Б" in "".join(caption.itertext())
        size = Pt(10 if len(table.columns) >= 6 or appendix else 12)
        following = table._tbl.getnext()
        if following is not None and following.tag == qn("w:p"):
            Paragraph(following, table._parent).paragraph_format.space_before = Pt(21)
        for index, row in enumerate(table.rows):
            for cell in row.cells:
                if index == 0:
                    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                for paragraph in cell.paragraphs:
                    paragraph.style = find_style(doc, "Compact")
                    if index == 0:
                        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for run in paragraph.runs:
                        run.font.size = size
                        if index == 0:
                            run.bold = True


def format_misc(doc: Any) -> None:
    for paragraph in doc.paragraphs:
        if paragraph.style.name == "Source Code":
            for run in paragraph.runs:
                run.font.size = Pt(8)
        text = paragraph.text
        if text.startswith(("Листинг ", "Отчёт администратору:")):
            paragraph.paragraph_format.first_line_indent = Cm(0)
            paragraph.paragraph_format.keep_with_next = True
    settings = doc.settings.element
    for tag in ("w:updateFields", "w:autoHyphenation"):
        element = OxmlElement(tag)
        element.set(qn("w:val"), "true")
        settings.append(element)


def postprocess(source: Path, target: Path) -> None:
    doc = Document(str(source))
    configure(
        doc,
        "Source Code",
        size=8,
        align=WD_ALIGN_PARAGRAPH.LEFT,
        indent=Cm(0),
        spacing=WD_LINE_SPACING.SINGLE,
        font="Courier New",
        after=14,
        hyphenate=False,
        create="SourceCode",
    )
    setup_page(doc)
    format_headings(doc)
    number_formulas(doc)
    format_tables(doc)
    format_misc(doc)
    title_page(doc)
    doc.save(str(target))


def heading_entries(doc: Any) -> list[tuple[int, str]]:
    entries = []
    for paragraph in doc.paragraphs:
        name = paragraph.style.name
        if name in {"Heading 1", "Heading 2"}:
            text = " ".join(paragraph.text.split())
            if text.startswith("ПРИЛОЖЕНИЕ"):
                text = "Приложение" + text.removeprefix("ПРИЛОЖЕНИЕ")
            entries.append((int(name[-1]), text))
    return entries


def locate_pages(pdf: Path, entries: list[tuple[int, str]]) -> list[int]:
    from pypdf import PdfReader  # noqa: PLC0415

    pages = [" ".join((page.extract_text() or "").split()) for page in PdfReader(str(pdf)).pages]
    found, cursor = [], 2
    for _, text in entries:
        probe = " ".join(text.split()[:6]).upper()
        for number in range(cursor, len(pages)):
            if probe in pages[number].upper():
                cursor = number
                break
        found.append(cursor + 1)
    return found


def fill_toc(doc: Any, entries: list[tuple[int, str]], pages: list[int]) -> None:
    sdt = next(
        e
        for e in doc.element.body.iter(qn("w:sdt"))
        if any("TOC" in (t.text or "") for t in e.iter(qn("w:instrText")))
    )
    content = sdt.find(qn("w:sdtContent"))
    field = next(
        p for p in content.findall(qn("w:p")) if p.find(f".//{qn('w:instrText')}") is not None
    )
    instruction = field.find(f".//{qn('w:instrText')}").text
    content.remove(field)

    def run(*children: Any) -> Any:
        element = OxmlElement("w:r")
        element.extend(children)
        return element

    def char(kind: str) -> Any:
        element = OxmlElement("w:fldChar")
        element.set(qn("w:fldCharType"), kind)
        return element

    def text(value: str) -> Any:
        element = OxmlElement("w:t")
        element.set(qn("xml:space"), "preserve")
        element.text = value
        return element

    for index, ((level, title), page) in enumerate(zip(entries, pages, strict=True)):
        paragraph = OxmlElement("w:p")
        properties = OxmlElement("w:pPr")
        style = OxmlElement("w:pStyle")
        style.set(qn("w:val"), f"TOC{level}")
        properties.append(style)
        paragraph.append(properties)
        if index == 0:
            command = OxmlElement("w:instrText")
            command.set(qn("xml:space"), "preserve")
            command.text = instruction
            paragraph.extend([run(char("begin")), run(command), run(char("separate"))])
        paragraph.extend([run(text(title)), run(OxmlElement("w:tab")), run(text(str(page)))])
        if index == len(entries) - 1:
            paragraph.append(run(char("end")))
        content.append(paragraph)


def paginate(docx_path: Path, soffice: str) -> None:
    """Render with LibreOffice, find heading pages and pre-fill the table of contents."""
    pdf_dir = BUILD / "pdf"
    subprocess.run(
        [
            soffice,
            "--headless",
            f"-env:UserInstallation=file://{BUILD / 'lo-profile'}",
            "--convert-to",
            "pdf",
            "--outdir",
            str(pdf_dir),
            str(docx_path),
        ],
        check=True,
        capture_output=True,
    )
    doc = Document(str(docx_path))
    entries = heading_entries(doc)
    fill_toc(doc, entries, locate_pages(pdf_dir / f"{docx_path.stem}.pdf", entries))
    doc.save(str(docx_path))


def main() -> None:
    shutil.rmtree(BUILD, ignore_errors=True)
    figures = BUILD / "figures"
    diagrams.architecture(figures / "architecture.png")
    diagrams.topology(figures / "topology.png")
    hard = load(REPORTS / "agent" / "heuristic-baseline" / "hard" / "summary.json")
    if hard is not None:
        confusion_heatmap(
            hard,
            REPORTS / "agent" / "heuristic-baseline" / "hard" / "confusion.png",
            "Эвристический агент, усложнённые сценарии",
        )
    markdown = BUILD / "report.md"
    markdown.write_text(render_markdown(), encoding="utf-8")
    reference = style_reference(BUILD / "reference.docx")
    raw = BUILD / "raw.docx"
    run_pandoc(markdown, reference, raw)
    postprocess(raw, OUTPUT)
    if soffice := os.environ.get("SOFFICE") or shutil.which("soffice"):
        paginate(OUTPUT, soffice)
    else:
        print(
            "LibreOffice not found: the table of contents will be filled when Word updates fields"
        )
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
