"""0/1 reward for MMK12 open answers.

The prompt requires the final answer in \\boxed{}. An unboxed generation scores 0.
The gold answer is compared after light LaTeX normalization, then numerically
when both sides are numbers.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

_LETTER_ONLY = re.compile(r"^[\(\[]?([A-Da-d])[\)\].:]?$")
_NUMBER = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")
_UNITS = (
    "\\circ",
    "cm",
    "mm",
    "km",
    "m",
    "kg",
    "g",
    "°",
    "degrees",
    "degree",
)


@dataclass(frozen=True)
class Grade:
    reward: int
    format_ok: int
    prediction: str | None


def extract_boxed(text: str) -> str | None:
    if not text:
        return None
    boxed = text.rfind(r"\boxed{")
    if boxed >= 0:
        brace = boxed + len(r"\boxed{") - 1
    else:
        bare = text.rfind("boxed{")
        if bare < 0:
            return None
        brace = bare + len("boxed{") - 1
    depth = 0
    for index in range(brace, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[brace + 1 : index]
    return None


def normalize_answer(text: str) -> str:
    value = text.strip().lower()
    value = value.replace("$", "")
    for token in ("\\,", "\\!", "\\;", "\\ ", "\\left", "\\right", "\\mathrm", "\\text"):
        value = value.replace(token, "")
    value = value.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    value = value.replace(" ", "").replace("\n", "").replace("\t", "")
    value = value.replace("{", "").replace("}", "")
    value = value.replace("\\degree", "°").replace("^\\circ", "°")
    for unit in _UNITS:
        if value.endswith(unit):
            value = value[: -len(unit)]
    return value


def _as_float(text: str) -> float | None:
    cleaned = normalize_answer(text).replace(",", "")
    if not _NUMBER.fullmatch(cleaned):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def grade(text: str, answer: str) -> Grade:
    boxed = extract_boxed(text)
    if boxed is None:
        return Grade(0, 0, None)
    pred = normalize_answer(boxed)
    gold = normalize_answer(answer)
    if pred == gold and pred != "":
        return Grade(1, 1, boxed)
    pred_number = _as_float(boxed)
    gold_number = _as_float(answer)
    if (
        pred_number is not None
        and gold_number is not None
        and math.isfinite(pred_number)
        and math.isfinite(gold_number)
        and abs(pred_number - gold_number) <= max(1e-3, 1e-3 * abs(gold_number))
    ):
        return Grade(1, 1, boxed)
    if _LETTER_ONLY.fullmatch(boxed.strip()) and _LETTER_ONLY.fullmatch(answer.strip()):
        same = boxed.strip()[0].upper() == answer.strip()[0].upper()
        return Grade(int(same), 1, boxed)
    return Grade(0, 1, boxed)
