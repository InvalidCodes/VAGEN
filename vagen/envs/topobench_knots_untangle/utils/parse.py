from __future__ import annotations

import re
from typing import Any, Dict, Optional


VALID_FORMATS = ("no_think", "free_think", "wm", "free_wm")

_NO_THINK_PATTERN = re.compile(r"<answer>(.*?)</answer>", flags=re.DOTALL | re.IGNORECASE)
_FREE_THINK_PATTERN = re.compile(
    r"<think>(.*?)</think>\s*<answer>(.*?)</answer>",
    flags=re.DOTALL | re.IGNORECASE,
)
_WM_PATTERN = re.compile(
    r"<observation>(.*?)</observation>\s*"
    r"<think>(.*?)</think>\s*"
    r"<answer>(.*?)</answer>\s*"
    r"<prediction>(.*?)</prediction>",
    flags=re.DOTALL | re.IGNORECASE,
)
_FREE_WM_PATTERN = re.compile(
    r"<observation>(.*?)</observation>"
    r"(.*?)"
    r"<answer>(.*?)</answer>"
    r"(.*?)"
    r"<prediction>(.*?)</prediction>",
    flags=re.DOTALL | re.IGNORECASE,
)


def _parse_action_idx(answer_content: str) -> Optional[int]:
    try:
        return int(answer_content.strip())
    except Exception:
        return None


def _build_result(
    *,
    response: str,
    prompt_format: str,
    matched: bool,
    action_content: str = "",
    think_content: str = "",
    observation_content: str = "",
    prediction_content: str = "",
    reasoning_content: str = "",
) -> Dict[str, Any]:
    action_content = action_content.strip()
    action_idx = _parse_action_idx(action_content)
    if not reasoning_content:
        reasoning_content = think_content

    if prompt_format == "no_think":
        llm_response = f"<answer>{action_content}</answer>"
    elif prompt_format == "free_think":
        llm_response = f"<think>{think_content}</think><answer>{action_content}</answer>"
    elif prompt_format == "wm":
        llm_response = (
            f"<observation>{observation_content}</observation>"
            f"<think>{think_content}</think>"
            f"<answer>{action_content}</answer>"
            f"<prediction>{prediction_content}</prediction>"
        )
    elif prompt_format == "free_wm":
        llm_response = (
            f"<observation>{observation_content}</observation>"
            f"<answer>{action_content}</answer>"
            f"<prediction>{prediction_content}</prediction>"
        )
    else:
        raise ValueError(f"Unknown prompt_format: {prompt_format}. Valid: {VALID_FORMATS}")

    return {
        "llm_raw_response": response,
        "llm_response": llm_response,
        "prompt_format": prompt_format,
        "observation_content": observation_content,
        "think_content": think_content,
        "reasoning_content": reasoning_content,
        "prediction_content": prediction_content,
        "action_content": action_content,
        "action_idx": action_idx,
        "actions": [] if action_idx is None else [action_idx],
        "format_correct": matched and action_idx is not None,
    }


def parse_response(response: str, prompt_format: str = "no_think") -> Dict[str, Any]:
    raw = response if response is not None else ""

    if prompt_format == "no_think":
        match = _NO_THINK_PATTERN.search(raw)
        return _build_result(
            response=raw,
            prompt_format=prompt_format,
            matched=match is not None,
            action_content="" if match is None else match.group(1),
        )

    if prompt_format == "free_think":
        match = _FREE_THINK_PATTERN.search(raw)
        return _build_result(
            response=raw,
            prompt_format=prompt_format,
            matched=match is not None,
            think_content="" if match is None else match.group(1).strip(),
            action_content="" if match is None else match.group(2),
        )

    if prompt_format == "wm":
        match = _WM_PATTERN.search(raw)
        return _build_result(
            response=raw,
            prompt_format=prompt_format,
            matched=match is not None,
            observation_content="" if match is None else match.group(1).strip(),
            think_content="" if match is None else match.group(2).strip(),
            action_content="" if match is None else match.group(3),
            prediction_content="" if match is None else match.group(4).strip(),
        )

    if prompt_format == "free_wm":
        match = _FREE_WM_PATTERN.search(raw)
        reasoning_content = ""
        if match is not None:
            reasoning_parts = [
                part.strip()
                for part in (match.group(2), match.group(4))
                if part.strip()
            ]
            reasoning_content = " ".join(reasoning_parts)
        return _build_result(
            response=raw,
            prompt_format=prompt_format,
            matched=match is not None,
            observation_content="" if match is None else match.group(1).strip(),
            reasoning_content=reasoning_content,
            action_content="" if match is None else match.group(3),
            prediction_content="" if match is None else match.group(5).strip(),
        )

    raise ValueError(f"Unknown prompt_format: {prompt_format}. Valid: {VALID_FORMATS}")


def parse_action_idx(response: str) -> Dict[str, Any]:
    return parse_response(response, prompt_format="no_think")
