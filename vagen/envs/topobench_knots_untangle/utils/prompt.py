from __future__ import annotations


VALID_FORMATS = ("no_think", "free_think", "wm", "free_wm")


_FORMAT_INSTRUCTIONS = {
    "no_think": (
        "You need to only give your action. Respond in this format:\n"
        "<answer>ACTION_IDX</answer>"
    ),
    "free_think": (
        "You need to think first, then give your action. Respond in this format:\n"
        "<think>...</think><answer>ACTION_IDX</answer>"
    ),
    "wm": (
        "You need to describe the visible state, think, give your action, then predict "
        "the next state. Respond in this format:\n"
        "<observation>...</observation><think>...</think>"
        "<answer>ACTION_IDX</answer><prediction>...</prediction>"
    ),
    "free_wm": (
        "You need to describe the visible state, give your action, then predict "
        "the next state. Free-form reasoning may appear between tags. Respond in this format:\n"
        "<observation>...</observation> ... <answer>ACTION_IDX</answer> ... <prediction>...</prediction>"
    ),
}


def get_format_instruction(prompt_format: str) -> str:
    if prompt_format not in _FORMAT_INSTRUCTIONS:
        raise ValueError(f"Unknown prompt_format: {prompt_format}. Valid: {VALID_FORMATS}")
    return _FORMAT_INSTRUCTIONS[prompt_format]


def system_prompt(prompt_format: str = "no_think") -> str:
    return f"""You are solving TopoBench: Knots Untangle.

Goal: untangle the ropes (reach success).

{get_format_instruction(prompt_format)}

- ACTION_IDX must be an integer in [0, action_space_n - 1].
- ACTION_IDX selects a single move (moving one rope endpoint to one hole).
"""


def init_observation_template(
    img_str: str,
    *,
    task_name: str,
    difficulty: str,
    prompt_format: str,
    grid_size: int,
    rope_count: int,
    action_space_n: int,
    step_idx: int,
    max_steps: int,
) -> str:
    return f"""[Initial Observation]
task_name={task_name}
difficulty={difficulty}
prompt_format={prompt_format}
grid_size={grid_size}x{grid_size}
rope_count={rope_count}
action_space_n={action_space_n}
step_idx={step_idx}
max_steps={max_steps}

{img_str}

Pick the next ACTION_IDX."""


def step_observation_template(
    img_str: str,
    *,
    task_name: str,
    difficulty: str,
    prompt_format: str,
    extracted_action_idx: str,
    action_is_valid: bool,
    done: bool,
    success: bool,
    step_idx: int,
    max_steps: int,
) -> str:
    return f"""After your answer, the extracted ACTION_IDX is: {extracted_action_idx}
task_name={task_name}
difficulty={difficulty}
prompt_format={prompt_format}
action_is_valid={action_is_valid}
done={done} success={success}
step_idx={step_idx}
max_steps={max_steps}

{img_str}

Pick the next ACTION_IDX."""
