#!/usr/bin/env python3
"""
Validate Sokoban (vision) multimodal prompt formatting on low-VRAM GPUs.

Goal: without loading the model, run 1 env step and ensure:
- obs_str placeholders match image count
- processor.apply_chat_template + processor(text=..., images=...) succeeds
"""

import asyncio
import os
import sys
from typing import Any, Dict, List

from PIL import Image
from transformers import AutoProcessor

SCRIPTDIR = os.path.dirname(os.path.abspath(__file__))
VAGEN_ROOT = os.path.abspath(os.path.join(SCRIPTDIR, "../../.."))
sys.path.insert(0, VAGEN_ROOT)
sys.path.insert(0, os.path.join(VAGEN_ROOT, "verl"))

from vagen.agent_loop.gym_agent_loop import convert_obs_to_content  # noqa: E402
from vagen.envs.sokoban.sokoban_env import Sokoban  # noqa: E402


def _collect_images(obs: Dict[str, Any]) -> List[Image.Image]:
    imgs = obs.get("multi_modal_input", {}).get("<image>", []) or []
    out: List[Image.Image] = []
    for im in imgs:
        if isinstance(im, Image.Image):
            out.append(im.convert("RGB"))
    return out


async def main() -> None:
    model_id = os.getenv("MODEL_ID", "Qwen/Qwen3-VL-2B-Instruct")
    print(f"[format-check] processor model_id={model_id}")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    env_config = {
        "render_mode": "vision",
        "min_solution_steps": (1, 5),
        "prompt_format": "free_wm",
    }
    env = Sokoban(env_config=env_config)

    sys_obs = await env.system_prompt()
    init_obs, _info = await env.reset(seed=1)

    messages: List[Dict[str, Any]] = []
    images: List[Image.Image] = []

    if sys_obs:
        messages.append({"role": "system", "content": convert_obs_to_content(sys_obs)})
        images.extend(_collect_images(sys_obs))
    if init_obs:
        messages.append({"role": "user", "content": convert_obs_to_content(init_obs)})
        images.extend(_collect_images(init_obs))

    raw_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    model_inputs = processor(text=[raw_prompt], images=images or None, return_tensors="pt")
    input_ids = model_inputs["input_ids"]
    print(f"[format-check] initial input_ids shape={tuple(input_ids.shape)}, images={len(images)}")

    action_str = "<observation></observation><answer>up</answer><prediction></prediction>"
    obs1, reward1, done1, info1 = await env.step(action_str)
    print(f"[format-check] step reward={reward1} done={done1} format_correct={info1.get('format_correct')}")

    # Validate next-turn suffix tokenization path too.
    user_msg = {"role": "user", "content": convert_obs_to_content(obs1)}
    suffix_prompt = processor.apply_chat_template(
        [{"role": "system", "content": "placeholder"}, user_msg],
        add_generation_prompt=True,
        tokenize=False,
    )
    new_images = _collect_images(obs1)
    suffix_inputs = processor(text=[suffix_prompt], images=new_images or None, return_tensors="pt")
    print(
        f"[format-check] suffix input_ids shape={tuple(suffix_inputs['input_ids'].shape)}, new_images={len(new_images)}"
    )

    await env.close()
    print("[format-check] OK")


if __name__ == "__main__":
    asyncio.run(main())
