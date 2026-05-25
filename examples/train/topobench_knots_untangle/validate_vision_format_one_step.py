#!/usr/bin/env python3
"""
Minimal TopoBench Knots Untangle format validator.

This intentionally avoids Ray, PPO, FSDP, and sglang. Use it to check the
smallest path that matters for the current VAGEN wrapper:
- reset one visual environment
- build the multimodal chat prompt
- parse/step one assistant answer in <answer>ACTION_IDX</answer> format
- optionally load a local HF vision-language model and test its raw output
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import Any, Dict, List, Sequence

from PIL import Image

SCRIPTDIR = os.path.dirname(os.path.abspath(__file__))
VAGEN_ROOT = os.path.abspath(os.path.join(SCRIPTDIR, "../../.."))
TOPOBENCH_ROOT = os.path.abspath(os.path.join(VAGEN_ROOT, ".."))
sys.path.insert(0, VAGEN_ROOT)
sys.path.insert(0, os.path.join(VAGEN_ROOT, "verl"))
sys.path.insert(0, TOPOBENCH_ROOT)

from vagen.envs.topobench_knots_untangle.topobench_knots_untangle_env import (  # noqa: E402
    TopobenchKnotsUntangle,
)
from vagen.envs.topobench_knots_untangle.utils.parse import parse_response  # noqa: E402


def _collect_images(obs: Dict[str, Any], image_placeholder: str = "<image>") -> List[Image.Image]:
    imgs = obs.get("multi_modal_input", {}).get(image_placeholder, []) or []
    return [img.convert("RGB") for img in imgs if isinstance(img, Image.Image)]


def _convert_obs_to_content(obs: Dict[str, Any], image_placeholder: str = "<image>") -> List[Dict[str, Any]]:
    text = obs["obs_str"]
    image_count = len(obs.get("multi_modal_input", {}).get(image_placeholder, []) or [])
    placeholder_count = text.count(image_placeholder)
    if placeholder_count != image_count:
        raise AssertionError(f"#images ({image_count}) != #{image_placeholder} ({placeholder_count})")

    content: List[Dict[str, Any]] = []
    parts = text.split(image_placeholder)
    for index, part in enumerate(parts):
        if part:
            content.append({"type": "text", "text": part})
        if index < len(parts) - 1:
            content.append({"type": "image"})
    return content


def _check_obs(obs: Dict[str, Any], label: str, image_placeholder: str = "<image>") -> None:
    obs_str = obs.get("obs_str", "")
    image_count = len(_collect_images(obs, image_placeholder=image_placeholder))
    placeholder_count = obs_str.count(image_placeholder)
    print(
        f"[topobench-format] {label}: placeholders={placeholder_count} images={image_count} "
        f"chars={len(obs_str)}"
    )
    if placeholder_count != image_count:
        raise AssertionError(
            f"{label}: #images ({image_count}) != #{image_placeholder} ({placeholder_count})"
        )


def _resize_images(images: Sequence[Image.Image], max_side: int) -> List[Image.Image]:
    if max_side <= 0:
        return [img.convert("RGB") for img in images]

    out: List[Image.Image] = []
    for image in images:
        resized = image.convert("RGB").copy()
        resized.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        out.append(resized)
    return out


def _format_messages(sys_obs: Dict[str, Any], init_obs: Dict[str, Any]) -> tuple[List[Dict[str, Any]], List[Image.Image]]:
    messages: List[Dict[str, Any]] = []
    images: List[Image.Image] = []

    if sys_obs:
        messages.append({"role": "system", "content": _convert_obs_to_content(sys_obs)})
        images.extend(_collect_images(sys_obs))
    if init_obs:
        messages.append({"role": "user", "content": _convert_obs_to_content(init_obs)})
        images.extend(_collect_images(init_obs))

    return messages, images


def _apply_chat_template(processor, messages: List[Dict[str, Any]], *, enable_thinking: bool) -> str:
    kwargs = {"enable_thinking": enable_thinking}
    try:
        return processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            **kwargs,
        )
    except TypeError:
        return processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )


def _processor_inputs(processor, messages: List[Dict[str, Any]], images: List[Image.Image], args):
    prompt = _apply_chat_template(processor, messages, enable_thinking=args.enable_thinking)
    resized_images = _resize_images(images, args.image_max_side)
    return prompt, processor(text=[prompt], images=resized_images or None, return_tensors="pt")


def _decode_generated(processor, generated_ids, input_ids) -> str:
    trimmed = [output_ids[len(prompt_ids) :] for prompt_ids, output_ids in zip(input_ids, generated_ids)]
    decoder = getattr(processor, "batch_decode", None) or processor.tokenizer.batch_decode
    return decoder(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def _load_model(args):
    import torch
    from transformers import AutoProcessor

    try:
        from transformers import AutoModelForImageTextToText as AutoModel
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoModel

    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
    }
    if args.device_map:
        model_kwargs["device_map"] = args.device_map
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

    model = AutoModel.from_pretrained(args.model_id, **model_kwargs)
    model.eval()
    if not args.device_map and torch.cuda.is_available():
        model.to("cuda")

    return processor, model, torch


async def _run(args) -> int:
    env_config = {
        "task_name": "knots_untangle",
        "difficulty": args.difficulty,
        "prompt_format": args.prompt_format,
        "max_steps": 1,
        "headless": True,
        "animate": False,
    }
    env = TopobenchKnotsUntangle(env_config)

    try:
        init_obs, reset_info = await env.reset(seed=args.seed)
        sys_obs = await env.system_prompt()

        _check_obs(sys_obs, "system")
        _check_obs(init_obs, "initial")
        print(
            f"[topobench-format] reset: action_space_n={reset_info.get('action_space_n')} "
            f"seed={args.seed} difficulty={args.difficulty}"
        )

        messages, images = _format_messages(sys_obs, init_obs)
        print(f"[topobench-format] messages={len(messages)} initial_images={len(images)}")

        if args.mode in {"tokenize", "generate"}:
            if args.mode == "tokenize":
                from transformers import AutoProcessor

                processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
            else:
                processor, model, torch = _load_model(args)

            raw_prompt, model_inputs = _processor_inputs(processor, messages, images, args)
            print(
                f"[topobench-format] prompt_chars={len(raw_prompt)} "
                f"input_ids_shape={tuple(model_inputs['input_ids'].shape)}"
            )

        if args.mode == "generate":
            device = getattr(model, "device", None)
            if device is not None:
                model_inputs = {
                    key: value.to(device) if hasattr(value, "to") else value
                    for key, value in model_inputs.items()
                }
            with torch.inference_mode():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            reply = _decode_generated(processor, generated_ids, model_inputs["input_ids"])
            print(f"[topobench-format] generated_reply={reply!r}")
        else:
            reply = args.reply

        parsed = parse_response(reply, prompt_format=args.prompt_format)
        print(
            f"[topobench-format] parsed: format_correct={parsed.get('format_correct')} "
            f"action_idx={parsed.get('action_idx')} action_content={parsed.get('action_content')!r}"
        )

        next_obs, reward, done, info = await env.step(reply)
        _check_obs(next_obs, "after_step")
        print(
            f"[topobench-format] step: reward={reward} done={done} "
            f"format_correct={info.get('format_correct')} "
            f"action_is_valid={info.get('action_is_valid')} "
            f"action_idx={info.get('action_idx')} success={info.get('success')}"
        )

        ok = bool(info.get("format_correct")) and bool(info.get("action_is_valid"))
        if not ok and not args.allow_invalid:
            print("[topobench-format] FAIL: model/env did not accept the output format", file=sys.stderr)
            return 2

        print("[topobench-format] OK")
        return 0
    finally:
        await env.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate TopoBench/VAGEN one-step vision output format.")
    parser.add_argument("--mode", choices=["env", "tokenize", "generate"], default="env")
    parser.add_argument("--model-id", default=os.getenv("MODEL_ID", "Qwen/Qwen3-VL-2B-Instruct"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--prompt-format", choices=["no_think", "free_think", "wm", "free_wm"], default="no_think")
    parser.add_argument("--reply", default="<answer>0</answer>")
    parser.add_argument("--image-max-side", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--allow-invalid", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run(_parse_args())))
