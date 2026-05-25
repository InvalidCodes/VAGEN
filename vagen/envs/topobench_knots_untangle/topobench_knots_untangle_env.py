from __future__ import annotations

import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from PIL import Image

from vagen.envs.gym_image_env import GymImageEnv
from vagen.envs.topobench_knots_untangle.utils.parse import parse_response
from vagen.envs.topobench_knots_untangle.utils.prompt import (
    init_observation_template,
    step_observation_template,
    system_prompt,
)


@dataclass
class TopobenchKnotsUntangleEnvConfig:
    task_name: str = "knots_untangle"
    difficulty: str = "medium"  # easy|medium|hard
    max_steps: int = 40
    headless: bool = True
    animate: bool = False
    viewport_width: int = 1440
    viewport_height: int = 960
    prompt_format: str = "no_think"  # no_think|free_think|wm|free_wm

    # Reward: pure success/fail only.
    success_reward: float = 1.0
    fail_reward: float = 0.0

    image_placeholder: str = "<image>"


class TopobenchKnotsUntangle(GymImageEnv):
    """
    TopoBench knots_untangle wrapper for VAGEN-Lite.

    - Observation: single RGB image + short metadata text.
    - Action: a single discrete ACTION_IDX (int).
    - Reward: success/fail only (default: 1 on success else 0).
    """

    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)
        self.config = TopobenchKnotsUntangleEnvConfig(**env_config)

        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="topobench-knots")
        self._env = None
        self._difficulty = str(self.config.difficulty).lower().strip() or "medium"
        self._grid_size: Optional[int] = None
        self._rope_count: Optional[int] = None
        self._action_space_n: Optional[int] = None

        self._last_extracted_action_idx: str = "N/A"
        self._last_action_is_valid = False
        self._episode_steps = 0

    async def _run_blocking(self, fn, /, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, functools.partial(fn, *args, **kwargs))

    async def close(self) -> None:
        if self._env is not None:
            try:
                await self._run_blocking(self._env.close)
            finally:
                self._env = None

        if self._executor is not None:
            executor = self._executor
            self._executor = None
            await asyncio.to_thread(executor.shutdown, wait=True, cancel_futures=True)

    async def system_prompt(self) -> Dict[str, Any]:
        return {"obs_str": system_prompt(prompt_format=self.config.prompt_format)}

    async def reset(self, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        from environments.knots_untangle.gym.env import KnotsUntangleEnv

        if self._env is not None:
            await self._run_blocking(self._env.close)
            self._env = None

        self._env = await self._run_blocking(
            KnotsUntangleEnv,
            difficulty=self._difficulty,
            seed=seed,
            headless=bool(self.config.headless),
            animate=bool(self.config.animate),
            max_steps=int(self.config.max_steps),
            viewport_width=int(self.config.viewport_width),
            viewport_height=int(self.config.viewport_height),
        )

        obs_np, info = await self._run_blocking(self._env.reset, seed=seed)
        self._grid_size = int(getattr(self._env, "grid_size"))
        self._rope_count = int(getattr(self._env, "rope_count"))
        self._action_space_n = int(getattr(self._env.action_space, "n"))
        self._last_extracted_action_idx = "N/A"
        self._last_action_is_valid = False
        self._episode_steps = 0

        obs = self._obs_from_np(obs_np, init=True, done=False, success=bool(info.get("success", False)))
        out_info = dict(info or {})
        out_info.update(
            {
                "task_name": self.config.task_name,
                "prompt_format": self.config.prompt_format,
                "action_space_n": self._action_space_n,
                "episode_steps": self._episode_steps,
                "success": bool(out_info.get("success", False)),
            }
        )
        return obs, out_info

    async def step(self, action_str: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self._env is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        parsed = parse_response(action_str, prompt_format=self.config.prompt_format)
        action_idx = parsed.get("action_idx")
        format_ok = bool(parsed.get("format_correct", False))

        if self._action_space_n is None:
            self._action_space_n = int(getattr(self._env.action_space, "n"))

        if not format_ok or action_idx is None or not (0 <= int(action_idx) < int(self._action_space_n)):
            action_idx = 0
            self._last_extracted_action_idx = "INVALID->0"
            parsed["action_is_valid"] = False
        else:
            self._last_extracted_action_idx = str(int(action_idx))
            parsed["action_is_valid"] = True
        self._last_action_is_valid = bool(parsed["action_is_valid"])

        obs_np, _env_reward, terminated, truncated, info = await self._run_blocking(self._env.step, int(action_idx))
        done = bool(terminated or truncated)
        success = bool((info or {}).get("success", False))
        self._episode_steps = int((info or {}).get("episode_steps", self._episode_steps + 1))

        reward = float(self.config.success_reward if success else self.config.fail_reward)

        out_info: Dict[str, Any] = dict(info or {})
        out_info.update(parsed)
        out_info["success"] = success
        out_info["task_name"] = self.config.task_name
        out_info["prompt_format"] = self.config.prompt_format
        out_info["action_space_n"] = self._action_space_n
        out_info["episode_steps"] = self._episode_steps
        out_info["metrics"] = {
            "turn_metrics": {
                "format_correct": bool(parsed.get("format_correct", False)),
                "action_is_valid": self._last_action_is_valid,
            },
            "traj_metrics": {
                "success": success,
            },
        }

        obs = self._obs_from_np(obs_np, init=False, done=done, success=success)
        return obs, reward, done, out_info

    def _obs_from_np(self, obs_np, *, init: bool, done: bool, success: bool) -> Dict[str, Any]:
        image = Image.fromarray(obs_np.astype("uint8"), mode="RGB")
        img_str = self.config.image_placeholder

        if init:
            obs_str = init_observation_template(
                img_str,
                task_name=self.config.task_name,
                difficulty=self._difficulty,
                prompt_format=self.config.prompt_format,
                grid_size=int(self._grid_size or 0),
                rope_count=int(self._rope_count or 0),
                action_space_n=int(self._action_space_n or 0),
                step_idx=self._episode_steps,
                max_steps=int(self.config.max_steps),
            )
        else:
            obs_str = step_observation_template(
                img_str,
                task_name=self.config.task_name,
                difficulty=self._difficulty,
                prompt_format=self.config.prompt_format,
                extracted_action_idx=self._last_extracted_action_idx,
                action_is_valid=self._last_action_is_valid,
                done=done,
                success=success,
                step_idx=self._episode_steps,
                max_steps=int(self.config.max_steps),
            )

        return {
            "obs_str": obs_str,
            "multi_modal_input": {self.config.image_placeholder: [image]},
        }
