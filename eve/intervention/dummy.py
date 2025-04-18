from typing import Any, Dict, List, Optional
import numpy as np
import gymnasium as gym

from .instrument import Instrument
from .intervention import Intervention
from .target import Target
from .vesseltree import VesselTree
from .fluoroscopy import Fluoroscopy


class InterventionDummy(Intervention):
    def __init__(
        self,
        instruments: List[Instrument],
        vessel_tree: VesselTree,
        fluoroscopy: Fluoroscopy,
        target: Target,
        normalize_action: bool = False,
    ) -> None:
        self.instruments = instruments
        self.vessel_tree = vessel_tree
        self.fluoroscopy = fluoroscopy
        self.target = target
        self.normalize_action = normalize_action

        self.last_action = np.zeros((len(self.instruments), 2), dtype=np.float32)
        self.velocity_limits = np.array(
            [instrument.velocity_limit for instrument in self.instruments]
        )

        if self.normalize_action:
            high = np.ones_like(self.velocity_limits)
            space = gym.spaces.Box(low=-high, high=high)
        else:
            space = gym.spaces.Box(low=-self.velocity_limits, high=self.velocity_limits)
        self.action_space = space

        self.instrument_lengths_maximum = [
            instrument.length for instrument in self.instruments
        ]
        self.instrument_diameters = [
            instrument.tip_section.diameter_outer for instrument in self.instruments
        ]
        self.instrument_lengths_inserted = [0.0 for _ in instruments]
        self.instrument_rotations = [0.0 for _ in instruments]

        self._np_random = np.random.default_rng()

    def step(self, action: np.ndarray) -> None:
        action = np.array(action).reshape(self.velocity_limits.shape)
        if self.normalize_action:
            action = np.clip(action, -1.0, 1.0)
            self.last_action = action
            high = self.velocity_limits
            low = -high
            action = (action + 1) / 2 * (high - low) + low
        else:
            action = np.clip(action, -self.velocity_limits, self.velocity_limits)
            self.last_action = action

        self.vessel_tree.step()
        self.fluoroscopy.step()
        self.target.step()

    def reset(
        self,
        episode_number: int = 0,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        vessel_seed = None if seed is None else self._np_random.integers(0, 2**31)
        self.vessel_tree.reset(episode_number, vessel_seed)
        target_seed = None if seed is None else self._np_random.integers(0, 2**31)
        self.target.reset(episode_number, target_seed)

    def reset_instruments(self) -> None: ...

    def close(self): ...
