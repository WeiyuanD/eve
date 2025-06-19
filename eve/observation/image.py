from copy import deepcopy
import numpy as np
import PIL.Image

from .observation import Observation, gym

from ..intervention import Intervention


class Image(Observation):
    def __init__(self, intervention: Intervention, name: str = "image") -> None:
        self.name = name
        self.intervention = intervention
        self.image: PIL.Image.Image = None
        self.obs = None

    @property
    def space(self) -> gym.spaces.Box:
        return self.intervention.fluoroscopy.image_space

    def step(self) -> None:
        self.image = deepcopy(self.intervention.fluoroscopy.image)
        # self.obs = np.array(self.image, dtype=np.float32)
        # WD:
        image_obs = np.array(self.image, dtype=np.float32)
        image_obs = np.expand_dims(image_obs.reshape((96, 96)), axis=-1)
        # image_obs = np.expand_dims(image_obs.reshape((64, 64)), axis=-1) # v1
        self.obs = np.array(image_obs, dtype=np.float32)

    def reset(self, episode_nr: int = 0) -> None:
        self.step()
