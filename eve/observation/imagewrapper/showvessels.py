from typing import Dict, Optional
from PIL import ImageChops, Image
import numpy as np

from .. import Image as ImageState
from ...intervention import Intervention


class ShowVessels(ImageState):
    def __init__(
        self,
        intervention: Intervention,
        wrapped_image: ImageState,
        name: Optional[str] = None,
    ) -> None:
        name = name or wrapped_image.name
        super().__init__(wrapped_image.intervention, name)
        self.intervention = intervention
        self.wrapped_image = wrapped_image
        self._overlay_image = None
        self.target = None
        self.vessel_image = None

    @property
    def space(self) -> Dict[str, np.ndarray]:
        return self.wrapped_image.space

    def step(self) -> None:
        self.wrapped_image.step()
        # self.image = ImageChops.blend(
        #     self.wrapped_image.image, self._overlay_image, 0.5
        # )
        # WD:
        img1 = Image.fromarray(self.wrapped_image.image)
        img2 = Image.fromarray(self._overlay_image)
        self.image = ImageChops.blend(
            img1, img2, 0.5
        )
        image_obs = np.array(self.image, dtype=np.float32)
        # image_obs = np.expand_dims(image_obs.reshape((64, 64)), axis=0)
        self.obs = np.array(image_obs)

    def reset(self, episode_nr: int = 0) -> None:
        self.wrapped_image.reset(episode_nr)
        self._create_overlay_image()
        # self.image = ImageChops.blend(
        #     self.wrapped_image.image, self._overlay_image, 0.5
        # )
        # wD:
        img1 = Image.fromarray(self.wrapped_image.image)
        img2 = Image.fromarray(self._overlay_image)
        print('============CHECK===========: ', img1.size, img2.size)
        self.image = ImageChops.blend(
            img1, img2, 0.5
        )
        image_obs = np.array(self.image, dtype=np.float32)
        # image_obs = np.expand_dims(image_obs.reshape((64, 64)), axis=0) 
        self.obs = np.array(image_obs)

    def _create_overlay_image(self):
        # self._overlay_image = self.intervention.fluoroscopy.get_new_image(color=255)
        self._overlay_image = self.intervention.fluoroscopy.image
        self.target = self.intervention.target.coordinates3d
        for branch in self.intervention.vessel_tree.branches:
            for coord, radius in zip(branch.coordinates, branch.radii):
                self._overlay_image = self.intervention.fluoroscopy._draw_circle(
                    self._overlay_image, coord, radius, 170
                )
        # draw target
        self._overlay_image = self.intervention.fluoroscopy.draw_target(
            self._overlay_image, self.target, 4, 255
        )
    
    # def _create_overlay_image(self):
    #     self._overlay_image = np.array(Image.new('L', (512, 512), color=128))
    #     for branch in self.intervention.vessel_tree.branches:
    #         for coord, radius in zip(branch.coordinates, branch.radii):
    #             self._overlay_image = self.intervention.fluoroscopy._draw_circle(
    #                 self._overlay_image, coord, radius, 170
    #             )
    #     self._overlay_image = Image.fromarray(self._overlay_image.astype(np.uint8))
    #     self._overlay_image = self._overlay_image.resize((96, 96), resample=Image.Resampling.LANCZOS)

    #     # draw target
    #     self.target = self.intervention.target.coordinates3d
    #     self._overlay_image = self.intervention.fluoroscopy.draw_target(
    #         np.array(self._overlay_image), self.target, 6, 255
    #     )
    #     print(np.array(self._overlay_image).shape)
