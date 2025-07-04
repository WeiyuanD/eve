from typing import Dict, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageChops
import numpy as np
import gymnasium as gym
from .trackingonly import TrackingOnly
from ..simulation import Simulation
from ..vesseltree import VesselTree
from ...util.coordtransform import (
    vessel_cs_to_tracking3d,
    tracking3d_to_vessel_cs,
    tracking3d_to_2d,
)


class Pillow(TrackingOnly):
    def __init__(
        self,
        simulation: Simulation,
        vessel_tree: VesselTree,
        image_frequency: float = 7.5,
        image_rot_zx: Tuple[float, float] = (0.0, 0.0),
        image_center: Optional[Tuple[float, float, float]] = None,
        field_of_view: Optional[Tuple[float, float]] = None,
        image_size: Tuple[int, int] = (96, 96),
        # image_size: Tuple[int, int] = (64, 64), # v1
        image_size_vessel: Tuple[int, int] = (512, 512),

    ) -> None:
        super().__init__(
            simulation=simulation,
            vessel_tree=vessel_tree,
            image_frequency=image_frequency,
            image_rot_zx=image_rot_zx,
            image_center=image_center,
            field_of_view=field_of_view,
        )
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = image_size
        self.image_size_vessel = image_size_vessel


        self._image_mode = "L"
        self.low = 0
        self.high = 255

        self._tracking_to_image_factor = 0
        self._tracking_to_image_factor_vessel = 0
        self._tracking_offset = [0, 0]
        self._image_offset = [0, 0]
        self._image_offset_vessel = [0, 0]
        self._image = None

    @property
    def image_space(self) -> gym.Space:
        return gym.spaces.Box(0, 255, self.image_size, dtype=np.uint8)
        # WD:
        # return gym.spaces.Box(0, 255, shape=(1, *self.image_size), dtype=np.uint8)



    @property
    def image(self) -> np.ndarray:
        # print(self._image.size)
        return self._image

    def step(self):
        trackings = self.instrument_trackings2d
        diameters = [
            instrument.tip_section.diameter_outer
            for instrument in self.simulation.instruments
        ]
        # Noise is around colour 128.
        # noise_image = Image.effect_noise(size=self.image_size, sigma=0.3)
        # canvas_image = Image.new('L', size=self.image_size, color=0)
        physics_image = self._render(trackings, diameters)
        # image = ImageChops.darker(physics_image, noise_image)
        # self._image = np.asarray(image, dtype=np.uint8)
        self._image = np.asarray(physics_image, dtype=np.uint8)


    def reset(self, episode_nr: int = 0) -> None:
        coords_high = self.tracking2d_space.high
        coords_low = self.tracking2d_space.low
        intervention_size_x = coords_high[0] - coords_low[0]
        intervention_size_y = coords_high[1] - coords_low[1]
        x_factor = self.image_size[0] / intervention_size_x
        y_factor = self.image_size[1] / intervention_size_y
        self._tracking_to_image_factor = min(x_factor, y_factor)
        self._tracking_offset = np.array([-coords_low[0], -coords_low[1]])
        x_image_offset = (
            self.image_size[0] - intervention_size_x * self._tracking_to_image_factor
        ) / 2
        y_image_offset = (
            self.image_size[1] - intervention_size_y * self._tracking_to_image_factor
        ) / 2
        self._image_offset = np.array([x_image_offset, y_image_offset])
        # WD:
        x_factor = self.image_size_vessel[0] / intervention_size_x
        y_factor = self.image_size_vessel[1] / intervention_size_y
        self._tracking_to_image_factor_vessel = min(x_factor, y_factor)
        x_image_offset = (
            self.image_size_vessel[0] - intervention_size_x * self._tracking_to_image_factor_vessel
        ) / 2
        y_image_offset = (
            self.image_size_vessel[1] - intervention_size_y * self._tracking_to_image_factor_vessel
        ) / 2
        self._image_offset_vessel = np.array([x_image_offset, y_image_offset])
        self.step()

    def _render(
        self, trackings: Dict[str, np.ndarray], diameters: Dict[str, float]
    ) -> None:
        physics_image = Image.new(
            mode=self._image_mode, size=self.image_size, color=255
        )
        lines = zip(trackings, diameters)


        # lines = [
        #     [tracking, diameter] for tracking, diameter in zip(trackings, diameters)
        # ]


        lines = sorted(lines, key=lambda line: line[1])
        for i, line in enumerate(lines):
            diameter = int(np.round(line[1] * self._tracking_to_image_factor))
            coord_points = line[0]
            if i < len(trackings) - 1:
                end = coord_points.shape[0] - trackings[i + 1][0].shape[0]
            else:
                end = coord_points.shape[0] - 1
            self._draw_lines(
                physics_image,
                coord_points[:end],
                int(diameter),
                grey_value=40,
            )
        return physics_image

    def _draw_circle(
        self,
        image: np.ndarray,
        position: np.ndarray,
        radius: float,
        grey_value: int,
    ) -> np.ndarray:
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        # WD
        position = tracking3d_to_2d(position)
        position = position.reshape(1, -1)  # Shape: (1, 2)

        position = self._coord_transform_tracking_to_image(position, is_vessel=True) # v2
        # position = self._coord_transform_tracking_to_image(position) # v1
        # radius *= self._tracking_to_image_factor # v1
        radius *= self._tracking_to_image_factor_vessel # v2
        circle_bb_low = (coord - radius for coord in position)
        circle_bb_high = (coord + radius for coord in position)
        # draw.ellipse([circle_bb_low, circle_bb_high], fill=grey_value)
        # WD:
        xy = np.concatenate((*circle_bb_low, *circle_bb_high))
        xy = xy.tolist()
        draw.ellipse(xy, fill=grey_value)
        return np.asarray(image)


    def _draw_lines(
        self,
        image: Image,
        point_cloud: np.ndarray,
        width=1,
        grey_value=0,
    ) -> np.ndarray:
        draw = ImageDraw.Draw(image)
        point_cloud_image = self._coord_transform_tracking_to_image(point_cloud, is_vessel=False) #v2
        # point_cloud_image = self._coord_transform_tracking_to_image(point_cloud) # v1
        draw.line(point_cloud_image, fill=grey_value, width=width, joint="curve")
        return np.asarray(image)

    # v1
    # def _coord_transform_tracking_to_image(
    #     self, coords: np.ndarray
    # ) -> List[Tuple[float, float]]:
    #     coords_image = (coords + self._tracking_offset) * self._tracking_to_image_factor
    #     coords_image += self._image_offset
    #     coords_image = np.round(coords_image, decimals=0).astype(np.int64)
    #     coords_image[:, 1] = -coords_image[:, 1] + self.image_size[1]
    #     coords_image = [(coord[0], coord[1]) for coord in coords_image]
    #     return coords_image

    # WD: v2
    def _coord_transform_tracking_to_image(
        self, coords: np.ndarray, is_vessel: bool
    ) -> List[Tuple[float, float]]:
        if is_vessel:
            coords_image = (coords + self._tracking_offset) * self._tracking_to_image_factor_vessel
            coords_image += self._image_offset_vessel
            coords_image = np.round(coords_image, decimals=0).astype(np.int64)
            coords_image[:, 1] = -coords_image[:, 1] + self.image_size_vessel[1]
        else:
            coords_image = (coords + self._tracking_offset) * self._tracking_to_image_factor
            coords_image += self._image_offset
            coords_image = np.round(coords_image, decimals=0).astype(np.int64)
            coords_image[:, 1] = -coords_image[:, 1] + self.image_size[1]

        coords_image = [(coord[0], coord[1]) for coord in coords_image]
        return coords_image

    def draw_target(
        self,
        image: np.ndarray,
        position: np.ndarray,
        radius: float,
        grey_value: int,
    ) -> np.ndarray:
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        # position = vessel_cs_to_tracking3d(
        #     position,
        #     # self.image_rot_zx,
        #     # (-self.image_rot_zx[0], 0),
        #     (0, 0),
        #     self.image_center,
        #     # (0.0, 0.0, 0.0),
        #     self.field_of_view,
        # )
        # position = tracking3d_to_vessel_cs(
        #     position, self.image_rot_zx, self.image_center
        # )
        position = tracking3d_to_2d(position)
        position = position.reshape(1, -1)  # Shape: (1, 2)
        position = self._coord_transform_tracking_to_image(position, is_vessel=False) # v2
        # position = self._coord_transform_tracking_to_image(position) # v1
        radius *= self._tracking_to_image_factor
        circle_bb_low = (coord - radius for coord in position)
        circle_bb_high = (coord + radius for coord in position)
        xy = np.concatenate((*circle_bb_low, *circle_bb_high))
        xy = xy.tolist()
        draw.ellipse(xy, fill=grey_value)
        return np.asarray(image)
