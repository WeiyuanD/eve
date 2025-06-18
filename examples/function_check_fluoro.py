# pylint: disable=no-member

from time import perf_counter
import eve.observation
import numpy as np
import eve
from eve.visualisation import FromImaging
from eve.visualisation.sofapygame import SofaPygame
from PIL import Image
from scipy.ndimage import gaussian_filter, distance_transform_edt, sobel

import importlib

# Set the Matplotlib backend dynamically
matplotlib = importlib.import_module("matplotlib")
matplotlib.use("TkAgg")  # Must be called before importing pyplot

import matplotlib.pyplot as plt

fig, ax = plt.subplots()

def render_full(image):
    ax.clear()
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.000001)

def crop_vessel(image):
    image = image.astype(np.uint8)
    pixels = image.flatten()
    counts = np.bincount(pixels)
    threshold = np.argmax(counts)
    mask = (image > (threshold + 1)) | (image < (threshold - 1))
    print(mask.size)
    coords = np.argwhere(mask)
    print(coords.shape[0], image.size)
    if coords.size>0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)+1
        height = y1 - y0
        width = x1 - x0
        size = max(height, width)

        center_y = (y0 + y1) // 2
        center_x = (x0 + x1) // 2

        # Calculate new square bounds
        half_size = size // 2
        new_y0 = max(center_y - half_size, 0)
        new_y1 = min(center_y + half_size, image.shape[0])
        new_x0 = max(center_x - half_size, 0)
        new_x1 = min(center_x + half_size, image.shape[1])

        # Ensure output is exactly square even if at border
        cropped_img = image[new_y0:new_y0 + size, new_x0:new_x0 + size]
        resized_img = Image.fromarray(cropped_img).resize((96, 96), resample=Image.Resampling.LANCZOS)
    return np.array(resized_img)
    # return mask

def delete_lines(n=1):
    for _ in range(n):
        print("\033[1A\x1b[2K", end="")

def soft_sdf(image):
    dx = sobel(img, axis=0, mode='reflect')
    dy = sobel(img, axis=1, mode='reflect')
    edge = np.hypot(dx, dy)
    edge = (edge/edge.max()) # normalize to 0-1

    edge_binary = edge < 0.1
    pseudo_sdf = distance_transform_edt(edge_binary)
    pseudo_sdf = gaussian_filter(pseudo_sdf, sigma=1)

    pseudo_sdf_normalized = (pseudo_sdf - pseudo_sdf.min()) / (pseudo_sdf.max() - pseudo_sdf.min()) * 255
    pseudo_sdf_image = Image.fromarray(pseudo_sdf_normalized.astype(np.uint8))
    pseudo_sdf_resized = pseudo_sdf_image.resize((96, 96), resample=Image.Resampling.LANCZOS)
    return np.array(pseudo_sdf_resized)

def lanczo(image):
    image = Image.fromarray(image.astype(np.uint8))
    resized_image = image.resize((96, 96), resample=Image.Resampling.LANCZOS)
    return np.array(resized_image)
# # Define Intervention
# vessel_tree = eve.intervention.vesseltree.AorticArch(
#     seed=30,
#     scaling_xyzd=[1.0, 1.0, 1.0, 0.75],
# )

# WD:
vessel_tree = eve.intervention.vesseltree.AorticArchRandom(
    episodes_between_change=1,
    scale_diameter_array=[0.85],
    arch_types_filter=[eve.intervention.vesseltree.ArchType.I],
    )

instrument = eve.intervention.instrument.Angled()

simulation = eve.intervention.simulation.SofaBeamAdapter(friction=0.001)

fluoroscopy = eve.intervention.fluoroscopy.Pillow(
    simulation=simulation,
    vessel_tree=vessel_tree,
    image_frequency=7.5,
    image_rot_zx=[0, 0],
)

target = eve.intervention.target.CenterlineRandom(
    vessel_tree=vessel_tree,
    fluoroscopy=fluoroscopy,
    threshold=5,
    branches=["lcca", "rcca", "lsa", "rsa", "bct", "co"],
)


intervention = eve.intervention.MonoPlaneStatic(
    vessel_tree=vessel_tree,
    instruments=[instrument],
    simulation=simulation,
    fluoroscopy=fluoroscopy,
    target=target,
)

# Helper Objects
start = eve.start.MaxInstrumentLength(intervention=intervention, max_length=500)
pathfinder = eve.pathfinder.BruteForceBFS(intervention=intervention)

# Define Observation
position = eve.observation.Tracking2D(intervention=intervention, n_points=5)
position = eve.observation.wrapper.NormalizeTracking2DEpisode(position, intervention)
target_state = eve.observation.Target2D(intervention=intervention)
target_state = eve.observation.wrapper.NormalizeTracking2DEpisode(
    target_state, intervention
)
rotation = eve.observation.Rotations(intervention=intervention)
# state = eve.observation.ObsDict(
#     {"position": position, "target": target_state, "rotation": rotation}
# )

# Define Observation
wrapped_image = eve.observation.Image(intervention=intervention)
wrapped_image = eve.observation.imagewrapper.ShowVessels(intervention, wrapped_image)
image = eve.observation.ObsDict(
    {"image": wrapped_image}
)

# Define Reward
target_reward = eve.reward.TargetReached(
    intervention=intervention,
    factor=1.0,
)
path_delta = eve.reward.PathLengthDelta(
    pathfinder=pathfinder,
    factor=0.01,
)
reward = eve.reward.Combination([target_reward, path_delta])


# Define Terminal and Truncation
target_reached = eve.terminal.TargetReached(intervention=intervention)
max_steps = eve.truncation.MaxSteps(200)

# Add Visualisation
# visualisation = FromImaging(intervention=intervention)
visualisation = SofaPygame(intervention=intervention)

# Combine everything in an env
env = eve.Env(
    intervention=intervention,
    # observation=state,
    observation=image,
    reward=reward,
    terminal=target_reached,
    truncation=max_steps,
    visualisation=visualisation,
    start=start,
    pathfinder=pathfinder,
)


n_steps = 0
r_cum = 0.0


for _ in range(12):
    env.reset()
    print("")

    for _ in range(80):
        start = perf_counter()
        trans = 35.0
        rot = 1.0
        action = (trans, rot)
        obs, reward, terminal, truncation, info = env.step(action=action)
        env.render()

        img = obs['image']  # shape: (128, 128), dtype=float32
        # img = soft_sdf(img)
        # img = lanczo(img)

        img = crop_vessel(img)
        render_full(img)
        
        # print(img.shape, img.min(), img.max())

        # plt.imsave('/home/mtgroup/logdir/steve_test/output_image2.png', img, cmap='gray')
    
        # img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
        # img_uint8 = (img_norm * 255).astype(np.uint8)
        # # Step 3: Convert to PIL image (mode 'L' = grayscale)
        # image = Image.fromarray(img_uint8, mode='L')
        # image.save('/home/mtgroup/logdir/steve_test/output_image.png')

        n_steps += 1

    # delete_lines(11)
    # print(f"Observation: \n {obs}\n")
    # print(f"Reward: {reward:.2f}\n")
    # print(f"FPS: {1/(perf_counter()-start):.2f} Hz")
env.close()
