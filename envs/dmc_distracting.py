"""DeepMindControl wrapper using the Distracting Control Suite.

Wraps distracting_control.suite.load() to produce the same observation
interface as envs/dmc.py so the rest of the training pipeline is unchanged.

The distracting_control library adds three types of visual distractions on top
of standard dm_control tasks:
  - background  : DAVIS video clips replace the skybox/floor
  - camera      : random pose jitter each step
  - color       : body colour randomisation each step

Usage:
    env = DistractingControl(
        "walker_walk",
        difficulty="easy",
        background_dataset_path="/data/DAVIS/JPEGImages/480p",
        dynamic=True,
        action_repeat=2,
        size=(64, 64),
        seed=0,
    )

Install distracting_control:
    pip install git+https://github.com/google-research/distracting_control.git

Download DAVIS 2017 480p TrainVal and point background_dataset_path to the
480p/JPEGImages directory.
"""

import gymnasium as gym
import numpy as np


class DistractingControl(gym.Env):
    """dm_control environment wrapped with distracting_control distractors.

    Produces the same observation dict as DeepMindControl:
      image        : (H, W, 3) uint8
      <proprio keys>: float32 vectors
      is_first     : bool
      is_last      : bool
      is_terminal  : bool
    """

    metadata = {}

    def __init__(
        self,
        name: str,
        difficulty: str = "easy",
        background_dataset_path: str = None,
        background_dataset_videos: str = "train",
        dynamic: bool = True,
        # Which distractors to enable (all on by default)
        background: bool = True,
        camera: bool = True,
        color: bool = True,
        # Floor video distraction
        floor_video: bool = False,
        floor_video_alpha: float = 1.0,
        ground_plane_alpha: float = None,
        action_repeat: int = 1,
        size: tuple = (64, 64),
        camera_id: int = None,
        seed: int = 0,
    ):
        from envs.distraction import suite as dc_suite
        from envs.distraction import suite_utils

        domain, task = name.rsplit("_", 1)

        # Build per-distractor kwargs only for enabled distractors.
        # We pass difficulty=None to suite.load and manually construct kwargs
        # so that disabled distractors are fully skipped (not just scaled to 0).
        scale = suite_utils.DIFFICULTY_SCALE[difficulty]
        num_videos = suite_utils.DIFFICULTY_NUM_VIDEOS[difficulty]

        background_kwargs = None
        camera_kwargs = None
        color_kwargs = None

        if background:
            if background_dataset_path is None:
                raise ValueError(
                    "background_dataset_path must be provided when background=True. "
                    "Download DAVIS 2017 480p TrainVal and set background_dataset_path "
                    "to the JPEGImages/480p directory."
                )
            bg_extra = {}
            if ground_plane_alpha is not None:
                bg_extra["ground_plane_alpha"] = ground_plane_alpha
            background_kwargs = suite_utils.get_background_kwargs(
                domain, num_videos, dynamic, background_dataset_path,
                background_dataset_videos,
                floor_video=floor_video,
                floor_video_alpha=floor_video_alpha,
                **bg_extra,
            )
        if camera:
            camera_kwargs = suite_utils.get_camera_kwargs(domain, scale, dynamic)
        if color:
            color_kwargs = suite_utils.get_color_kwargs(scale, dynamic)

        dc_kwargs = dict(
            domain_name=domain,
            task_name=task,
            difficulty=None,  # we handle difficulty manually per distractor
            dynamic=dynamic,
            task_kwargs={"random": seed},
            pixels_only=False,  # keep proprioceptive observations
            render_kwargs={"width": size[1], "height": size[0]},
        )
        if background_kwargs is not None:
            dc_kwargs["background_kwargs"] = background_kwargs
        if background_dataset_path is not None:
            dc_kwargs["background_dataset_path"] = background_dataset_path
            dc_kwargs["background_dataset_videos"] = background_dataset_videos
        if camera_kwargs is not None:
            dc_kwargs["camera_kwargs"] = camera_kwargs
        if color_kwargs is not None:
            dc_kwargs["color_kwargs"] = color_kwargs

        self._env = dc_suite.load(**dc_kwargs)
        self._action_repeat = action_repeat
        self._size = size

        if camera_id is None:
            camera_id = dict(quadruped=2, fish=3).get(domain, 0)
        self._camera_id = camera_id

        self.reward_range = [-np.inf, np.inf]

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    @property
    def observation_space(self):
        spaces = {}
        for key, value in self._env.observation_spec().items():
            if key == "pixels":
                continue  # handled separately as "image"
            shape = value.shape if len(value.shape) > 0 else (1,)
            spaces[key] = gym.spaces.Box(-np.inf, np.inf, shape, dtype=np.float32)
        spaces["image"] = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        spec = self._env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _obs_from_timestep(self, time_step):
        obs = dict(time_step.observation)
        # The pixel wrapper already renders at our configured size — reuse it
        # instead of calling physics.render() a second time.
        pixels = obs.pop("pixels", None)
        obs = {k: np.array([v]) if np.array(v).ndim == 0 else np.array(v)
               for k, v in obs.items()}
        if pixels is not None:
            obs["image"] = pixels
        else:
            obs["image"] = self._render()
        obs["is_terminal"] = False if time_step.first() else time_step.discount == 0
        obs["is_first"] = time_step.first()
        obs["is_last"] = time_step.last()
        return obs

    def _render(self):
        # Fallback: render directly (used by render() public method).
        return self._env.physics.render(*self._size, camera_id=self._camera_id)

    # ------------------------------------------------------------------
    # Core step / reset
    # ------------------------------------------------------------------

    def step(self, action):
        assert np.isfinite(action).all(), action
        reward = 0.0
        for _ in range(self._action_repeat):
            time_step = self._env.step(action)
            reward += time_step.reward or 0.0
            if time_step.last():
                break
        obs = self._obs_from_timestep(time_step)
        done = time_step.last()
        info = {"discount": np.array(time_step.discount, np.float32)}
        return obs, reward, done, info

    def reset(self, **kwargs):
        time_step = self._env.reset()
        return self._obs_from_timestep(time_step)

    def render(self, *args, **kwargs):
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._render()
