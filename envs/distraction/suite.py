# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Distracting Control Suite — environment factory.

Vendored from google-research/distracting_control with the TensorFlow
dependency removed (background.py uses Pillow + numpy instead).
"""

try:
    from dm_control import suite as _dm_suite
    from dm_control.suite.wrappers import pixels as _pixels_wrapper
except ImportError:
    _dm_suite = None
    _pixels_wrapper = None

from envs.distraction import background
from envs.distraction import camera
from envs.distraction import color
from envs.distraction import suite_utils


def is_available():
    return _dm_suite is not None


def load(domain_name,
         task_name,
         difficulty=None,
         dynamic=False,
         background_dataset_path=None,
         background_dataset_videos='train',
         background_kwargs=None,
         camera_kwargs=None,
         color_kwargs=None,
         task_kwargs=None,
         environment_kwargs=None,
         visualize_reward=False,
         render_kwargs=None,
         pixels_only=True,
         pixels_observation_key='pixels',
         env_state_wrappers=None):
    """Load a dm_control task wrapped with configurable visual distractions.

    Args:
        domain_name: dm_control domain string (e.g. 'walker').
        task_name:   dm_control task string (e.g. 'walk').
        difficulty:  One of 'easy', 'medium', 'hard', or None for raw kwargs.
        dynamic:     Whether distractors change per step (True) or are fixed
                     at episode reset (False).
        background_dataset_path: Path to DAVIS 2017 JPEGImages/480p directory.
        background_dataset_videos: 'train' | 'val' or list of video names.
        background_kwargs: Dict of overrides for DistractingBackgroundEnv.
        camera_kwargs:     Dict of overrides for DistractingCameraEnv.
        color_kwargs:      Dict of overrides for DistractingColorEnv.
        task_kwargs:       dm_control task kwargs (e.g. {'random': seed}).
        environment_kwargs: dm_control environment kwargs.
        visualize_reward:  Colour objects to indicate reward.
        render_kwargs:     Passed to the pixel wrapper (width, height, camera_id).
        pixels_only:       Exclude state observations when True.
        pixels_observation_key: Key for rendered image in the observation.
        env_state_wrappers: Extra wrappers applied before the pixel wrapper.

    Returns:
        A dm_control Environment with distractions and pixel rendering.
    """
    if not is_available():
        raise ImportError(
            "dm_control is not available. Install it with: pip install dm_control"
        )

    if difficulty not in (None, 'easy', 'medium', 'hard'):
        raise ValueError("difficulty must be one of: 'easy', 'medium', 'hard'.")

    render_kwargs = dict(render_kwargs or {})
    if 'camera_id' not in render_kwargs:
        render_kwargs['camera_id'] = 2 if domain_name == 'quadruped' else 0

    env = _dm_suite.load(
        domain_name,
        task_name,
        task_kwargs=task_kwargs,
        environment_kwargs=environment_kwargs,
        visualize_reward=visualize_reward,
    )

    # ---- Background ----
    if difficulty or background_kwargs:
        bg_path = background_dataset_path or suite_utils.DEFAULT_BACKGROUND_PATH
        final_bg_kwargs = {}
        if difficulty:
            num_videos = suite_utils.DIFFICULTY_NUM_VIDEOS[difficulty]
            final_bg_kwargs.update(
                suite_utils.get_background_kwargs(
                    domain_name, num_videos, dynamic, bg_path,
                    background_dataset_videos,
                )
            )
        else:
            final_bg_kwargs.update(
                dict(dataset_path=bg_path,
                     dataset_videos=background_dataset_videos)
            )
        if background_kwargs:
            final_bg_kwargs.update(background_kwargs)
        env = background.DistractingBackgroundEnv(env, **final_bg_kwargs)

    # ---- Camera ----
    if difficulty or camera_kwargs:
        final_cam_kwargs = dict(camera_id=render_kwargs['camera_id'])
        if difficulty:
            scale = suite_utils.DIFFICULTY_SCALE[difficulty]
            final_cam_kwargs.update(
                suite_utils.get_camera_kwargs(domain_name, scale, dynamic)
            )
        if camera_kwargs:
            final_cam_kwargs.update(camera_kwargs)
        env = camera.DistractingCameraEnv(env, **final_cam_kwargs)

    # ---- Color ----
    if difficulty or color_kwargs:
        final_color_kwargs = {}
        if difficulty:
            scale = suite_utils.DIFFICULTY_SCALE[difficulty]
            final_color_kwargs.update(suite_utils.get_color_kwargs(scale, dynamic))
        if color_kwargs:
            final_color_kwargs.update(color_kwargs)
        env = color.DistractingColorEnv(env, **final_color_kwargs)

    if env_state_wrappers:
        for wrapper in env_state_wrappers:
            env = wrapper(env)

    # Pixel wrapper must come last so all distractor mutations happen before render.
    env = _pixels_wrapper.Wrapper(
        env,
        pixels_only=pixels_only,
        render_kwargs=render_kwargs,
        observation_key=pixels_observation_key,
    )
    return env
