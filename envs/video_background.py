"""VideoBackground wrapper: replaces the DMC skybox/floor with video frames.

Uses MuJoCo's segmentation buffer to precisely separate foreground (agent
body parts) from background (skybox + floor), then composites video frames
behind the agent.  This works for any DMC domain regardless of colour scheme.

Usage:
    env = DeepMindControl("walker_walk", ...)
    env = VideoBackground(env, video_dir="kinetics400/videos/train", size=(64, 64))
"""

import glob
import os
import random

import cv2
import gymnasium as gym
import numpy as np


class VideoBackground(gym.Wrapper):
    """Replace background pixels in DMC renders with video frames.

    At reset, a random video is selected from *video_dir*. Each step
    advances the video by one frame (looping when exhausted). Background
    pixels are detected via the MuJoCo segmentation buffer and replaced.

    Background is defined as: skybox (geom_id == -1) and floor (geom_id == 0).
    """

    # MuJoCo geom IDs that we treat as "background" and replace with video.
    # -1 = skybox (no geometry), 0 = ground plane (first geom in the model).
    _BG_GEOM_IDS = {-1, 0}

    def __init__(self, env, video_dir, size=(64, 64), seed=0):
        super().__init__(env)
        self._size = size
        self._rng = random.Random(seed)
        self._video_paths = sorted(
            glob.glob(os.path.join(video_dir, "**", "*.mp4"), recursive=True)
            + glob.glob(os.path.join(video_dir, "**", "*.avi"), recursive=True)
            + glob.glob(os.path.join(video_dir, "**", "*.mkv"), recursive=True)
            + glob.glob(os.path.join(video_dir, "**", "*.webm"), recursive=True)
        )
        if not self._video_paths:
            raise FileNotFoundError(
                f"No video files found in '{video_dir}'. "
                "Run download_videos.py first or provide a directory with .mp4 files."
            )
        self._frames = None
        self._frame_idx = 0

        # Resolve the underlying DMC env to access physics for segmentation
        self._dmc_env = self._find_dmc_env(env)
        self._camera = self._dmc_env._camera

    @staticmethod
    def _find_dmc_env(env):
        """Walk the wrapper chain to find the DeepMindControl env."""
        e = env
        while hasattr(e, "env"):
            if hasattr(e, "_env") and hasattr(e._env, "physics"):
                return e
            e = e.env
        if hasattr(e, "_env") and hasattr(e._env, "physics"):
            return e
        raise RuntimeError(
            "VideoBackground requires a DeepMindControl env in the wrapper chain."
        )

    def _load_random_video(self):
        """Load frames from a random video, resized to self._size."""
        path = self._rng.choice(self._video_paths)
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(
                frame, (self._size[1], self._size[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            frames.append(frame)
        cap.release()
        if len(frames) == 0:
            raise RuntimeError(f"Could not read any frames from {path}")
        return np.stack(frames, axis=0)  # (N, H, W, 3)

    def _get_bg_mask(self):
        """Return boolean mask of background pixels using the segmentation buffer."""
        seg = self._dmc_env._env.physics.render(
            *self._size, camera_id=self._camera, segmentation=True
        )
        # seg shape: (H, W, 2) — channel 0 is geom_id, channel 1 is object type
        geom_ids = seg[:, :, 0]
        mask = np.zeros(geom_ids.shape, dtype=bool)
        for gid in self._BG_GEOM_IDS:
            mask |= (geom_ids == gid)
        return mask

    def _composite(self, image):
        """Replace background pixels with the current video frame."""
        if self._frames is None:
            return image
        video_frame = self._frames[self._frame_idx % len(self._frames)]
        self._frame_idx += 1
        mask = self._get_bg_mask()
        result = image.copy()
        result[mask] = video_frame[mask]
        return result

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._frames = self._load_random_video()
        self._frame_idx = 0
        obs["image"] = self._composite(obs["image"])
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs["image"] = self._composite(obs["image"])
        return obs, reward, done, info
