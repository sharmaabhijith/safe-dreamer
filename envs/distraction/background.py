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

"""A wrapper for dm_control environments which applies background distractions.

Vendored and adapted from the original to remove the TensorFlow dependency;
image I/O and resizing are done with Pillow + numpy instead.
"""

import collections
import os

import numpy as np
from PIL import Image
from dm_control.mujoco.wrapper import mjbindings
from dm_control.rl import control

DAVIS17_TRAINING_VIDEOS = [
    'bear', 'bmx-bumps', 'boat', 'boxing-fisheye', 'breakdance-flare', 'bus',
    'car-turn', 'cat-girl', 'classic-car', 'color-run', 'crossing',
    'dance-jump', 'dancing', 'disc-jockey', 'dog-agility', 'dog-gooses',
    'dogs-scale', 'drift-turn', 'drone', 'elephant', 'flamingo', 'hike',
    'hockey', 'horsejump-low', 'kid-football', 'kite-walk', 'koala',
    'lady-running', 'lindy-hop', 'longboard', 'lucia', 'mallard-fly',
    'mallard-water', 'miami-surf', 'motocross-bumps', 'motorbike', 'night-race',
    'paragliding', 'planes-water', 'rallye', 'rhino', 'rollerblade',
    'schoolgirls', 'scooter-board', 'scooter-gray', 'sheep', 'skate-park',
    'snowboard', 'soccerball', 'stroller', 'stunt', 'surf', 'swing', 'tennis',
    'tractor-sand', 'train', 'tuk-tuk', 'upside-down', 'varanus-cage', 'walking',
]
DAVIS17_VALIDATION_VIDEOS = [
    'bike-packing', 'blackswan', 'bmx-trees', 'breakdance', 'camel',
    'car-roundabout', 'car-shadow', 'cows', 'dance-twirl', 'dog', 'dogs-jump',
    'drift-chicane', 'drift-straight', 'goat', 'gold-fish', 'horsejump-high',
    'india', 'judo', 'kite-surf', 'lab-coat', 'libby', 'loading', 'mbike-trick',
    'motocross-jump', 'paragliding-launch', 'parkour', 'pigs', 'scooter-black',
    'shooting', 'soapbox',
]

SKY_TEXTURE_INDEX = 0
FLOOR_MATERIAL_NAME = 'grid'
Texture = collections.namedtuple('Texture', ('size', 'address', 'textures'))


def _imread(path):
    """Load an image as a uint8 numpy array (H, W, 3)."""
    if isinstance(path, bytes):
        path = path.decode()
    img = Image.open(path).convert('RGB')
    return np.asarray(img, dtype=np.uint8)


def _size_and_flatten(image, ref_height, ref_width):
    """Resize image to (ref_height, ref_width) and flatten to 1-D."""
    h, w = image.shape[:2]
    if h != ref_height or w != ref_width:
        image = np.asarray(
            Image.fromarray(image).resize((ref_width, ref_height), Image.BILINEAR),
            dtype=np.uint8,
        )
    return image.reshape(-1)


def _blend_to_background(alpha, image, background):
    if alpha == 1.0:
        return image
    if alpha == 0.0:
        return background
    return (alpha * image.astype(np.float32)
            + (1.0 - alpha) * background.astype(np.float32)).astype(np.uint8)


def _listdir_sorted(path):
    """Return sorted list of files in *path*, filtering out directories."""
    return sorted(
        f if isinstance(f, str) else f.decode()
        for f in os.listdir(path)
        if os.path.isfile(os.path.join(
            path, f if isinstance(f, str) else f.decode()
        ))
    )


class DistractingBackgroundEnv(control.Environment):
    """Environment wrapper for background visual distraction.

    **NOTE**: Apply BEFORE the pixel wrapper so background changes are rendered.
    """

    def __init__(self, env, dataset_path=None, dataset_videos=None,
                 video_alpha=1.0, ground_plane_alpha=1.0, num_videos=None,
                 dynamic=False, seed=None, shuffle_buffer_size=None,
                 floor_video=False, floor_video_alpha=1.0):
        if not 0 <= video_alpha <= 1:
            raise ValueError('`video_alpha` must be in [0, 1].')
        if not 0 <= floor_video_alpha <= 1:
            raise ValueError('`floor_video_alpha` must be in [0, 1].')

        self._env = env
        self._video_alpha = video_alpha
        self._ground_plane_alpha = ground_plane_alpha
        self._floor_video = floor_video
        self._floor_video_alpha = floor_video_alpha
        self._random_state = np.random.RandomState(seed=seed)
        self._dynamic = dynamic
        self._shuffle_buffer_size = shuffle_buffer_size
        self._background = None
        self._floor_background = None
        self._current_img_index = 0
        self._step_direction = 1

        if not dataset_path or num_videos == 0:
            self._video_paths = []
        else:
            if not dataset_videos:
                dataset_videos = sorted(
                    e if isinstance(e, str) else e.decode()
                    for e in os.listdir(dataset_path)
                )
            elif dataset_videos in ('train', 'training'):
                dataset_videos = DAVIS17_TRAINING_VIDEOS
            elif dataset_videos in ('val', 'validation'):
                dataset_videos = DAVIS17_VALIDATION_VIDEOS

            video_paths = [
                os.path.join(dataset_path, subdir)
                for subdir in dataset_videos
                if os.path.isdir(os.path.join(dataset_path, subdir))
            ]

            if num_videos is not None:
                if num_videos > len(video_paths) or num_videos < 0:
                    raise ValueError(
                        f'`num_videos` is {num_videos} but must be in '
                        f'[0, {len(video_paths)}].'
                    )
                video_paths = video_paths[:num_videos]

            self._video_paths = video_paths

    # ------------------------------------------------------------------
    def reset(self):
        time_step = self._env.reset()
        self._reset_background()
        return time_step

    def _reset_background(self):
        if self._ground_plane_alpha is not None:
            self._env.physics.named.model.mat_rgba['grid', 'a'] = self._ground_plane_alpha

        # Resize sky texture height to something sensible (avoids MuJoCo quirks).
        self._env.physics.model.tex_height[SKY_TEXTURE_INDEX] = 800

        sky_height = self._env.physics.model.tex_height[SKY_TEXTURE_INDEX]
        sky_width  = self._env.physics.model.tex_width[SKY_TEXTURE_INDEX]
        sky_nchan  = self._env.physics.model.tex_nchannel[SKY_TEXTURE_INDEX]
        sky_size   = sky_height * sky_width * sky_nchan
        sky_addr   = self._env.physics.model.tex_adr[SKY_TEXTURE_INDEX]
        sky_texture = self._env.physics.model.tex_data[
            sky_addr : sky_addr + sky_size
        ].astype(np.float32)

        if self._video_paths:
            if self._shuffle_buffer_size:
                file_names = [
                    os.path.join(path, fn)
                    for path in self._video_paths
                    for fn in _listdir_sorted(path)
                ]
                self._random_state.shuffle(file_names)
                file_names = file_names[: self._shuffle_buffer_size]
                images = [_imread(fn) for fn in file_names]
            else:
                video_path = self._random_state.choice(self._video_paths)
                file_names = _listdir_sorted(video_path)
                if not self._dynamic:
                    file_names = [self._random_state.choice(file_names)]
                images = [_imread(os.path.join(video_path, fn)) for fn in file_names]

            self._current_img_index = self._random_state.choice(len(images))
            self._step_direction = self._random_state.choice([-1, 1])

            texturized_images = []
            for image in images:
                flat = _size_and_flatten(image, sky_height, sky_width)
                new_tex = _blend_to_background(self._video_alpha, flat, sky_texture)
                texturized_images.append(new_tex)
        else:
            self._current_img_index = 0
            texturized_images = [sky_texture]

        self._background = Texture(sky_size, sky_addr, texturized_images)

        # ---- Floor texture ----
        self._floor_background = None
        if self._floor_video and self._video_paths:
            floor_tex_index = self._find_floor_texture_index()
            if floor_tex_index is not None:
                fl_height = self._env.physics.model.tex_height[floor_tex_index]
                fl_width  = self._env.physics.model.tex_width[floor_tex_index]
                fl_nchan  = self._env.physics.model.tex_nchannel[floor_tex_index]
                fl_size   = fl_height * fl_width * fl_nchan
                fl_addr   = self._env.physics.model.tex_adr[floor_tex_index]
                fl_texture = self._env.physics.model.tex_data[
                    fl_addr : fl_addr + fl_size
                ].astype(np.float32)

                # Pick a different (or same) random video for the floor
                floor_video_path = self._random_state.choice(self._video_paths)
                floor_file_names = _listdir_sorted(floor_video_path)
                if not self._dynamic:
                    floor_file_names = [self._random_state.choice(floor_file_names)]
                floor_images = [_imread(os.path.join(floor_video_path, fn))
                                for fn in floor_file_names]

                floor_texturized = []
                for image in floor_images:
                    flat = _size_and_flatten(image, fl_height, fl_width)
                    new_tex = _blend_to_background(self._floor_video_alpha, flat, fl_texture)
                    floor_texturized.append(new_tex)

                self._floor_background = Texture(fl_size, fl_addr, floor_texturized)

        self._apply()

    def _find_floor_texture_index(self):
        """Return the MuJoCo texture index used by the 'grid' floor material, or None."""
        model = self._env.physics.model
        try:
            # mat_texid is 2D (nmat x ntexrole); the diffuse slot (index 0 in the
            # row, but MuJoCo flattens it differently) — just scan the row for the
            # first non-negative value.
            mat_names = [model.id2name(i, 'material')
                         for i in range(model.nmat)]
            if FLOOR_MATERIAL_NAME in mat_names:
                mat_id = mat_names.index(FLOOR_MATERIAL_NAME)
                row = model.mat_texid[mat_id]  # array of texids per role slot
                for tex_id in row.flat:
                    if tex_id >= 0:
                        return int(tex_id)
        except Exception:
            pass
        return None

    def step(self, action):
        time_step = self._env.step(action)

        if time_step.first():
            self._reset_background()
            return time_step

        if self._dynamic and self._video_paths:
            self._current_img_index += self._step_direction

            if self._current_img_index <= 0:
                self._current_img_index = 0
                self._step_direction = abs(self._step_direction)

            if self._current_img_index >= len(self._background.textures):
                self._current_img_index = len(self._background.textures) - 1
                self._step_direction = -abs(self._step_direction)

            self._apply()

        return time_step

    def _apply(self):
        if self._background is None:
            return
        start = self._background.address
        end   = self._background.address + self._background.size
        self._env.physics.model.tex_data[start:end] = (
            self._background.textures[self._current_img_index]
        )
        with self._env.physics.contexts.gl.make_current() as ctx:
            ctx.call(
                mjbindings.mjlib.mjr_uploadTexture,
                self._env.physics.model.ptr,
                self._env.physics.contexts.mujoco.ptr,
                SKY_TEXTURE_INDEX,
            )

        # Apply floor video texture if enabled
        if self._floor_background is not None:
            fl_idx = min(self._current_img_index,
                         len(self._floor_background.textures) - 1)
            fl_start = self._floor_background.address
            fl_end   = self._floor_background.address + self._floor_background.size
            self._env.physics.model.tex_data[fl_start:fl_end] = (
                self._floor_background.textures[fl_idx]
            )
            floor_tex_index = self._find_floor_texture_index()
            if floor_tex_index is not None:
                with self._env.physics.contexts.gl.make_current() as ctx:
                    ctx.call(
                        mjbindings.mjlib.mjr_uploadTexture,
                        self._env.physics.model.ptr,
                        self._env.physics.contexts.mujoco.ptr,
                        floor_tex_index,
                    )

    def __getattr__(self, attr):
        if hasattr(self._env, attr):
            return getattr(self._env, attr)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{attr}'"
        )
