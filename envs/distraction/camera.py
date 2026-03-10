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

"""A wrapper for dm_control environments which applies camera distractions."""

import copy
from dm_control.rl import control
import numpy as np

CAMERA_MODES = ['fixed', 'track', 'trackcom', 'targetbody', 'targetbodycom']


def eul2mat(theta):
    return np.array([
        [
            np.cos(theta[1]) * np.cos(theta[2]),
            np.sin(theta[0]) * np.sin(theta[1]) * np.cos(theta[2]) - np.sin(theta[2]) * np.cos(theta[0]),
            np.sin(theta[1]) * np.cos(theta[0]) * np.cos(theta[2]) + np.sin(theta[0]) * np.sin(theta[2]),
        ],
        [
            np.sin(theta[2]) * np.cos(theta[1]),
            np.sin(theta[0]) * np.sin(theta[1]) * np.sin(theta[2]) + np.cos(theta[0]) * np.cos(theta[2]),
            np.sin(theta[1]) * np.sin(theta[2]) * np.cos(theta[0]) - np.sin(theta[0]) * np.cos(theta[2]),
        ],
        [
            -np.sin(theta[1]),
            np.sin(theta[0]) * np.cos(theta[1]),
            np.cos(theta[0]) * np.cos(theta[1]),
        ],
    ])


def _mat_from_theta(cos_theta, sin_theta, a):
    row1 = [
        cos_theta + a[0] ** 2.0 * (1.0 - cos_theta),
        a[0] * a[1] * (1 - cos_theta) - a[2] * sin_theta,
        a[0] * a[2] * (1 - cos_theta) + a[1] * sin_theta,
    ]
    row2 = [
        a[1] * a[0] * (1 - cos_theta) + a[2] * sin_theta,
        cos_theta + a[1] ** 2.0 * (1 - cos_theta),
        a[1] * a[2] * (1.0 - cos_theta) - a[0] * sin_theta,
    ]
    row3 = [
        a[2] * a[0] * (1.0 - cos_theta) - a[1] * sin_theta,
        a[2] * a[1] * (1.0 - cos_theta) + a[0] * sin_theta,
        cos_theta + (a[2] ** 2.0) * (1.0 - cos_theta),
    ]
    return np.stack([row1, row2, row3])


def rotvec2mat(theta, vec):
    a = vec / np.sqrt(np.sum(vec ** 2.0))
    return _mat_from_theta(np.cos(theta), np.sin(theta), a)


def get_lookat_xmat_no_roll(agent_pos, camera_pos):
    delta_vec = agent_pos - camera_pos
    delta_vec /= np.sqrt(np.sum(delta_vec ** 2.0))
    yaw = np.arctan2(delta_vec[0], delta_vec[1])
    pitch = np.arctan2(delta_vec[2], np.sqrt(np.sum(delta_vec[:2] ** 2.0)))
    pitch += np.pi / 2.0
    return eul2mat([pitch, 0.0, -yaw]).flatten()


def get_lookat_xmat(agent_pos, camera_pos):
    delta_vec = agent_pos - camera_pos
    delta_vec /= np.sqrt(np.sum(delta_vec ** 2.0))
    y_vec = np.array([0.0, 0.0, -1.0])
    a = np.cross(y_vec, delta_vec)
    sin_theta = np.sqrt(np.sum(a ** 2.0))
    cos_theta = np.dot(delta_vec, y_vec)
    a /= np.sqrt(np.sum(a ** 2.0)) + 1e-4
    return _mat_from_theta(cos_theta, sin_theta, a)


def cart2sphere(cart):
    r = np.sqrt(np.sum(cart ** 2.0))
    h_angle = np.arctan2(cart[1], cart[0])
    v_angle = np.arctan2(np.sqrt(np.sum(cart[:2] ** 2.0)), cart[2])
    return np.array([r, h_angle, v_angle])


def sphere2cart(sphere):
    r, h_angle, v_angle = sphere
    x = r * np.sin(v_angle) * np.cos(h_angle)
    y = r * np.sin(v_angle) * np.sin(h_angle)
    z = r * np.cos(v_angle)
    return np.array([x, y, z])


def clip_cam_position(position, min_radius, max_radius, min_h_angle,
                      max_h_angle, min_v_angle, max_v_angle):
    return [
        np.clip(position[0], min_radius, max_radius),
        np.clip(position[1], min_h_angle, max_h_angle),
        np.clip(position[2], min_v_angle, max_v_angle),
    ]


def get_lookat_point(physics, camera_id):
    dist_to_agent = (
        physics.named.data.cam_xpos[camera_id]
        - physics.named.data.subtree_com[1]
    )
    dist_to_agent = np.sqrt(np.sum(dist_to_agent ** 2.0))
    initial_viewing_mat = copy.deepcopy(
        physics.named.data.cam_xmat[camera_id]
    ).reshape(3, 3)
    z_vec = np.array([0.0, 0.0, -dist_to_agent])
    return np.dot(initial_viewing_mat, z_vec) + physics.named.data.cam_xpos[camera_id]


class DistractingCameraEnv(control.Environment):
    """Environment wrapper for camera pose visual distraction.

    **NOTE**: Apply BEFORE the pixel wrapper so camera changes are rendered.
    """

    def __init__(self, env, camera_id, horizontal_delta, vertical_delta,
                 max_vel, vel_std, roll_delta, max_roll_vel, roll_std,
                 max_zoom_in_percent, max_zoom_out_percent,
                 limit_to_upper_quadrant=False, seed=None):
        self._env = env
        self._camera_id = camera_id
        self._horizontal_delta = horizontal_delta
        self._vertical_delta = vertical_delta
        self._max_vel = max_vel
        self._vel_std = vel_std
        self._roll_delta = roll_delta
        self._max_roll_vel = max_roll_vel
        self._roll_vel_std = roll_std
        self._max_zoom_in_percent = max_zoom_in_percent
        self._max_zoom_out_percent = max_zoom_out_percent
        self._limit_to_upper_quadrant = limit_to_upper_quadrant
        self._random_state = np.random.RandomState(seed=seed)

        self._camera_type = None
        self._cam_initial_lookat_point = None
        self._camera_vel = None
        self._radius = None
        self._h_angle = None
        self._v_angle = None
        self._roll = None
        self._roll_vel = None
        self._vel_scaling = None

    def setup_camera(self):
        mode = self._env._physics.model.cam_mode[0]
        camera_type = CAMERA_MODES[mode]
        assert camera_type in ['fixed', 'trackcom']
        self._camera_type = camera_type
        self._cam_initial_lookat_point = get_lookat_point(
            self._env.physics, self._camera_id
        )

        start_pos = copy.deepcopy(
            self._env.physics.named.data.cam_xpos[self._camera_id]
        )
        if self._camera_type != 'fixed':
            start_pos -= self._env.physics.named.data.subtree_com[1]

        start_r, start_h_angle, start_v_angle = cart2sphere(start_pos)
        self._vel_scaling = start_r / 4.0

        self._max_h_angle = start_h_angle + self._horizontal_delta
        self._min_h_angle = start_h_angle - self._horizontal_delta
        self._max_v_angle = start_v_angle + self._vertical_delta
        self._min_v_angle = start_v_angle - self._vertical_delta

        if self._limit_to_upper_quadrant:
            self._max_v_angle = min(self._max_v_angle, np.pi / 2.0)
            self._min_v_angle = max(self._min_v_angle, 0.0)
            self._max_h_angle = min(self._max_h_angle, 0.0)
            self._min_h_angle = max(self._min_h_angle, -np.pi)

        self._max_roll = self._roll_delta
        self._min_roll = -self._roll_delta
        self._min_radius = max(start_r - start_r * self._max_zoom_in_percent, 0.0)
        self._max_radius = start_r + start_r * self._max_zoom_out_percent

        self._h_angle = self._random_state.uniform(self._min_h_angle, self._max_h_angle)
        self._v_angle = self._random_state.uniform(self._min_v_angle, self._max_v_angle)
        self._radius = self._random_state.uniform(self._min_radius, self._max_radius)
        self._roll = self._random_state.uniform(self._min_roll, self._max_roll)

        vel = self._random_state.randn(3)
        vel /= np.sqrt(np.sum(vel ** 2.0))
        vel *= self._random_state.uniform(0.0, self._max_vel)
        self._camera_vel = vel
        self._roll_vel = self._random_state.uniform(-self._max_roll_vel, self._max_roll_vel)

    def reset(self):
        time_step = self._env.reset()
        self.setup_camera()
        self._apply()
        return time_step

    def step(self, action):
        time_step = self._env.step(action)
        if time_step.first():
            self.setup_camera()
        self._apply()
        return time_step

    def _apply(self):
        if not self._camera_type:
            self.setup_camera()

        vel_delta = self._random_state.randn(3)
        self._camera_vel += vel_delta * self._vel_std * self._vel_scaling
        self._roll_vel += self._random_state.randn() * self._roll_vel_std

        vel_norm = np.sqrt(np.sum(self._camera_vel ** 2.0))
        if vel_norm > self._max_vel * self._vel_scaling:
            self._camera_vel *= (self._max_vel * self._vel_scaling) / vel_norm
        self._roll_vel = np.clip(self._roll_vel, -self._max_roll_vel, self._max_roll_vel)

        cart_cam_pos = sphere2cart([self._radius, self._h_angle, self._v_angle])
        sphere_cam_pos2 = cart2sphere(cart_cam_pos + self._camera_vel)
        sphere_cam_pos2 = clip_cam_position(
            sphere_cam_pos2,
            self._min_radius, self._max_radius,
            self._min_h_angle, self._max_h_angle,
            self._min_v_angle, self._max_v_angle,
        )
        self._camera_vel = sphere2cart(sphere_cam_pos2) - cart_cam_pos
        self._radius, self._h_angle, self._v_angle = sphere_cam_pos2

        roll2 = np.clip(self._roll + self._roll_vel, self._min_roll, self._max_roll)
        self._roll_vel = roll2 - self._roll
        self._roll = roll2

        cart_cam_pos = sphere2cart(sphere_cam_pos2)
        lookat_method = get_lookat_xmat_no_roll if self._limit_to_upper_quadrant else get_lookat_xmat

        if self._camera_type == 'fixed':
            lookat_mat = lookat_method(self._cam_initial_lookat_point, cart_cam_pos)
        else:
            cart_cam_pos += self._env.physics.named.data.subtree_com[1]
            lookat_mat = lookat_method(
                get_lookat_point(self._env.physics, self._camera_id), cart_cam_pos
            )

        lookat_mat = np.reshape(lookat_mat, (3, 3))
        xmat = np.dot(lookat_mat, rotvec2mat(self._roll, np.array([0.0, 0.0, 1.0])))
        self._env.physics.named.data.cam_xpos[self._camera_id] = cart_cam_pos
        self._env.physics.named.data.cam_xmat[self._camera_id] = xmat.flatten()

    def __getattr__(self, attr):
        if hasattr(self._env, attr):
            return getattr(self._env, attr)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{attr}'"
        )
