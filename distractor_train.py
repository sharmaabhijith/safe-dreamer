"""Training entry point for the Distracting Control Suite.

Drop-in replacement for train.py with the distracting env config as default.

Example usage:
    # Easy background + camera + color distractors, walker_walk:
    python distractor_train.py \
        env.background_dataset_path=/data/DAVIS/JPEGImages/480p \
        env.task=distract_walker_walk \
        env.difficulty=easy

    # Hard distractors, cheetah_run:
    python distractor_train.py \
        env.background_dataset_path=/data/DAVIS/JPEGImages/480p \
        env.task=distract_cheetah_run \
        env.difficulty=hard

    # Disable camera distractor, keep background + color:
    python distractor_train.py \
        env.background_dataset_path=/data/DAVIS/JPEGImages/480p \
        env.camera=false

    # Override model size or rep loss as usual:
    python distractor_train.py \
        env.background_dataset_path=/data/DAVIS/JPEGImages/480p \
        model=size25M \
        model.rep_loss=dreamer

Install distracting_control:
    pip install git+https://github.com/google-research/distracting_control.git

Download DAVIS 2017 480p TrainVal:
    wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
    unzip DAVIS-2017-trainval-480p.zip
    # Then point background_dataset_path to DAVIS/JPEGImages/480p
"""

import atexit
import pathlib
import sys
import os
import warnings

import hydra
import torch

import tools
from buffer import Buffer
from dreamer import Dreamer
from envs import make_envs
from trainer import OnlineTrainer

warnings.filterwarnings("ignore")
sys.path.append(str(pathlib.Path(__file__).parent))
torch.set_float32_matmul_precision("high")


def _setup_gpu(config):
    """Restrict CUDA and MuJoCo EGL to the same physical GPU."""
    from omegaconf import OmegaConf

    device = config.device
    if device.startswith("cuda"):
        gpu_id = device.split(":")[-1] if ":" in device else "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ["MUJOCO_EGL_DEVICE_ID"] = gpu_id
        OmegaConf.update(config, "device", "cuda:0")


@hydra.main(version_base=None, config_path="configs", config_name="configs")
def main(config):
    _setup_gpu(config)
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)

    console_f = tools.setup_console_log(logdir, filename="console.log")
    atexit.register(lambda: console_f.close())

    print("Logdir", logdir)

    logger = tools.Logger(logdir)
    logger.log_hydra_config(config)

    replay_buffer = Buffer(config.buffer)

    print("Create envs.")
    train_envs, eval_envs, obs_space, act_space = make_envs(config.env)

    print("Simulate agent.")
    agent = Dreamer(
        config.model,
        obs_space,
        act_space,
    ).to(config.device)

    if config.model.use_multimodal_encoder:
        task_name = config.env.task
        # Strip suite prefix (distract_ or dmc_) to get the bare task name
        for prefix in ("distract_", "dmc_"):
            if task_name.startswith(prefix):
                task_name = task_name[len(prefix):]
                break
        agent.set_task_name(task_name)
        print(f"Task: {task_name} (text descriptions loaded for random sampling)")

    policy_trainer = OnlineTrainer(
        config.trainer, replay_buffer, logger, logdir, train_envs, eval_envs
    )
    policy_trainer.begin(agent)

    items_to_save = {
        "agent_state_dict": agent.state_dict(),
        "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
    }
    torch.save(items_to_save, logdir / "latest.pt")


if __name__ == "__main__":
    # Inject the distracting env as the default so users don't have to type it,
    # while still allowing any override via Hydra CLI syntax.
    # If the user already passed an env= override this will be ignored by Hydra.
    if not any(arg.startswith("env=") for arg in sys.argv[1:]):
        sys.argv.insert(1, "env=dmc_distracting")

    main()
