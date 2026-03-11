"""Training entry point for all experiment variants.

Select the experiment config via Hydra's --config-name flag:

    # DMC standard + CNN encoder
    python train.py --config-name dmc/cnn env.task=dmc_walker_walk

    # DMC standard + multimodal FiLM encoder
    python train.py --config-name dmc/multimodal env.task=dmc_walker_walk

    # DMC distracting + CNN encoder
    python train.py --config-name dmc/distractor_cnn env.task=distract_walker_walk

    # DMC distracting + multimodal FiLM encoder
    python train.py --config-name dmc/distractor_multimodal env.task=distract_walker_walk

    # Override any parameter via CLI
    python train.py --config-name dmc/cnn model.lr=1e-4 model.imag_horizon=20
"""

import atexit
import pathlib
import os
import warnings

import hydra
import torch

from utils import tools
from utils.buffer import Buffer
from world_model.dreamer import Dreamer
from envs import make_envs
from utils.trainer import OnlineTrainer

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")


def _setup_gpu(config):
    """Ensure MuJoCo uses EGL on the same GPU as PyTorch.

    When running under SLURM, CUDA_VISIBLE_DEVICES is already set by the
    scheduler — do NOT overwrite it.  We only need to point MuJoCo's EGL
    renderer at device 0 (the first *visible* GPU).
    """
    from omegaconf import OmegaConf

    device = config.device
    if device.startswith("cuda"):
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
        OmegaConf.update(config, "device", "cuda:0")


@hydra.main(version_base=None, config_path="configs", config_name="dmc/cnn")
def main(config):
    from omegaconf import OmegaConf
    OmegaConf.set_struct(config, False)
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
    main()
