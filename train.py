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
# torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

def _setup_gpu(config):
    """Restrict CUDA and MuJoCo EGL to the same physical GPU.

    Parses the GPU index from config.device (e.g. 'cuda:1'), sets
    CUDA_VISIBLE_DEVICES to that physical GPU, points MuJoCo EGL at
    device 0 (the only visible one), and remaps config.device to 'cuda:0'
    so all downstream code sees a consistent single-GPU view.
    """
    from omegaconf import OmegaConf

    device = config.device
    if device.startswith("cuda"):
        gpu_id = device.split(":")[-1] if ":" in device else "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        os.environ.setdefault("MUJOCO_GL", "egl")
        # EGL device IDs are physical — not affected by CUDA_VISIBLE_DEVICES.
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

    # Mirror stdout/stderr to a file under logdir while keeping console output.
    console_f = tools.setup_console_log(logdir, filename="console.log")
    atexit.register(lambda: console_f.close())

    print("Logdir", logdir)

    logger = tools.Logger(logdir)
    # save config
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

    # Set task name for multimodal encoder (loads text pool for random sampling)
    if config.model.use_multimodal_encoder:
        task_name = config.env.task
        if task_name.startswith("dmc_"):
            task_name = task_name[4:]
        agent.set_task_name(task_name)
        print(f"Task: {task_name} (100 text descriptions loaded for random sampling)")

    policy_trainer = OnlineTrainer(config.trainer, replay_buffer, logger, logdir, train_envs, eval_envs)
    policy_trainer.begin(agent)

    items_to_save = {
        "agent_state_dict": agent.state_dict(),
        "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
    }
    torch.save(items_to_save, logdir / "latest.pt")


if __name__ == "__main__":
    main()
