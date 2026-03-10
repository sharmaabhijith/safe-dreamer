from . import parallel, wrappers


def make_envs(config):
    def env_constructor(idx):
        return lambda: make_env(config, idx)

    train_envs = parallel.SerialEnv(env_constructor, config.env_num, config.device)
    eval_envs = parallel.SerialEnv(env_constructor, config.eval_episode_num, config.device)
    obs_space = train_envs.observation_space
    act_space = train_envs.action_space
    return train_envs, eval_envs, obs_space, act_space


def make_eval_envs(config, num_envs=None):
    """Create evaluation-only environments, optionally with video background."""
    n = num_envs or config.eval_episode_num
    def env_constructor(idx):
        return lambda: make_env(config, idx)
    envs = parallel.SerialEnv(env_constructor, n, config.device)
    obs_space = envs.observation_space
    act_space = envs.action_space
    return envs, obs_space, act_space


def make_env(config, id):
    suite, task = config.task.split("_", 1)
    if suite == "dmc":
        import envs.dmc as dmc

        env = dmc.DeepMindControl(task, config.action_repeat, config.size, seed=config.seed + id)
        env = wrappers.NormalizeActions(env)
        video_dir = getattr(config, "video_dir", None)
        if video_dir:
            from envs.video_background import VideoBackground
            env = VideoBackground(env, video_dir=video_dir, size=config.size, seed=config.seed + id)
    elif suite == "distract":
        from envs.dmc_distracting import DistractingControl

        env = DistractingControl(
            task,
            difficulty=getattr(config, "difficulty", "easy"),
            background_dataset_path=getattr(config, "background_dataset_path", None),
            background_dataset_videos=getattr(config, "background_dataset_videos", "train"),
            dynamic=getattr(config, "dynamic", True),
            background=getattr(config, "background", True),
            camera=getattr(config, "camera", True),
            color=getattr(config, "color", True),
            action_repeat=config.action_repeat,
            size=tuple(config.size),
            seed=config.seed + id,
        )
        env = wrappers.NormalizeActions(env)
    elif suite == "metaworld":
        import envs.metaworld as metaworld

        env = metaworld.MetaWorld(
            task,
            config.action_repeat,
            config.size,
            config.camera,
            config.seed + id,
        )
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit // config.action_repeat)
    return wrappers.Dtype(env)
