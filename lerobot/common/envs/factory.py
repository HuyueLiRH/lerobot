from torchrl.envs.transforms import StepCounter, TransformedEnv


def make_env(cfg, transform=None):
    kwargs = {
        "frame_skip": cfg.env.action_repeat,
        "from_pixels": cfg.env.from_pixels,
        "pixels_only": cfg.env.pixels_only,
        "image_size": cfg.env.image_size,
    }

    if cfg.env.name == "simxarm":
        from lerobot.common.envs.simxarm import SimxarmEnv

        kwargs["task"] = cfg.env.task
        clsfunc = SimxarmEnv
    elif cfg.env.name == "pusht":
        from lerobot.common.envs.pusht import PushtEnv

        clsfunc = PushtEnv
    else:
        raise ValueError(cfg.env.name)

    env = clsfunc(**kwargs)

    # limit rollout to max_steps
    env = TransformedEnv(env, StepCounter(max_steps=cfg.env.episode_length))

    if transform is not None:
        # useful to add normalization
        env.append_transform(transform)

    return env


# def make_env(env_name, frame_skip, device, is_test=False):
#     env = GymEnv(
#         env_name,
#         frame_skip=frame_skip,
#         from_pixels=True,
#         pixels_only=False,
#         device=device,
#     )
#     env = TransformedEnv(env)
#     env.append_transform(NoopResetEnv(noops=30, random=True))
#     if not is_test:
#         env.append_transform(EndOfLifeTransform())
#         env.append_transform(RewardClipping(-1, 1))
#     env.append_transform(ToTensorImage())
#     env.append_transform(GrayScale())
#     env.append_transform(Resize(84, 84))
#     env.append_transform(CatFrames(N=4, dim=-3))
#     env.append_transform(RewardSum())
#     env.append_transform(StepCounter(max_steps=4500))
#     env.append_transform(DoubleToFloat())
#     env.append_transform(VecNorm(in_keys=["pixels"]))
#     return env
