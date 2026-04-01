from __future__ import annotations

from stable_baselines3 import PPO


def create_ppo_agent(env, **kwargs) -> PPO:
    params = dict(
        policy="MlpPolicy",
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        device="cpu",
    )
    params.update(kwargs)
    return PPO(env=env, **params)
