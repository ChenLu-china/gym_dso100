from gymnasium.envs.registration import register

register(
    id="gym_dso100/Dso100-v0",
    entry_point="gym_dso100.tasks:Lift",
    max_episode_steps=300,
    kwargs={"obs_type": "state"},
)