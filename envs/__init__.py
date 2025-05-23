import gym

# ANCHOR: PickupDropoff
gym.envs.register(
    id='PickupDropoff-v0',
    entry_point='envs.grid_envs:PickupDropoff',
    max_episode_steps=400,
)

# ANCHOR: Office environments
gym.envs.register(
    id='Office-v0',
    entry_point='envs.grid_envs:Office',
    max_episode_steps=200,
)

# ----------------------------------------- WATER
for i in range(11):
    w_id = 'Water-M%d-v0'%i
    w_en = 'envs.water.water_environment:WaterRMEnvM%d'%i
    gym.envs.register(
        id=w_id,
        entry_point=w_en,
        max_episode_steps=600
    )

for i in range(11):
    w_id = 'Water-single-M%d-v0'%i
    w_en = 'envs.water.water_environment:WaterRM10EnvM%d'%i
    gym.envs.register(
        id=w_id,
        entry_point=w_en,
        max_episode_steps=600
    )

gym.envs.register(
    id='OfficeEval-v0',
    entry_point='envs.grid_envs:Office',
    kwargs={'add_obj_to_start': False},
    max_episode_steps=200,
)


gym.envs.register(
    id='Office-v1',
    entry_point='envs.grid_envs:Office',
    max_episode_steps=200,
    kwargs={'random_act_prob': 0.3}
)

gym.envs.register(
    id="OfficeAreas-v0",
    entry_point="envs.grid_envs:OfficeAreas",
    max_episode_steps=200,
)


gym.envs.register(
    id="OfficeAreasEval-v0",
    entry_point="envs.grid_envs:OfficeAreas",
    kwargs={'add_obj_to_start': False, 'add_empty_to_start': False},
    max_episode_steps=200,
)

gym.envs.register(
    id="OfficeAreasRBF-v0",
    entry_point="envs.grid_envs:OfficeAreasRBF",
    max_episode_steps=200,
)

gym.envs.register(
    id="OfficeAreasRBFEval-v0",
    entry_point="envs.grid_envs:OfficeAreasRBF",
    kwargs={'add_obj_to_start': False, 'add_empty_to_start': False},
    max_episode_steps=200,
)

gym.envs.register(
    id="OfficeAreasRBFOnly-v0",
    entry_point="envs.grid_envs:OfficeAreasRBF",
    kwargs={'only_rbf': True},
    max_episode_steps=200,
)

gym.envs.register(
    id="OfficeAreasRBFOnlyEval-v0",
    entry_point="envs.grid_envs:OfficeAreasRBF",
    kwargs={'only_rbf': True, 'add_obj_to_start': False, 'add_empty_to_start': False},
    max_episode_steps=200,
)

gym.envs.register(
    id="OfficeAreasFeatures-v0",
    entry_point="envs.grid_envs:OfficeAreasFeatures",
    max_episode_steps=200,
)

gym.envs.register(
    id="OfficeAreasFeaturesEval-v0",
    entry_point="envs.grid_envs:OfficeAreasFeatures",
    kwargs={'add_obj_to_start': False, 'add_empty_to_start': False},
    max_episode_steps=200,
)

gym.envs.register(
    id="OfficeAreasFeaturesDiscrete-v0",
    entry_point="envs.grid_envs:OfficeAreasFeaturesDiscrete",
    max_episode_steps=200,
)

gym.envs.register(
    id="OfficeAreasFeaturesDiscreteEval-v0",
    entry_point="envs.grid_envs:OfficeAreasFeaturesDiscrete",
    kwargs={'add_obj_to_start': False, 'add_empty_to_start': False},
    max_episode_steps=200,
)

gym.envs.register(
    id="OfficeAreasFeaturesContinuous-v0",
    entry_point="envs.grid_envs:OfficeAreasFeaturesContinuous",
    max_episode_steps=200,
)

gym.envs.register(
    id="OfficeAreasFeaturesContinuousEval-v0",
    entry_point="envs.grid_envs:OfficeAreasFeaturesContinuous",
    kwargs={'add_obj_to_start': False, 'add_empty_to_start': False},
    max_episode_steps=200,
)

# ANCHOR: Delivery environments

gym.envs.register(
    id='Delivery-v0',
    entry_point='envs.grid_envs:Delivery',
    max_episode_steps=200,
)

gym.envs.register(
    id='DeliveryEval-v0',
    entry_point='envs.grid_envs:Delivery',
    kwargs={'add_obj_to_start': False},
    max_episode_steps=200,
)

# ANCHOR: DoubleSlit environments
gym.envs.register(
    id='DoubleSlit-v0',
    entry_point='envs.grid_envs:DoubleSlit',
    max_episode_steps=1000,
)

gym.envs.register(
    id='DoubleSlitRS-v0',
    entry_point='envs.grid_envs:DoubleSlitRS',
    max_episode_steps=1000,
)

gym.envs.register(
    id='DoubleSlitRSEval-v0',
    entry_point='envs.grid_envs:DoubleSlitRS',
    max_episode_steps=1000,
)

gym.envs.register(
    id='DoubleSlit-v1',
    entry_point='envs.grid_envs:DoubleSlit',
    max_episode_steps=1000,
    kwargs={'max_wind': 3},
)

gym.envs.register(
    id='DoubleSlitEval-v0',
    entry_point='envs.grid_envs:DoubleSlit',
    max_episode_steps=1000,
    kwargs={
             'max_wind': 1},
)

gym.envs.register(
    id='DoubleSlitEval-v1',
    entry_point='envs.grid_envs:DoubleSlit',
    max_episode_steps=1000,
    kwargs={
             'max_wind': 3},
)

# ANCHOR: IceCorridor environments
gym.envs.register(
    id='IceCorridor-v0',
    entry_point='envs.grid_envs:IceCorridor',
    max_episode_steps=10000,
)

gym.envs.register(
    id='IceCorridorEval-v0',
    entry_point='envs.grid_envs:IceCorridor',
    max_episode_steps=10000,
)

