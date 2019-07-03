import numpy as np
from baselines.meta_a2c.utils import discount_with_dones
from baselines.common.runners import AbstractEnvRunner
from baselines import logger

class Runner(AbstractEnvRunner):
    """
    We use this class to generate batches of experiences

    __init__:
    - Initialize the runner

    run():
    - Make a mini batch of experiences
    """
    def __init__(self, env, model, nsteps=5, gamma=0.99):
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.gamma = gamma
        self.batch_action_shape = [x if x is not None else -1 for x in model.train_model.action.shape.as_list()]
        self.ob_dtype = model.train_model.X.dtype.as_numpy_dtype

    def run(self):
        # We initialize the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [],[],[],[],[]
        mb_timesteps = []
        mb_infodicts = []
        mb_states = self.states
        for n in range(self.nsteps):
            # Given observations, take action and value (V(s))
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            
            actions, values, states, _ = self.model.step(
                self.obs,
                self.p_actions,
                self.p_rewards,
                self.timesteps,
                S=self.states,
                M=self.dones
            )

            # Append the experiences
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)

            # Take actions in env and look the results
            obs, rewards, dones, info_dicts = self.env.step(actions)

            mb_infodicts.extend(list(info_dicts))
            self.states = states
            self.dones = dones
            self.obs = obs
            self.p_actions = actions
            self.p_rewards = rewards
            self.timesteps = self.timesteps + 1
            mb_timesteps.append(self.timesteps)
            mb_rewards.append(rewards)

        mb_dones.append(self.dones)

        # Batch of steps to batch of rollouts
        mb_timesteps = np.asarray(mb_timesteps, dtype=np.int32).swapaxes(1, 0)

        mb_obs = np.asarray(mb_obs, dtype=self.ob_dtype).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=self.model.train_model.action.dtype.name).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]

        if self.gamma > 0.0:
            # Discount/bootstrap off value fn
            last_values = self.model.value(self.obs, self.p_actions, self.p_rewards, self.timesteps, S=self.states, M=self.dones).tolist()
            for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
                rewards = rewards.tolist()
                dones = dones.tolist()
                if dones[-1] == 0:
                    rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
                else:
                    rewards = discount_with_dones(rewards, dones, self.gamma)

                mb_rewards[n] = rewards

        mb_actions = mb_actions.reshape(self.batch_action_shape)

        mb_rewards = mb_rewards.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()

        mb_timesteps = mb_timesteps.flatten().reshape(self.batch_action_shape + [1])

        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values, mb_timesteps, mb_infodicts
