from skrl.memories.torch import RandomMemory
import torch
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
import statistics
from tqdm import tqdm

from policy import Policy
from value import Value

class BinPackingAgent:

    def __init__(self, env, device, config, name, directory):
        self.env = env
        self.device = device
        self.memory = RandomMemory(memory_size=1024, num_envs=env.num_envs, device=self.device)
        self.space_utilizations = []
        
        cfg = PPO_DEFAULT_CONFIG.copy()
        cfg["rollouts"] = 1024
        cfg["learning_epochs"] = config.learning_epochs
        cfg["mini_batches"] = config.mini_batches
        cfg["discount_factor"] = config.discount_factor
        cfg["entropy_loss_scale"] = 0.02
        cfg["learning_rate_scheduler"] = KLAdaptiveRL
        cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
        #cfg["value_loss_scale"] = 0.5
        # cfg["state_preprocessor"] = RunningStandardScaler
        # cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
        # cfg["value_preprocessor"] = RunningStandardScaler
        # cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
        # logging to TensorBoard and write checkpoints (in timesteps)
        cfg["experiment"]["write_interval"] = 1000
        cfg["experiment"]["checkpoint_interval"] = 50000
        cfg["experiment"]["directory"] = "runs/" + directory
        cfg["experiment"]["experiment_name"] = name
        cfg["experiment"]["wandb"] = False

        self.actor = Policy(observation_space=self.env.observation_space,
             action_space=self.env.action_space,
             device=self.env.device,
             unnormalized_log_prob=True)

        critic = Value(observation_space=self.env.observation_space,
                 action_space=self.env.action_space,
                 device=self.env.device,
                 clip_actions=False)

        models = {}
        models["policy"] = self.actor
        models["value"] = critic
        
        self.agent = PPO(models=models,
                    memory=self.memory,
                    cfg=cfg,
                    observation_space=self.env.observation_space,
                    action_space=self.env.action_space,
                    device=self.device)

    def train(self, timesteps, headless = True):
        cfg_trainer = {"timesteps": timesteps, "headless": headless}
        trainer = SequentialTrainer(cfg=cfg_trainer, env=self.env, agents=[self.agent])
        trainer.train()

    def load_agent(self, path):
        self.agent.load(path)
        self.agent._rnn = False

    def get_action(self, observation):
        action = torch.vstack([self.agent.act(observation, timestep=0, timesteps=10)[0]])
        return action

    def validate(self, rollouts_no = 1000, single_rollout_max_length = 240):
        space_utilized = []
        illegal_actions = []
        
        for rollout in tqdm (range(rollouts_no), desc="Loading..."):
            observation, info = self.env.reset()
            for t in range(single_rollout_max_length):
               #agent.pre_interaction(timestep=t, timesteps=10)
               action = torch.vstack([self.agent.act(observation[0:2], timestep=t, timesteps=10)[0]])
               observation, reward, terminated, truncated, info = self.env.step(action)
               if terminated:
                   break
            space_utilized.append(self.env.get_volume_used())
            illegal_actions.append(self.env.illegal_actions)
        print("The sapce utilzied is: {}%.".format(statistics.mean(space_utilized)))
        print("Number of illegal actions is: {}.".format(statistics.mean(illegal_actions)))
        self.space_utilizations = space_utilized
