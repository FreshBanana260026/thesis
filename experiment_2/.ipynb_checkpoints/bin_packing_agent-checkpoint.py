from skrl.memories.torch import RandomMemory
import torch
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.resources.preprocessors.torch import RunningStandardScaler
import statistics
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import norm

from policy import Policy
from value import Value
import numpy as np

class BinPackingAgent:

    def __init__(self, env, device, config, name, directory):
        self.env = env
        self.device = device
        self.memory = RandomMemory(memory_size=1024, num_envs=env.num_envs, device=self.device)
        self.space_utilizations = []
        self.area_utilizations = []
        
        cfg = PPO_DEFAULT_CONFIG.copy()
        cfg["rollouts"] = 1024
        cfg["learning_epochs"] = config.learning_epochs
        cfg["mini_batches"] = config.mini_batches
        cfg["discount_factor"] = config.discount_factor
        cfg["entropy_loss_scale"] = 0.02
        cfg["learning_rate_scheduler"] = KLAdaptiveRL
        cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
        cfg["state_preprocessor"] = RunningStandardScaler
        cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
        cfg["value_preprocessor"] = RunningStandardScaler
        cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
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

    #run validation and collect data for a confidence interval
    def validate(self, rollouts_no = 1000, single_rollout_max_length = 120):
        space_utilized = []
        illegal_actions = []
        episode_length = []
        
        for rollout in tqdm (range(rollouts_no), desc="Loading..."):
            observation, info = self.env.reset()
            counter = 0
            for t in range(single_rollout_max_length):
               counter = counter + 1
               #agent.pre_interaction(timestep=t, timesteps=10)
               action = torch.vstack([self.agent.act(observation[0:2], timestep=t, timesteps=10)[0]])
               observation, reward, terminated, truncated, info = self.env.step(action)
               if terminated:
                   break
            space_utilized.append(self.env.get_volume_used())
            episode_length.append(counter)
            illegal_actions.append(self.env.illegal_actions)
        print("Episode Length: {}.".format(statistics.mean(episode_length)))
        print("Mean: {}%.".format(statistics.mean(space_utilized)))
        print("Median: {}%.".format(statistics.median(space_utilized)))
        print("Std: {}%.".format(statistics.stdev(space_utilized)))
        print("Variance: {}%.".format(statistics.variance(space_utilized)))
        
        self.space_utilizations = space_utilized

    #sample and calculate confidence interval
    def get_confidence_interval(self, conf_level = 0.95, n_bootstrap = 80000):
        mean = statistics.mean(self.space_utilizations)
        bootstrap_sampels = []

        for i in range(n_bootstrap):
            sample = np.random.choice(self.space_utilizations, size = len(self.space_utilizations), replace = True)
            bootstrap_sampels.append(sample)
        means = [np.mean(s) for s in bootstrap_sampels]

        num_bins = 200
        hist, bins = np.histogram(self.space_utilizations, bins=num_bins)
        
        # Plot the histogram as a bar plot
        plt.bar(bins[:-1], hist, width=(bins[1] - bins[0]), alpha=0.8, color = "#f700f8", label = "Freqency")

        ### Fit a normal distribution to the data
        mu, std = norm.fit(self.space_utilizations)
        
        # Plot the PDF over the histogram
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 2000)
        p = norm.pdf(x, mu, std) * len(self.space_utilizations) * (bins[1] - bins[0])  # Scaling PDF to match histogram
        plt.plot(x, p, 'k', linewidth=2, label='Normal Distribution', color="#02fdff")

        lower_bound = mu - 1.96 * std
        upper_bound = mu + 1.96 * std
        plt.fill_between(x, p, where=[(lower_bound <= xi <= upper_bound) for xi in x], color="#02fdff", alpha=0.3, label='95% CI')

       # Add vertical dashed lines for lower and upper bounds
        pdf_lower_bound = norm.pdf(lower_bound, mu, std) * len(self.space_utilizations) * (bins[1] - bins[0])
        pdf_upper_bound = norm.pdf(upper_bound, mu, std) * len(self.space_utilizations) * (bins[1] - bins[0])

        # Add vertical dashed lines for lower and upper bounds
        plt.axvline(lower_bound, color='blue', linestyle='--', linewidth=2, label='Lower Bound', ymin=0, ymax=pdf_lower_bound)
        plt.axvline(upper_bound, color='blue', linestyle='--', linewidth=2, label='Upper Bound', ymin=0, ymax=pdf_upper_bound)
                
        # Add labels and title
        plt.xlabel('Space Utilization (%)')
        plt.ylabel('Frequency')
        plt.title('Space Utilization Distribution')
        
        # Show the plot
        plt.legend()
        plt.show()
        print("Lower bound: ", lower_bound)
        print("Upper bound: ", upper_bound)
        print("Span: ", upper_bound - lower_bound)

    def evaluate(self, headless = True):
        cfg_trainer = {"timesteps": 1000, "headless": headless}
        trainer = SequentialTrainer(cfg=cfg_trainer, env=self.env, agents=[self.agent])
        trainer.eval()
