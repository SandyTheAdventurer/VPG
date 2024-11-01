from gymnasium import Env
from main import Base, printb
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter
import torch
import numpy as np
import torch.nn as nn

torch.set_default_device('cuda')
torch.set_default_dtype(torch.float64)

class VPG(Base):
    def __init__(self, env: Env, logdir: str, gamma=1) -> None:
        self.env = env
        self.logdir = logdir
        self.actionlen = self.env.action_space.n
        self.writer = SummaryWriter(self.logdir)
        self.gamma = gamma
        self.buffer = []
        super().__init__(input_size=self.env.observation_space.shape[0], output_size=self.actionlen, probs=True)

    def compute_returns(self, rewards):
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return torch.tensor(returns, dtype=torch.float64)

    def act(self, obs : torch.Tensor):
        obs = obs.to(torch.float64)
        dist = self(obs)
        select = torch.distributions.Categorical(dist)
        action = select.sample()
        logit = select.log_prob(action)
        return action, logit
    
    def infer(self):
        obs, logit, reward = zip(*self.buffer)
        reward = self.compute_returns(reward)
        obs = torch.stack(obs)

        reward = (reward - reward.mean()) / (reward.std() + 1e-8)

        logit = torch.stack(logit)
        loss = -(logit.view(-1) * reward).sum()
        self.step_optimizer(loss)

        self.buffer.clear()
        return loss.item()
    
    def learn(self, timesteps: int):
        self.epsilon = 1
        rew_list = []
        loss_list = []
        steps_list = []

        for i in tqdm(range(timesteps)):
            obs, _ = self.env.reset()
            obs = torch.tensor(obs, dtype=torch.float64)
            terminated = False
            total_reward = 0
            steps = 0

            while not terminated:
                prev_obs = obs.clone()
                action, logit = self.act(obs)
                obs, rew, terminated, truncated, _ = self.env.step(action.item())
                obs=torch.tensor(obs, dtype=torch.float64)
                total_reward += rew
                self.buffer.append([prev_obs, logit, rew])                
                steps += 1
                if truncated:
                    break
            
            loss = self.infer()
            loss_list.append(loss)

            rew_list.append(total_reward)
            steps_list.append(steps)

            self.writer.add_scalar("Reward/avg", np.mean(rew_list), i)
            self.writer.add_scalar("Reward/max", np.max(rew_list), i)
            self.writer.add_scalar("Reward/min", np.min(rew_list), i)
            self.writer.add_scalar("Loss/avg", np.mean(loss_list), i)
            self.writer.add_scalar("Loss/max", np.max(loss_list), i)
            self.writer.add_scalar("Loss/min", np.min(loss_list), i)
            self.writer.add_scalar("Steps/avg", np.mean(steps_list), i)
            self.writer.add_scalar("Steps/max", np.max(steps_list), i)
            self.writer.add_scalar("Steps/min", np.min(steps_list), i)
            self.writer.add_scalar("Epsilon", self.epsilon, i)

            if i % 1000 == 0 and i > 0:
                printb(f"Avg Reward for {i}th iteration: {np.mean(rew_list):.2f}",
                       f"Max Reward for {i}th iteration: {np.max(rew_list):.2f}",
                       f"Min Reward for {i}th iteration: {np.min(rew_list):.2f}",
                       f"Avg Loss for {i}th iteration: {np.mean(loss_list):.4f}",
                       f"Max Loss for {i}th iteration: {np.max(loss_list):.4f}",
                       f"Min Loss for {i}th iteration: {np.min(loss_list):.4f}",
                       f"Avg Steps for {i}th iteration: {np.mean(steps_list):.2f}",
                       f"Max Steps for {i}th iteration: {np.max(steps_list):.2f}",
                       f"Min Steps for {i}th iteration: {np.min(steps_list):.2f}",
                       f"Epsilon: {self.epsilon:.4f}")
                rew_list.clear()
                loss_list.clear()
                steps_list.clear()
        self.writer.close()