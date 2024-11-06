import torch
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
from gymnasium import Env
from tqdm import tqdm
from main import Base, printb

torch.set_default_device('cuda')
torch.set_default_dtype(torch.float32)

class VPG(Base):
    def __init__(self, env: Env, logdir: str, gamma=1, batch_size=32, accumulation_steps=8) -> None:
        super().__init__(input_size=env.observation_space.shape[0], output_size=env.action_space.n, probs=True)
        self.env = env
        self.logdir = logdir
        self.batch_size = batch_size
        self.gamma = gamma
        self.writer = SummaryWriter(logdir)
        self.accumulation_steps = accumulation_steps
        self.accumulation_counter = 0
        self.buffer = []

    def compute_returns(self, rewards):
        returns = torch.zeros(len(rewards), dtype=torch.float32, device='cuda')
        G = 0
        for i in reversed(range(len(rewards))):
            G = rewards[i] + self.gamma * G
            returns[i] = G
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def step_optimizer(self, loss, accumulate=False):
        loss.backward()
        if accumulate:
            self.accumulation_counter += 1
            if self.accumulation_counter % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.accumulation_counter = 0
        else:
            self.optimizer.step()
            self.optimizer.zero_grad()

    def act(self, obs: torch.Tensor):
        dist = self(torch.tensor(obs, dtype=torch.float32))
        select = torch.distributions.Categorical(dist)
        action = select.sample()
        logit = select.log_prob(action)
        return action.item(), logit

    def infer(self):
        obs, logit, reward = zip(*self.buffer)
        reward = self.compute_returns(reward)
        obs = torch.stack(obs).to('cuda')
        logit = torch.stack(logit).to('cuda')

        loss = -(logit.view(-1) * reward).sum()
        self.step_optimizer(loss, accumulate=True)
        self.buffer.clear()

    def learn(self, timesteps: int):
        self.epsilon = 1
        rew_list, steps_list = [], []

        for i in tqdm(range(1, timesteps + 1)):
            obs, _ = self.env.reset()
            obs = torch.tensor(obs, dtype=torch.float32, device='cuda')
            terminated = False
            total_reward, steps = 0, 0

            while not terminated:
                action, logit = self.act(obs)
                obs, rew, terminated, truncated, _ = self.env.step(action)
                obs = torch.tensor(obs, dtype=torch.float32, device='cuda')
                total_reward += rew
                self.buffer.append((obs, logit, rew))
                steps += 1
                if truncated:
                    break

            rew_list.append(total_reward)
            steps_list.append(steps)

            if i % 32 == 0:
                self.infer()
                self.writer.add_scalar("Reward/avg", np.mean(rew_list), i)
                self.writer.add_scalar("Reward/max", np.max(rew_list), i)
                self.writer.add_scalar("Reward/min", np.min(rew_list), i)
                self.writer.add_scalar("Steps/avg", np.mean(steps_list), i)
                self.writer.add_scalar("Steps/max", np.max(steps_list), i)
                self.writer.add_scalar("Steps/min", np.min(steps_list), i)
                self.writer.add_scalar("Epsilon", self.epsilon, i)

                rew_list.clear()
                steps_list.clear()
        
        self.writer.close()
