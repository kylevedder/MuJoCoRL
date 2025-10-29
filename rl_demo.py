# file: reinforce_mujoco_min.py
import numpy as np, torch, torch.nn as nn, torch.optim as optim
import gymnasium as gym

ENV_ID = "Hopper-v4"  # use v2/v3/v4 depending on your Gymnasium version
SEED = 1
HORIZON = 2048
GAMMA = 0.99
LR = 3e-4
EPOCHS = 20

torch.manual_seed(SEED)
np.random.seed(SEED)

env = gym.make(ENV_ID)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
act_high = torch.as_tensor(env.action_space.high)
act_low = torch.as_tensor(env.action_space.low)


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh()
        )
        self.mu = nn.Linear(64, act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))  # exp(log_std) ~ 0.6

    def forward(self, x):
        z = self.net(x)
        mu = self.mu(z)
        std = self.log_std.exp()
        return mu, std

    def sample(self, obs):
        mu, std = self(obs)
        dist = torch.distributions.Normal(mu, std)
        a = dist.rsample()
        logp = dist.log_prob(a).sum(-1)
        a = torch.tanh(a)  # squash to (-1,1)
        # scale to env bounds
        act = (act_high - act_low) * (a + 1) / 2 + act_low
        return act, logp


pi = Policy()
opt = optim.Adam(pi.parameters(), lr=LR)


def rollout():
    obs, _ = env.reset(seed=SEED)
    buf_obs, buf_act, buf_logp, buf_ret = [], [], [], []
    ep_rets = []
    rew_hist = []
    for t in range(HORIZON):
        o = torch.as_tensor(obs, dtype=torch.float32)
        act, logp = pi.sample(o)
        act_np = act.detach().numpy()
        obs, r, term, trunc, _ = env.step(act_np)
        buf_obs.append(o)
        buf_act.append(act)
        buf_logp.append(logp)
        rew_hist.append(r)
        done = term or trunc
        if done or t == HORIZON - 1:
            # compute returns for this segment
            G = 0.0
            for rr in reversed(rew_hist):
                G = rr + GAMMA * G
                buf_ret.insert(0, G)
            ep_rets.append(sum(rew_hist))
            rew_hist.clear()
            if done:
                obs, _ = env.reset()
    return (
        torch.stack(buf_obs),
        torch.stack(buf_act),
        torch.stack(buf_logp),
        torch.as_tensor(buf_ret, dtype=torch.float32),
        np.mean(ep_rets),
        np.max(ep_rets),
    )


for epoch in range(1, EPOCHS + 1):
    obs_t, act_t, logp_t, ret_t, avg_ret, max_ret = rollout()
    # normalize returns
    ret_n = (ret_t - ret_t.mean()) / (ret_t.std() + 1e-8)
    # policy gradient loss: -E[logpi * return]
    loss = -(logp_t * ret_n).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(
        f"epoch {epoch:02d} | loss {loss.item():.3f} | avg_ret {avg_ret:.1f} | max_ret {max_ret:.1f}"
    )

env.close()
