import os
from pathlib import Path
import gymnasium as gym
import matplotlib.pyplot as plt

print("MUJOCO_GL:", os.environ.get("MUJOCO_GL"))
env = gym.make("FetchPickAndPlace-v3", render_mode="rgb_array")
env.reset()
frame = env.render()
print("frame:", frame.shape)

# Save frame to file
plt.imsave(str(Path("frame.png")), frame)
env.close()
