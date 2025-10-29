import os
from pathlib import Path
import gymnasium as gym
import gymnasium_robotics as gym_robotics
import matplotlib.pyplot as plt

print("MUJOCO_GL:", os.environ.get("MUJOCO_GL"))
env = gym.make("FetchPickAndPlace-v4", render_mode="rgb_array", width=640, height=480)
env.reset()
frame = env.render()
print("frame:", frame.shape)

# Save frame to file
plt.imsave(str(Path("frame.png")), frame)
env.close()
