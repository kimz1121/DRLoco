import sys
import os
sys.path.append(os.getcwd())

import random
import numpy as np
from tqdm import tqdm
from drloco.mujoco import ant_env
import gym
import sys
from drloco.common import utils
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecCheckNan, VecNormalize

import argparse
import matplotlib.pyplot as plt


from drloco.common.utils import load_env
from drloco.custom.policies_mcp import MCPNaive, MPPO



parser = argparse.ArgumentParser()
parser.add_argument("--direction", type=int)
parser.add_argument("--id", type=str)
parser.add_argument("--checkpoint", type=str, default="final")

args = parser.parse_args()

direction = args.direction
run_id = args.id
checkpoint = args.checkpoint
save_gif = True

if direction is None:
    direction = random.randint(0, 360)
    eval_env = make_vec_env(ant_env.RandomGoalAnt, n_envs=1, env_kwargs={'direction_range': None, 'direction_list': [direction]})
else:
    eval_env = make_vec_env(ant_env.RandomGoalAnt, n_envs=1, env_kwargs={'direction_range': None, 'direction_list': [direction]})

eval_env = VecNormalize(eval_env)
eval_env = VecCheckNan(eval_env, raise_exception=True)

# eval_env = ant_env.RandomGoalAnt(direction)
# 차후 다시 학습한 것으로 수정하기
model_path = f'logs_0615_transfer_01/NewRandomGoalAnt-v2/{run_id}/seed0/models/NewRandomGoalAnt-v2/{run_id}_{checkpoint}_steps.zip'
# model_path = f'logs_0615_transfer/NewRandomGoalAnt-v2/{run_id}/seed0/models/final.zip'

model = PPO.load(model_path, env=eval_env)
save_path = "evaluation_data/{}_{}".format(run_id, checkpoint)
os.makedirs(save_path, exist_ok=True)

obs = eval_env.reset()
print("----------------------------")
print(obs)
img = eval_env.render("rgb_array")
imgs = [img]
done = False
tot_r = 0.0
weights=[]
print(f"Begin Evaluation")
ep_max_len = 1000
pbar = tqdm(total=ep_max_len)

i = 0
while not done:
    # eval_env.envs[0].set_direction(direction)
    action, _ = model.predict(obs, deterministic=True)
    weight = model.policy.predict_weights(obs)
    weights.append(weight)
    obs, reward, done, info = eval_env.step(action)
    img = eval_env.render("rgb_array")
    imgs.append(img)
    tot_r += reward
    pbar.update(1)
    i+= 1
    if i > ep_max_len - 1:
        done = True
print("----------------------------")
print(obs)
pbar.close()

print(f"Evaluation Reward: {tot_r}")
weights = np.array(weights).squeeze(1)
fname=os.path.join(save_path, f"{run_id}_weights.npy")
np.save(fname, weights)
ep_len = weights.shape[0]
print(f"Ep Len: {ep_len}")
for i in range(weights.shape[1]):
    plt.plot(weights[:, i], label=f"Model {i}")
plt.xlim(0, ep_len)
plt.ylim(0, 1)
plt.title(f"Weights assigned to primitives {eval_env.envs[0].direction}")
plt.tight_layout()
plt.legend()
fname=os.path.join(save_path, f"{run_id}_weights.jpg")
plt.savefig(fname, bbox_inches="tight", dpi=120)
plt.close()
imgs = np.array(imgs)
fname=os.path.join(save_path, f"{run_id}_eval_video.gif")
fps = 30 if ep_len < 200 else 60
utils.write_gif_to_disk(imgs, fname, fps)
