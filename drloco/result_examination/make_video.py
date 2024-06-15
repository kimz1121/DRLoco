# add current working directory to the system path
import sys
from os import getcwd
sys.path.append(getcwd())
sys.path.append('/home/lab6/robot_learn/DRLoco')
sys.path.append('/home/kimz1121/robot_learn/DRLoco')

# import required modules
import argparse
import os.path
import numpy as np
import wandb

import torch as th

from drloco.config import config as cfgl
from drloco.config import hypers as cfg

from stable_baselines3.common.env_util import make_vec_env

parser = argparse.ArgumentParser()
parser.add_argument("--direction", type=int)
parser.add_argument("--id", type=str, default="baseline")
parser.add_argument("--algo", type=str, default="PPO")
parser.add_argument("--logdir_primitive", type=str, default="logs")
parser.add_argument("--logdir_transfer", type=str, default="logs_transfer")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--vec_normalise", type=str, default="False")
parser.add_argument("--checkpoint_freq", type=int, default="250000")
parser.add_argument("--eval_freq", type=int, default="1000000")
parser.add_argument("--save_video", type=str, default="True")
args = parser.parse_args()

run_id = args.id
direction = args.direction
algo = args.algo
logdir_primitive = args.logdir_primitive
logdir_transfer = args.logdir_transfer
seed = args.seed
vec_normalise = args.vec_normalise == "True" or args.vec_normalise == "true"
checkpoint_freq = args.checkpoint_freq
eval_freq = args.eval_freq
save_video = args.save_video == "True" or args.save_video == "true"

tag_name = os.path.join(f"{cfgl.ENV_ID}", f"{algo}_{run_id}")
# log_dir_primitive = os.path.join(logdir_primitive, tag_name, f"seed{str(seed)}")
log_dir_primitive = os.path.join(logdir_primitive)
log_dir_transfer = os.path.join(logdir_transfer, tag_name, f"seed{str(seed)}")
model_dir = os.path.join(log_dir_transfer, "models")
mon_dir = os.path.join(log_dir_transfer, "gym")
tbdir = os.path.join(log_dir_transfer, "tb_logs")

checkpoint_dir = os.path.join(log_dir_transfer, "checkpint")

# make torch using the CPU instead of the GPU
if cfgl.USE_CPU: use_cpu()

# create model directories
if not os.path.exists(log_dir_transfer):
    os.makedirs(log_dir_transfer, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(mon_dir, exist_ok=True)

direction = 0

env_direction = {
    0: 0,
    1: 180,
    2: 90,
    3: 270,
}
env_kwargs = {
    "direction": 0,
    "direction_range": None,
    "direction_list": [direction]
}

eval_env = make_vec_env(
    cfgl.ENV_ID, n_envs=1, env_kwargs=env_kwargs, seed=seed
)
 
model_path = f"{log_dir_primitive}/final.zip"
assert os.path.exists(model_path)
mcp_model = model = PPO.load(model_path, eval_env)

obs = eval_env.reset()
eval_direction = eval_env.direction
img = eval_env.render("rgb_array")
imgs = [img]
done = False
tot_r = 0.0
print(f"Begin Evaluation")
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = eval_env.step(action)
    img = eval_env.render("rgb_array")
    imgs.append(img)
    tot_r += reward
print(f"Evaluation Reward: {tot_r}")
ep_len = len(imgs)
print(f"Ep Len: {ep_len}")
imgs = np.array(imgs)

if self.save_path is not None:
    fname=os.path.join(self.save_path, "eval_video_{:0>5}_deg_{:0>3}.gif".format(self.n_calls // self.eval_freq, eval_direction))
    fps = 30 if ep_len < 200 else 60
    utils.write_gif_to_disk(imgs, fname, fps)