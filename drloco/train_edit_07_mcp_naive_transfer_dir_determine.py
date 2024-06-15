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
from drloco.common import utils
from drloco.common.schedules import LinearDecay, ExponentialSchedule, CosSchedule, CosDecaySchedule
from drloco.common.callback import TrainingMonitor
from drloco.common.callback import SaveVideoCallback

from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import VecCheckNan, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

from drloco.custom.policies import CustomActorCriticPolicy
from drloco.custom.policies_mcp import MCPNaive, MPPO

import drloco.mujoco.ant_env


# determine the name of saved models before (init) and after training (final)
INIT_CHECKPOINT_SUFFIX = 'init'
FINAL_CHECKPOINT_SUFFIX = 'final'

def use_cpu():
    """
    Force PyTorch to use CPU instead of GPU.
    In some cases, e.g. training many agents in parallel on a CPU cluster,
    it might be useful to use CPU instead of GPU. This function fools PyTorch
    to think there is no GPU available on the PC, so that it uses the CPU.
    """
    from os import environ
    # fool python to think there is no CUDA device
    environ["CUDA_VISIBLE_DEVICES"] = ""
    # to avoid massive slow-down when using torch with cpu
    import torch
    n_envs = cfg.n_envs
    torch.set_num_threads(n_envs if n_envs <= 16 else 8)

def train(args):
    
    # organize argument
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

    # setup environment
    # direction = 0
    if (not direction is None):
        assert not direction is None 
        "please input direction"
    
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
    if "Ant" in cfgl.ENV_ID:
        env = make_vec_env(
            cfgl.ENV_ID, n_envs=cfg.n_envs, monitor_dir=mon_dir, env_kwargs=env_kwargs, seed=seed
        )
        eval_env = make_vec_env(
            cfgl.ENV_ID, n_envs=1, monitor_dir=mon_dir, env_kwargs=env_kwargs, seed=seed
        )
    else:
        env = utils.vec_env(cfgl.ENV_ID, norm_rew=True, num_envs=cfg.n_envs)
        eval_env = utils.vec_env(cfgl.ENV_ID, norm_rew=True, num_envs=1)

    if vec_normalise:
        if os.path.exists(os.path.join(log_dir_primitive, "vec_normalize.pkl")):
            print("Found VecNormalize Stats. Using stats")
            env = VecNormalize.load(os.path.join(log_dir_primitive, "vec_normalize.pkl"), env)
        else:
            print("No previous stats found. Using new VecNormalize instance.")
            env = VecNormalize(env)
    else:
        print("Not using VecNormalize")
    env = VecCheckNan(env, raise_exception=True)
    
    # setup model/algorithm
    training_timesteps = int(cfg.mio_samples * 1e6)
    lr_start = cfg.lr_start
    lr_end = cfg.lr_final

    # learning_rate_schedule = LinearDecay(lr_start, lr_end).value
    period_wave_num = cfg.lr_period_wave_num
    period_wave_len = cfg.lr_period_wave_len_mio / cfg.mio_samples
    learning_rate_schedule = CosDecaySchedule(lr_start, lr_end, period_wave_num, period_wave_len).value

    learn_log_std = False
    num_primitives = 8
    use_mcp_ppo_args = True
    big_model = False

    policy_kwargs = {
    "state_dim": env.observation_space.shape[0] - 2,
    "goal_dim": 2,
    "num_primitives": num_primitives,
    "learn_log_std": learn_log_std,
    "big_model": big_model,
    }

    if use_mcp_ppo_args:
        ppo_kwargs = {
            "learning_rate": 2e-4,
            "n_steps": 4096,
            "batch_size": 256,
            "gamma": 0.95,
            "gae_lambda": 0.95,
            "clip_range": 0.02,
        }
    else:
        ppo_kwargs = {}

    # print model path and modification parameters
    # utils.log('RUN DESCRIPTION: \n' + cfgl.WB_RUN_DESCRIPTION)
    utils.log('Training started',
              ['Model: ' + model_dir, 'Modifications: ' + cfg.modification])

    np.seterr(divide='raise')

    # callback list

    from stable_baselines3.common.callbacks import BaseCallback

    class SaveVideoCallback_custom(SaveVideoCallback):
        def _on_step(self) -> bool:
            if self.n_calls % self.eval_freq == 0:
                obs = self.eval_env.reset()
                eval_direction = eval_env.envs[0].direction
                img = self.eval_env.render("rgb_array")
                imgs = [img]
                done = False
                tot_r = 0.0
                print(f"Begin Evaluation")
                while not done:
                    action, _ = self.model.predict(self.preprocess(obs), deterministic=True)
                    obs, reward, done, info = self.eval_env.step(action)
                    img = self.eval_env.render("rgb_array")
                    imgs.append(img)
                    tot_r += reward
                print(f"Evaluation Reward: {tot_r}")
                ep_len = len(imgs)
                print(f"Ep Len: {ep_len}")
                imgs = np.array(imgs)

                if self.save_path is not None:
                    fname=os.path.join(self.save_path, "eval_video_{:0>5}_deg_{:0>3}_reward_{:0000.3f}.gif".format(self.n_calls // self.eval_freq, eval_direction, tot_r))
                    fps = 30 if ep_len < 200 else 60
                    utils.write_gif_to_disk(imgs, fname, fps)

            return True

    callback_input_array = list()
    checkpoint_callback = CheckpointCallback(int(checkpoint_freq // cfg.n_envs), model_dir, tag_name, 2)
    callback_input_array.append(checkpoint_callback)
    
    if(save_video):
        save_video_callback = SaveVideoCallback_custom(eval_env, int(eval_freq // cfg.n_envs), vec_normalise, log_dir_transfer, 2)
        callback_input_array.append(save_video_callback)
    # callbackList = CallbackList([checkpoint_callback, save_video_callback])
    # callback_input_array.append(TrainingMonitor())# 원래 코드와 save path 차이에 관련된 오류 발생
    callbackList = CallbackList(callback_input_array)

    # train model
    model_path = f"{log_dir_primitive}/final.zip"
    print(log_dir_primitive)
    print("os.path.exists(log_dir_primitive) : {}".format(os.path.exists(log_dir_primitive)))
    print("os.path.exists(model_path) : {}".format(os.path.exists(model_path)))
    assert os.path.exists(model_path)
    mcp_model = PPO.load(model_path, env, verbose=1, tensorboard_log=tbdir, seed=seed)
    # save model and weights before training
    if not cfgl.DEBUG:
    #     utils.save_model(model, model_dir, INIT_CHECKPOINT_SUFFIX)
        model_init_dir = os.path.join(model_dir, INIT_CHECKPOINT_SUFFIX)
        os.makedirs(model_init_dir, exist_ok=True)
        mcp_model.save(model_init_dir)

    mcp_model.policy.freeze_primitives()

    mcp_model.learn(total_timesteps=training_timesteps, callback=callbackList)

    # save model after training
    # utils.save_model(model, checkpoint_dir, FINAL_CHECKPOINT_SUFFIX)
    model_init_dir = os.path.join(model_dir, FINAL_CHECKPOINT_SUFFIX)
    os.makedirs(model_init_dir, exist_ok=True)
    mcp_model.save(model_init_dir)

    # close environment
    env.close()
    eval_env.close()
    # evaluate the trained model
    # eval.eval_model()


if __name__ == '__main__':
    # argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", type=int)
    parser.add_argument("--id", type=str, default="baseline")
    parser.add_argument("--algo", type=str, default="PPO")
    parser.add_argument("--logdir_primitive", type=str, default="logs")
    parser.add_argument("--logdir_transfer", type=str, default="logs_transfer")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vec_normalise", type=str, default="False")
    parser.add_argument("--checkpoint_freq", type=int, default="250000")
    parser.add_argument("--eval_freq", type=int, default="100000")
    parser.add_argument("--save_video", type=str, default="True")
    args = parser.parse_args()

    train(args)
