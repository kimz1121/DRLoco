import numpy as np

from drloco.config import config as cfgl
from drloco.config import hypers as cfg
from drloco.common.utils import get_project_path
from drloco.common.utils import exponential_running_smoothing as smooth
from drloco.common.utils import log
from drloco.common.utils import is_remote

from mujoco_py.builder import MujocoException

from drloco.mujoco.mimic_env import MimicEnv
import drloco.ref_trajecs.loco3d_trajecs as refs

# specify which joint trajectories are required for the current walker
ref_trajecs_qpos_indices = [refs.PELVIS_TX, refs.PELVIS_TZ, refs.PELVIS_TY,
                            refs.PELVIS_LIST, refs.PELVIS_TILT, refs.PELVIS_ROTATION,
                            refs.LUMBAR_BENDING, refs.LUMBAR_EXTENSION, refs.LUMBAR_ROTATION,
                            refs.HIP_FLEXION_R, refs.HIP_ADDUCTION_R, refs.HIP_ROTATION_R,
                            refs.KNEE_ANG_R, refs.ANKLE_ANG_R,
                            refs.HIP_FLEXION_L, refs.HIP_ADDUCTION_L, refs.HIP_ROTATION_L,
                            refs.KNEE_ANG_L, refs.ANKLE_ANG_L]

# the indices in the joint position and joint velocity matrices are the same for all joints
ref_trajecs_qvel_indices = ref_trajecs_qpos_indices

# empty, as model representing the same subject the mocap data was collected from!
adaptations = {}

class MimicWalker165cm65kgEnv(MimicEnv):
    def __init__(self):
        # init reference trajectories
        ref_trajecs = refs.Loco3dReferenceTrajectories(
            ref_trajecs_qpos_indices, ref_trajecs_qvel_indices, adaptations)
        # specify absolute path to the MJCF file
        mujoco_xml_file = get_project_path() + 'drloco/mujoco/xml/walker_165cm_65kg.xml'
        # init the mimic environment
        MimicEnv.__init__(self, mujoco_xml_file, ref_trajecs)

    # ----------------------------
    # Methods we override:
    # ----------------------------
    def step(self, action):
        # when rendering: pause sim on startup to change rendering speed, camera perspective etc.
        # todo: make it a constant in the config file or a constant in the mimicEnv here.
        pause_mujoco_viewer_on_start = True and not is_remote()
        if pause_mujoco_viewer_on_start:
            self._get_viewer('human')._paused = True
            pause_mujoco_viewer_on_start = False

        # todo: the base class should have a method
        #  called modify_actions or preprocess actions instead of this?
        #  otherwise document, that we're rescaling actions
        #  and add action ranges into the environment (get_action_ranges())
        action = self._rescale_actions(action)

        # todo: remove that after you've made sure, the simple env works as before
        # todo: Add a static method to each environment
        #  that allows to mirror the experiences (s,a,r,s')
        # when we're mirroring the policy (phase based mirroring), mirror the action
        if cfg.is_mod(cfg.MOD_MIRR_POLICY) and self.refs.is_step_left():
            action = self.mirror_action(action)

        # execute simulation with desired action for multiple steps
        try:
            self.do_simulation(action, self._frame_skip)
            # self.render()
        # If a MuJoCo Exception is raised, catch it and reset the environment
        except MujocoException as mex:
            log('MuJoCo Exception catched!',
                [f'- Episode Duration: {self.ep_dur}',
                 f'Exception: \n {mex}'])
            obs = self.reset()
            return obs, 0, True, {}



        # increment the current position on the reference trajectories
        self.refs.next()

        # get state observation after simulation step
        obs = self._get_obs()

        # workaround due to MujocoEnv calling step() during __init__()
        if not self.finished_init:
            return obs, 3.33, False, {}

        # increment episode duration
        self.ep_dur += 1

        # update the so far traveled distance
        self.update_walked_distance()

        # todo: add a function is_done() that can be overwritten
        # check if we entered a terminal state
        com_z_pos = self.get_COM_Z_position()
        # was max episode duration or max walking distance reached?

        max_eplen_reached = self.ep_dur >= cfg.ep_dur_max

        # terminate the episode?
        # todo: should be is_done() or self.ep_dur >= cfg.ep_dur_max
        done = com_z_pos < 0.5 or max_eplen_reached

        # todo: do we need this necessarily in the simple straight walking case?
        # terminate_early, _, _, _ = self.do_terminate_early()
        reward = self.get_reward(done)
        if(done == False):
            self.ep_rews.append(reward)

        return obs, reward, done, {}
    
    def get_reward(self, done: bool):
        """ Returns the reward of the current state.
            :param done: is True, when episode finishes and else False"""
        return self._get_ET_reward() if done \
            else self.get_imitation_reward() + cfg.alive_bonus


    def _get_ET_reward(self):
        """ Punish falling hard and reward reaching episode's end a lot. """

        # calculate a running mean of the ep_return
        self.mean_epret_smoothed = smooth('mimic_env_epret', np.sum(self.ep_rews), 0.5)
        self.ep_rews = []

        # reward reaching the end of the episode without falling
        # reward = expected cumulative future reward
        max_eplen_reached = self.ep_dur >= cfg.ep_dur_max
        if max_eplen_reached:
            # estimate future cumulative reward expecting getting the mean reward per step
            mean_step_rew = self.mean_epret_smoothed / self.ep_dur
            act_ret_est = np.sum(mean_step_rew * np.power(cfg.gamma, np.arange(self.ep_dur)))
            reward = act_ret_est
        # punish for ending the episode early
        else:
            reward = -1 * self.mean_epret_smoothed

        return reward
    
    def _get_COM_indices(self):
        return [0,1,2]

    def _get_trunk_rot_joint_indices(self):
        return [3, 4, 5]

    def get_joint_indices_for_phase_estimation(self):
        # return both hip joint indices in the saggital plane
        # and both knee joint indices in the saggital plane
        return [9,12,14,17]

    def _get_not_actuated_joint_indices(self):
        return self._get_COM_indices() + [3,4,5]