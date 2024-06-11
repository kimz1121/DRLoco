import numpy as np

from drloco.config import config as cfgl
from drloco.config import hypers as cfg
from drloco.common.utils import get_project_path
from drloco.common.utils import exponential_running_smoothing as smooth

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
            reward = -1

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