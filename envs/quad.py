from time import sleep
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym import utils
import numpy as np
from numpy.lib.function_base import select
from scipy.spatial.transform import Rotation as R
import os
from gym  import spaces

class QuadState:
    ROT_SEQ='zyx'
    def __init__(self):
        self.state = np.zeros(12)
        self.pos_range = range(0,3)
        self.att_range = range(self.pos_range.stop, self.pos_range.stop+3)
        self.pos_dot_range = range(self.att_range.stop, self.att_range.stop+len(self.pos_range))
        self.att_dot_range = range(self.pos_range.stop, self.pos_range.stop+len(self.att_range))
        
    def set_state_from_components(self, pos, att, pos_dot, att_dot):
        self.set_pos(pos)
        self.set_att(att)
        self.set_pos_dot(pos_dot)
        self.set_att_dot(att_dot)
    
    def set_state_from_vec(self, state_vec):
        self.state[:] = state_vec

    def set_state_from_mujoco_vec(self, state_vec):
        self.set_pos(state_vec[0:3])

        # x, quat_att, x_dot, rpy_dot
        att_quat = state_vec[3:3+4]
        # todo: check rotation order eg. ZXY
        att_euler = R.from_quat(att_quat).as_euler(seq=self.ROT_SEQ)

        self.set_att(att_euler)
        self.set_pos_dot(state_vec[7:7+3])
        self.set_att_dot(state_vec[10:])
    
    def get_mujoco_state(self):
        att_quat = R.from_euler(seq=self.ROT_SEQ, angles=self.get_att()).as_quat()
        qpos=np.hstack((self.get_pos(), att_quat))
        qvel=np.hstack((self.get_pos_dot(), self.get_att_dot()))
        return qpos, qvel

    def get_pos(self):
        return self._get_at_range(self.pos_range)

    def get_att(self):
        return self._get_at_range(self.att_range)

    def get_pos_dot(self):
        return self._get_at_range(self.pos_dot_range)

    def get_att_dot(self):
        return self._get_at_range(self.att_dot_range)

    def get_pose(self):
            return np.hstack((self.get_pos(), self.get_att()))

    def get_pose_dot(self):
            return np.hstack((self.get_pos_dot(), self.get_att_dot()))

    def set_pos(self, pos):
        self._set_at_range(self.pos_range, pos)

    def set_att(self, att):
        self._set_at_range(self.att_range, att)

    def set_pos_dot(self, pos_dot):
        self._set_at_range(self.pos_dot_range, pos_dot)

    def set_att_dot(self, att_dot):
        self._set_at_range(self.att_dot_range, att_dot)

    def vec(self):
        return self.state.copy()

    def _get_at_range(self, range):
        return self.state[range]
        
    def _set_at_range(self, range, vec):
        self.state[range] = vec


# refer: https://github.com/openai/gym/blob/master/gym/envs/mujoco/hopper.py
class QuadEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):        
        self.target_state = QuadState()
        # target_pos = np.array([0,0,0], dtype=np.float32)
        # # rpy
        # target_orientation = np.array([0,0,0], dtype=np.float32)
        # target_vel = np.array([0,0,0], dtype=np.float32)
        # target_ang_vel = np.array([0,0,0], dtype=np.float32)               
        # target_state = np.hstack((target_pos, target_orientation, target_vel, target_ang_vel))
        self.state_tol = QuadState()
        self.state_tol.set_state_from_vec(np.repeat(0.1, self.target_state.vec().size))
        assert((self.state_tol.vec() >= 0).all())

        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()

        CONTROL_FREQ  = 250
        SIM_PHY_FREQ = int(1e3)#np.rint(1/self.model.opt.timestep)
        CONTROL_NUM_STEPS = SIM_PHY_FREQ//CONTROL_FREQ
        self.stepping_freq = CONTROL_FREQ
        mujoco_robot_desc_file = os.path.abspath(os.path.dirname(__file__) + "/../xmls/quad.xml")        
        mujoco_env.MujocoEnv.__init__(self, mujoco_robot_desc_file, CONTROL_NUM_STEPS)
        utils.EzPickle.__init__(self)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        obs, reward, done = self._get_obs(), self._get_reward(), self._get_done()
        if type(done)!=bool:
            done=done.item()
        return obs, reward, done, {}

    def step_freq(self):
        return self.stepping_freq
    
    def _get_obs(self):
        return np.concatenate(
            [self.sim.data.qpos.flat, self.sim.data.qvel.flat]
        )
    
    def _get_state(self):
        state = QuadState()
        state.set_state_from_mujoco_vec(self.state_vector())
        return state

    def _get_reward(self):
        # unweighted
        return -np.linalg.norm(self._get_state().vec() - self.target_state.vec())
       

    def _get_done(self) -> bool:
        return (
            not self._is_within_state_space()
            or self._reached_goal())

    def _is_within_state_space(self):
        state = self._get_state()
        # lower and upper
        pos_space = (np.repeat(-2, self.target_state.get_pos().size),np.repeat(2, self.target_state.get_pos().size))
        return (
            np.isfinite(state.vec()).all()
            # todo: make sure reward not maximised by exiting wall quickly
            and ((pos_space[0] < state.get_pos()).all() and (state.get_pos() < pos_space[1]).all())
            # todo: do we need to check max time steps here?
        )

    def _get_observation_space(self):
        obs_radius = QuadState()
        # todo: check angle and nlg vel. bounds
        obs_radius.set_pos(np.repeat(2., obs_radius.get_pos().size))
        obs_radius.set_att(np.repeat(np.deg2rad(180), obs_radius.get_pos().size))
        obs_radius.set_pos_dot(np.repeat(10., obs_radius.get_pos_dot().size))
        obs_radius.set_att_dot(np.repeat(3*(2*np.pi), obs_radius.get_att_dot().size))

        return spaces.Box(low=np.negative(obs_radius.vec()),
                    high=obs_radius.vec(),
                    dtype=np.float32
                    )

    def _get_action_space(self):
        # recommended as underlying is gaussian
        # ref: https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html#tips-and-tricks-when-creating-a-custom-environment
        return spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
    
    def _reached_goal(self):
        return self._is_within_goal_tol(self._get_state())

    def _is_within_goal_tol(self, state: QuadState):
        return (np.abs(state.vec()-self.target_state.vec())<self.state_tol.vec()).all()
       
    def reset_model(self):
        print('episode complete. resetting')
        state_init_radius = QuadState()
        state_init_radius.set_pos(np.repeat(1., state_init_radius.get_pos().size))
        state_init_radius.set_att(np.repeat(np.deg2rad(45), state_init_radius.get_pos().size))
        state_init_radius.set_pos_dot(np.repeat(0.1, state_init_radius.get_pos_dot().size))
        state_init_radius.set_att_dot(np.repeat(0.1, state_init_radius.get_att_dot().size))

        init_state = QuadState()
        while True:
            init_state.set_state_from_vec(self.np_random.uniform(low=-1, high=1, size=state_init_radius.vec().size)*state_init_radius.vec())
            # SB3 does not allows initial state to be already within goal
            if not self._is_within_goal_tol(init_state):
                break
        qpos, qvel = init_state.get_mujoco_state()
        self.set_state(qpos, qvel)
        return self._get_obs()
    
    def viewer_setup(self):
        pass