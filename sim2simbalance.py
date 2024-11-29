# -*- encoding: utf-8 -*-
# author : Zhelong Huang
import sys
import os
import time
from tkinter import W
path = os.path.dirname(__file__)
path = "/".join(path.split("\\")[:-1])
sys.path.append(path)

import gym
from gym import spaces
from robot.utils import *
from robot.balancebot import *
from robot.scene import *
import glob

from torch import normal, optim

from RL_algorithm.DDPG import DDPG

param_path = os.path.join(os.path.dirname(__file__), "training_parameters.yaml")
param_dict2 = load(open(param_path, "r", encoding="utf-8"), Loader=Loader)

class MaplessNaviEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, scene_name : str = "plane_static_obstacle-A", render : bool = False, evaluate : bool = False):
        """
            :param scene_name: 场景名称(场景是否存在的判断逻辑在_register_scene的construct方法中)
            :param render:     是否需要渲染，训练情况下为了更快的训练速度，一般设为False
            :param evaluate:   是否为评估模式，评估模式下会绘制终点标记
        """
        self.all_scene = ["plane_static_obstacle-A", "plane_static_obstacle-B", "plane_static_obstacle-C"]
        #self.all_scene = ["plane_static_obstacle-A", "plane_static_obstacle-C"]
        self.scene_name = scene_name
        self._render = render
        self._evaluate = evaluate
        # 读入各项参数
        for file in os.listdir("./robot/config"):
            param_path = os.path.join("./robot/config/", file)
            param_dict = load(open(param_path, "r", encoding="utf-8"), Loader=Loader)
            for key, value in param_dict.items():
                setattr(self, key, value)

        # 动作空间: 左轮速度， 右轮速度
        self.action_space = spaces.Box(
            low=np.array([-self.TARGET_VELOCITY, -self.TARGET_VELOCITY]),
            high=np.array([self.TARGET_VELOCITY, self.TARGET_VELOCITY]),
            dtype=np.float32
        )
        # 状态空间: laser1, ..., 5,   distance, alpha
        self.observation_space = spaces.Box(
            low=np.array([0.] * self.LASER_NUM + [0., -1, 0]),
            high=np.array([self.LASER_LENGTH + 1] * self.LASER_NUM + [self.MAX_DISTANCE, 1, 1])
        )
        # 根据参数选择引擎的连接方式
        #self._physics_client_id = p.connect(p.GUI if render else p.DIRECT)
        self._physics_client_id = p.connect(p.GUI if render else p.DIRECT,options="--use_gpu=1 --gpu_device=0")
        
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) # 注释这行打开控件  打开这行，关闭控件
        # p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
        
        # 获取注册环境并reset
        self._register_scene = RegisterScenes()
        self.seed()
        self.reset()

        self.epsion =param_dict2["EPSILON"]
    
    def __reward_func(self, state):

        # 读入各项参数
        diff = np.clip(np.abs((np.abs(state[3]) - np.abs(state[4]))), 0, 100)
        normalized_pitch, pitch_ang_vel, yaw_ang_vel, wheel_l_vel, wheel_r_vel = state
        diff = np.clip(np.abs((np.abs(state[3]) - np.abs(state[4]))), 0, 100)
        yaw_penalty = np.clip(np.abs(yaw_ang_vel), 0, 2)
        reward = 1.0 - np.abs(normalized_pitch) * 6.0  - np.abs(diff * 0.01) - yaw_penalty * 0.5#- np.abs(wheel_l_vel) * 0.01 - np.abs(wheel_r_vel) * 0.01
        return reward

    def __distance(self, v1, v2):
        return sqrt(sum([(x - y) * (x - y) for x, y in zip(v1, v2)]))
    
    def sample(self):
        return self.np_random.uniform(low=-TARGET_VELOCITY, high=TARGET_VELOCITY, size=(2,))
    
    def step(self, action):
        done = False
        self.robot.apply_action(action)
        # p.setTimeStep(1./240.)
        time.sleep(0.01)
        p.stepSimulation(physicsClientId=self._physics_client_id)
        self.step_num += 1
        state = self.__noise_obs(self.robot.get_obs())

        reward = self.__reward_func(state)
        if abs(state[0]) > 0.7: 
            done = True
        # froms, tos, results = rayTest(self.robot.robot, ray_length=self.LASER_LENGTH, ray_num=self.LASER_NUM)
        # for index, result in enumerate(results):
        #     self.rayDebugLineIds[index] = p.addUserDebugLine(
        #         lineFromXYZ=froms[index], 
        #         lineToXYZ=tos[index] if result[0] == -1 else result[3], 
        #         lineColorRGB=self.MISS_COLOR if result[0] == -1 else self.HIT_COLOR, 
        #         lineWidth=self.RAY_DEBUG_LINE_WIDTH, 
        #         replaceItemUniqueId=self.rayDebugLineIds[index]
        #     )
        key_dict = p.getKeyboardEvents()
        state = self.control_miniBox(key_dict, state)
        # state[0] = state[0] - 0.1
        return np.array(state), reward, done, {}
    
    def __noise_obs(self, obs):
        noise = np.random.normal(0, 0.05, 5)
        obs += noise
        # obs[0] = obs[0] - 0.001
        return obs
    
    def reset(self):

        p.resetSimulation(physicsClientId=self._physics_client_id)
        p.setGravity(0., 0., -9.8, physicsClientId=self._physics_client_id)
        self.step_num = 0
        ROBOT_Orientation = [-1.57, 0, 0]
        # ROBOT_Orientation[0] += np.random.uniform(-0.12, 0.12)
        
        self.robot = balance_navi_Robot(
            basePos=self.DEPART_POS,
            baseOri=p.getQuaternionFromEuler(ROBOT_Orientation),
            physicsClientId=self._physics_client_id
        )
        p.setJointMotorControlArray(
                bodyUniqueId=self.robot.robot,
                jointIndices=[LEFT_WHEEL_JOINT_INDEX, RIGHT_WHEEL_JOINT_INDEX],
                controlMode=p.VELOCITY_CONTROL,
                targetVelocities=[-3000, 3000],
                forces=[0, 0],
        )
        self.scene = self._register_scene.construct(scene_name=self.scene_name)
        state = self.robot.get_obs()

        
        self.rayDebugLineIds = []
        # froms, tos, results = rayTest(self.robot.robot, ray_length=self.LASER_LENGTH, ray_num=self.LASER_NUM)
        # for index, result in enumerate(results):
        #     color = self.MISS_COLOR if result[0] == -1 else self.HIT_COLOR
        #     self.rayDebugLineIds.append(p.addUserDebugLine(froms[index], tos[index], color, self.RAY_DEBUG_LINE_WIDTH))
      
        return np.array(state)
    
    def render(self, mode='human'):
        pass
    
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    def close(self):
        if self._physics_client_id >= 0:
            p.disconnect()
        self._physics_client_id = -1

    def control_miniBox(self, key_dict, obs):
        anplify = 1.5
        if len(key_dict) == 0:
            return obs
        elif UP in key_dict:
            obs[0] = obs[0] + 0.2
        elif DOWN in key_dict:
            obs[0] = obs[0] - 0.2
        elif LEFT in key_dict:
            obs[2] = obs[2] + 0.3
        elif RIGHT in key_dict:
            obs[2] = obs[2] - 0.3
        return obs
            
        
    

if __name__ == "__main__":
    env = MaplessNaviEnv(render=True,evaluate=True)
    obs = env.reset()    
    
    agent = DDPG(obs.size, 2, 64) # n_states, n_actions, hidden_size, num_layers
    best_reward = 10
    last = 0
    epcilon = 0.1

    # p.setRealTimeSimulation(1)
    best_flag = False
    files = glob.glob('*best_*')
    for file in files:
        if os.path.exists(file):
            print("file : ", file)
            agent.load(file)
            print("load best") 
        # agent.actor_optimizer = optim.Adam(agent.actor.parameters(), lr=1e-4)
        # agent.critic_optimizer = optim.Adam(agent.critic.parameters(), lr=1e-3)

    """
    TEST
    """
    
    for i in range(1000000):
        if  i% 1 == 0:
            # done = False
            # action = agent.choose_action(True, obs)
            action = np.ones(2) * 1.2 * (1 - (1 / np.exp(i/1000)))
            next_obs, reward, done, _ = env.step(action)
            obs = next_obs
            with open('train_log.txt', 'a') as file:
                        file.write('torque: '+str((1 - (1 / np.exp(i/1000))) *1.2)+' speed: '+ str(p.getJointState(env.robot.robot, RIGHT_WHEEL_JOINT_INDEX)[1])+'\n')  # 追加内容到文件
            if done :
                print("done")
                # env.reset()
        


    """
    TRAIN
    """
    
    # for i in range(100000):
    #     obs = env.reset()
    #     total_reward = 0
    #     done = False
    #     files = glob.glob('*best_*')
    #     if i  ==0 :
    #         for file in files:
    #             if os.path.exists(file):
    #                 print("file : ", file)
    #                 agent.load(file)
    #                 print("load best") 
    #             agent.actor_optimizer = optim.Adam(agent.actor.parameters(), lr=1e-4)
    #             agent.critic_optimizer = optim.Adam(agent.critic.parameters(), lr=1e-3)
    #     for j in range(3000):
    #         if agent.memory.size() > 3000 :
    #             # print("update modle"
    #                 transition_dict = agent.memory.sample(agent.batch_size)
    #                 states, actions, rewards, next_states, dones = transition_dict
    #                 agent.update({
    #                                     'state': states,
    #                                     'action': actions,
    #                                     'reward': rewards,
    #                                     'next_state': next_states,
    #                                     'done': dones
    #                 })
                
                                     
    #         if np.random.rand() < 0.01 and j % 1 ==0:
    #             action = np.random.uniform(-1.2, 1.2, 2)
    #         elif j % 1 == 0:
    #             action = agent.choose_action(True, obs)
    #         next_obs, reward, done, _ = env.step(action)

    #         if j % 2 ==0:
    #             agent.memory.push(obs, action, reward, next_obs, done)

            
                
    #         obs = next_obs
    #         total_reward += reward

    #         if done:
    #             break
                
    #         if total_reward > best_reward:
    #             best_reward = total_reward
    #             best_flag = True
            
    #         if best_flag:
    #             best_flag = False
    #             files = glob.glob('best_*')
    #             for file in files:
    #                 if os.path.exists(file) :
    #                     os.remove(file)
    #             name = "best_" + str(total_reward)
    #             agent.save(name)
    #     print(f"Episode {i}, Reward {total_reward}, Best Reward {best_reward}, j {j}")
    #     if not done: 
    #         last += 1
    #         print("last : ", last)
    #     else:
    #         last = 0
    #     if last > 10:
    #         break
            
