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
        self._physics_client_id = p.connect(p.GUI if render else p.DIRECT,options="--use_gpu=0 --gpu_device=0")
        
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) # 注释这行打开控件  打开这行，关闭控件
        # p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
        
        # 获取注册环境并reset
        self._register_scene = RegisterScenes()
        self.seed()
        self.reset()

        self.epsion =param_dict2["EPSILON"]

    def _reward(self, state):
        """
            根据状态计算奖励
        """
        distance, alpha = state[-2], state[-1]
        reward = 1. / (1. + distance) * np.cos(alpha)
        return reward
    
    def _done(self, state):
        """
            根据状态判断是否终止
        """
        
        