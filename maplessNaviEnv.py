# -*- encoding: utf-8 -*-
# author : Zhelong Huang
from re import T
import sys
import os
import time
path = os.path.dirname(__file__)
path = "/".join(path.split("\\")[:-1])
sys.path.append(path)

import gym
from gym import spaces
from robot.utils import *
from robot.miniBox import *
from robot.scene import *
import glob

from torch import optim

from RL_algorithm.DDPG import DDPG_LSTM, DDPG

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

        """动作空间: 左轮速度， 右轮速度
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
        根据参数选择引擎的连接方式
        self._physics_client_id = p.connect(p.GUI if render else p.DIRECT)"""
        self._physics_client_id = p.connect(p.GUI if render else p.DIRECT,options="--use_gpu=1 --gpu_device=0")
        
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) # 注释这行打开控件  打开这行，关闭控件
        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
        
        # 获取注册环境并reset
        self._register_scene = RegisterScenes()

        self.seed()
        self.reset()

        self.done_flag=0

        self.pre_pos=self.robot.curPos()

        self.pose_diff=0
        self.angle_diff=0  

        self.epsion =param_dict2["EPSILON"]
    
    def __reward_func(self, state : list):
        if checkCollision(self.robot.robot, debug=False):
            self.collision_num += 1          
            Rc = self.COLLISION_REWARD
            #self.done = True
        else:
            Rc = 0
        
        cur_dis = self.__distance(self.robot.curPos(), self.TARGET_POS)
        #cur_dis=cur_dis/18.0
        Rp = self.DISTANCE_CHANGE_REWARD_COE * (self.pre_dis - cur_dis)
        
        self.pre_dis = cur_dis

        if state[-3] < self.TARGET_RADIUS:#/18.0:
            Rr = self.REACH_TARGET_REWARD
            self.done_flag=200
        else:
            Rr = self.DISTANCE_REWARD_COE * cur_dis / self.depart_target_distance
            #Rr = 0.

        self.pose_diff = self.__distance(self.robot.curPos(), self.pre_pos)*10
    
        
        Rt = self.TIME_REWARD_COE      

        return Rc + Rp + Rr + Rt

    def __distance(self, v1, v2):
        return sqrt(sum([(x - y) * (x - y) for x, y in zip(v1, v2)]))
    
    def sample(self):
        return self.np_random.uniform(low=-TARGET_VELOCITY, high=TARGET_VELOCITY, size=(2,))
    
    def step(self, action):
        """
            first set, second step
            then calculate the reward
            return state, reward, done, info
        """
        self.robot.apply_action(action=action)
        # print("action: ", action)
        p.setTimeStep(0.005)
        p.stepSimulation(physicsClientId=self._physics_client_id)    
        self.step_num += 1
        state = self.robot.get_observation(self.TARGET_POS)
        
        reward = self.__reward_func(state)

        if state[-3] < self.TARGET_RADIUS:#/18.0:           
            done = True
            if self.epsion>0.0001:
                self.epsion-=0.0001
            else:
                self.epsion=0.0001
            
        

        elif self.step_num > self.DONE_STEP_NUM:
            done = True
        else:
            done = False

        if self.collision_num>200:
            done = True
            
        info = {"distance" : state[-2], "collision_num" : self.collision_num}

        # under evaluate mode, extra debug items need to be rendered
        if self._evaluate:
            froms, tos, results = rayTest(self.robot.robot, ray_length=self.LASER_LENGTH, ray_num=self.LASER_NUM)
            for index, result in enumerate(results):
                self.rayDebugLineIds[index] = p.addUserDebugLine(
                    lineFromXYZ=froms[index], 
                    lineToXYZ=tos[index] if result[0] == -1 else result[3], 
                    lineColorRGB=self.MISS_COLOR if result[0] == -1 else self.HIT_COLOR, 
                    lineWidth=self.RAY_DEBUG_LINE_WIDTH, 
                    replaceItemUniqueId=self.rayDebugLineIds[index]
                )

        return np.array(state), reward, done, info
    
    def reset(self):
        """
            what you need do here:
            - reset scene items
            - reload robot
        """
        p.resetSimulation(physicsClientId=self._physics_client_id)
        p.setGravity(0., 0., -9.8, physicsClientId=self._physics_client_id)
        p.setRealTimeSimulation(0)
        self.step_num = 0

        #random init pose and target pose
        pose_flag=np.random.uniform(-1, 1)

        if pose_flag>0:
            self.TARGET_POS[1] = np.random.uniform(7, 9)
            self.TARGET_POS[0] = np.random.uniform(-5, 5)
            self.DEPART_POS[1] = np.random.uniform(-9, -7)
            self.DEPART_POS[0] = np.random.uniform(-5, 5) # type: ignore
        else:
            self.DEPART_POS[1] = np.random.uniform(7, 9)
            self.DEPART_POS[0] = np.random.uniform(-5, 5)
            self.TARGET_POS[1] = np.random.uniform(-9, -7)
            self.TARGET_POS[0] = np.random.uniform(-5, 5)

        #self.TARGET_POS[1]=np.random.uniform(7, 9)
        #self.TARGET_POS[0] = np.random.uniform(-5, 5)
        # print("\033[32m enter \033[0m")
        self.collision_num = 0
        self.pre_dis = self.__distance(self.DEPART_POS, self.TARGET_POS)#/18.0                    # previous distance between robot and target
        self.depart_target_distance = self.__distance(self.DEPART_POS, self.TARGET_POS)     # distance between depart pos and target pos

        #print(self.DEPART_EULER[2])
        self.DEPART_EULER[2]=np.random.uniform(0, 3.14)
        self.DEPART_EULER[0]=1.57
        #self.DEPART_POS[1] = np.random.uniform(-9, -7)
        #self.DEPART_POS[0] = np.random.uniform(-5, 5)
        self.robot = Robot(
            basePos=self.DEPART_POS,
            baseOri=p.getQuaternionFromEuler(self.DEPART_EULER),
            physicsClientId=self._physics_client_id
        )
        if self.scene_name == "mix":
            self.scene_name = np.random.choice(self.all_scene)
        self.scene = self._register_scene.construct(scene_name=self.scene_name)
        state = self.robot.get_observation(targetPos=self.TARGET_POS)
        
        self.done_flag=0
        self.pre_pos=self.robot.curPos()
        
        # add debug items to the target pos
        if self._evaluate:
            self.target_line = p.addUserDebugLine(
                lineFromXYZ=[self.TARGET_POS[0], self.TARGET_POS[1], 0.],
                lineToXYZ=[self.TARGET_POS[0], self.TARGET_POS[1], 5.],
                lineColorRGB=[1., 1., 0.2]
            )
            self.rayDebugLineIds = []
            froms, tos, results = rayTest(self.robot.robot, ray_length=self.LASER_LENGTH, ray_num=self.LASER_NUM)
            for index, result in enumerate(results):
                color = self.MISS_COLOR if result[0] == -1 else self.HIT_COLOR
                self.rayDebugLineIds.append(p.addUserDebugLine(froms[index], tos[index], color, self.RAY_DEBUG_LINE_WIDTH))

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

    def balance_obs(self):
        return self.robot.balance_obs()
    def balance_reset(self):
        # p.resetStepSimulation(physicsClientId=self._physics_client_id)
        p.resetSimulation(physicsClientId=self._physics_client_id)
        p.setGravity(0., 0., -9.8, physicsClientId=self._physics_client_id)
        # p.setRealTimeSimulation(0)
        ROBOT_Orientation = [1.57, 0, 0]
        self.robot = Robot(
            basePos=[8., 5., 0.6],
            baseOri=p.getQuaternionFromEuler(ROBOT_Orientation),
            physicsClientId=self._physics_client_id
        )

        self.scene = self._register_scene.construct(scene_name=self.scene_name)
        if self._evaluate:
            self.target_line = p.addUserDebugLine(
                lineFromXYZ=[self.TARGET_POS[0], self.TARGET_POS[1], 0.],
                lineToXYZ=[self.TARGET_POS[0], self.TARGET_POS[1], 5.],
                lineColorRGB=[1., 1., 0.2]
            )
            self.rayDebugLineIds = []
            froms, tos, results = rayTest(self.robot.robot, ray_length=self.LASER_LENGTH, ray_num=self.LASER_NUM)
            for index, result in enumerate(results):
                color = self.MISS_COLOR if result[0] == -1 else self.HIT_COLOR
                self.rayDebugLineIds.append(p.addUserDebugLine(froms[index], tos[index], color, self.RAY_DEBUG_LINE_WIDTH))

        return self.robot.balance_obs()
    def balance_step(self, action):
        self.robot.apply_action(action=action)
        p.setTimeStep(1./240.)
        p.stepSimulation(physicsClientId=self._physics_client_id)
        state = self.robot.balance_obs()

        froms, tos, results = rayTest(self.robot.robot, ray_length=self.LASER_LENGTH, ray_num=self.LASER_NUM)
        for index, result in enumerate(results):
            self.rayDebugLineIds[index] = p.addUserDebugLine(
                    lineFromXYZ=froms[index], 
                    lineToXYZ=tos[index] if result[0] == -1 else result[3], 
                    lineColorRGB=self.MISS_COLOR if result[0] == -1 else self.HIT_COLOR, 
                    lineWidth=self.RAY_DEBUG_LINE_WIDTH, 
                    replaceItemUniqueId=self.rayDebugLineIds[index]
                )
        
        return np.array(state), 0, False, {}
        
    

if __name__ == "__main__":
    env = MaplessNaviEnv(render=True,evaluate=True)
    
    # obs = env.reset()
    obs = env.balance_reset()

    
    # agent = DDPG_LSTM(obs.size, 2, 128, 2) # n_states, n_actions, hidden_size, num_layers
    agent = DDPG(5, 2, 64) # n_states, n_actions, hidden_size, num_layers
    best_reward = 0
    last_reward = 0

    best_flag = False

    files = glob.glob('*0*')
    for file in files:
            if os.path.exists(file):
                print("file : ", file)
                agent.load_from_onnx(file)
                print("load best")
    


    for epochs in range(10000):

        total_reward = 0
        steps = 0
        # obs = env.reset()
        obs = env.balance_reset()

        if best_flag:
            files = glob.glob('*best*')
            for file in files:
                if os.path.exists(file):
                    print("file : ", file)
                    agent.load_from_onnx(file)
                    print("load best\n")
            best_flag = False

        

        """
        print("=================================================")
        print("epochs:",epochs)
        print("=================================================")

        while True:
            if epochs%1000000==0 and epochs>10000:
                ev_reward=0
                
                for i in range(10): #test 10 times
                    total_reward, steps = 0., 0
                    obs = env.reset()
                    while True:
                        if env.step_num%2==0:
                            action = agent.choose_action(True, obs)
                        next_obs, reward, done, _ = env.step(action)
                        obs = next_obs
                        total_reward += reward

                        print(f"Test {i}: total reward: {total_reward:.3f} , step: {env.step_num:.3f}",end='\r')
                        sys.stdout.flush()  # 立即刷新输出

                        if done:
                            ev_reward+=total_reward
                            print(f"Test {i}: total reward: {total_reward:.3f} , step: {env.step_num:.3f}",end='\r')
                            print("\n")
                            with open('test_log.txt', 'a') as file:
                                file.write(str(epochs)+' '+str(ev_reward)+'\n')  # 追加内容到文件
                            break
               
                #ev_reward/=5

                    
                if ev_reward >= best_reward and ev_reward>20:
                    print(f"Test score: {ev_reward:.3f},find best!")
                    best_flag=1
                    best_reward = ev_reward
                    files = glob.glob('best_*')
                    for file in files:
                        if os.path.exists(file) :
                            os.remove(file)
                    name = "best_" + str(best_reward)
                    agent.save(name)

                    #agent.critic_optimizer = optim.Adam(agent.critic_lstm.parameters(), lr=1e-4)  #lr=3e-3
                    #agent.actor_optimizer = optim.Adam(agent.actor_lstm.parameters(), lr=1e-4) #lr=3e-4

                    break
                else:
                    print(f"Test score: {ev_reward:.3f},not good")
                    break

            
            else:
            #===============================================================
            #Train
            #===============================================================

                if env.step_num%1==0:
                        if np.random.random() < env.epsion or agent.memory.size() < 2000:
                                action = np.random.uniform(-1., 1., size=(2,))             
                        else:                       
                                action = agent.choose_action(True, obs)

                

                next_obs, reward, done, _ = env.step(action)

                if env.step_num%1==0:
                    agent.memory.push(obs, action[:2], reward, next_obs, done)

                obs = next_obs
                
                if agent.memory.size() > 1000:
                    if env.step_num%5==0:
                        transition_dict = agent.memory.sample(agent.batch_size)
                        states, actions, rewards, next_states, dones = transition_dict
                        agent.update({
                                        'state': states,
                                        'action': actions,
                                        'reward': rewards,
                                        'next_state': next_states,
                                        'done': dones
                                    })
                        agent.memory.clean()
                
                total_reward += reward
                print(f"total reward: {total_reward:.3f} , epsion: {env.epsion:.3f}",end=' ')
                sys.stdout.flush()  # 立即刷新输出
            
                print(f"完成进度: {(int)((env.step_num/env.DONE_STEP_NUM)*100)} %", end='\r')  # \r 表示回到行首，end='' 表示不换行
                sys.stdout.flush()  # 立即刷新输出

                if done:
                    with open('train_log.txt', 'a') as file:
                        file.write('epoch: '+str(epochs)+' reward: '+str(total_reward)+' '+str(env.done_flag)+'\n')  # 追加内容到文件
                    break

        print("epochs: ")
        print(epochs)
        print("  ")
        print("total reward: ")
        print(total_reward)
        print("best reward: ")
        print(best_reward)
        print("memory size")
        print(agent.memory.size())
        """
        
        while True:
            # key_dict = p.getKeyboardEvents()
            # env.robot.control_miniBox(key_dict)
            action = agent.choose_action(True, obs) * 500
            action[0] = action[0]*(-1)
            # action = action * (-1)
            next_obs, reward, done, _ = env.balance_step(action)
            obs = next_obs

            
            
        
           
    
