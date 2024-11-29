import re
import sys 
import os
import numpy as np
from robot.utils import *
from functools import partial

path = os.path.dirname(__file__)
path = "/".join(path.split("\\")[:-1])
sys.path.append(path)

class balance_navi_Robot(object):
    def __init__(self, 
                 basePos : list = [0., 0., 0.], 
                 baseOri : list = [0., 0., 0., 1.], 
                 physicsClientId : int = 0):
        self._physics_client_id = physicsClientId
        # 读入各项参数
        param_path = os.path.join(os.path.dirname(__file__), "config/miniBox_parameters.yaml")
        param_dict = load(open(param_path, "r", encoding="utf-8"), Loader=Loader)
        for key, value in param_dict.items():
            setattr(self, key, value)

        urdf_path = os.path.join(os.path.dirname(__file__), "urdf/balancebot.urdf")
        self.robot = p.loadURDF(
            fileName=urdf_path,
            basePosition=basePos,
            baseOrientation=baseOri,
            # useMaximalCoordinates=self.USE_MAX_COOR,
            physicsClientId=physicsClientId
        )
        
        # 该偏函数用于将输入的速度进行合适的裁剪
        self.clipv = partial(np.clip, a_min=-TARGET_VELOCITY, a_max=TARGET_VELOCITY)
        
        self.prePos = basePos

    def get_bothId(self):
        return self._physics_client_id, self.robot  
    
    def apply_action(self, action):     # 施加动作
        if not (isinstance(action, list) or isinstance(action, np.ndarray)):
            assert f"apply_action() only receive list or ndarray, but receive {type(action)}"
        left_t, right_t = action

        p.setJointMotorControl2(
            bodyUniqueId=self.robot,
            jointIndex=LEFT_WHEEL_JOINT_INDEX,
            controlMode=p.TORQUE_CONTROL,
            force=-left_t,
            maxVelocity = 10000
        )
        p.setJointMotorControl2(
            bodyUniqueId=self.robot,
            jointIndex=RIGHT_WHEEL_JOINT_INDEX,
            controlMode=p.TORQUE_CONTROL,
            force=right_t,
            maxVelocity=10000
        )

    def get_obs(self):

        # _, _, results = rayTest(self.robot, self.LASER_LENGTH, self.LASER_NUM)
        # lasers_info = [self.LASER_LENGTH if result[0] == -1 else self.__distance(basePos, result[3]) for index, result in enumerate(results)]
        basePos, baseOri = p.getBasePositionAndOrientation(self.robot, physicsClientId=self._physics_client_id)
        baseVel, baseAngVel = p.getBaseVelocity(self.robot, physicsClientId=self._physics_client_id)
        baseEuler = p.getEulerFromQuaternion(baseOri)
        baseEuler = np.array(baseEuler)
        #归一化euler, 使其在-pi到pi之间
        
        baseEuler = (baseEuler + np.pi) % (2 * np.pi) - np.pi
        normalized_pitch = baseEuler[0] - (np.pi / 2)
        # normalized_pitch = -baseEuler[0]
        pitch_ang_vel = baseAngVel[0]
        yaw_ang_vel = baseAngVel[2]
        # print("yaw_ang_vel : ", yaw_ang_vel)

        wheel_l_vel = p.getJointState(self.robot, LEFT_WHEEL_JOINT_INDEX, physicsClientId=self._physics_client_id)[1]
        # print("wheel_l_vel : ", p.getJointState(self.robot, LEFT_WHEEL_JOINT_INDEX, physicsClientId=self._physics_client_id))
        wheel_r_vel = p.getJointState(self.robot, RIGHT_WHEEL_JOINT_INDEX, physicsClientId=self._physics_client_id)[1]

        return [normalized_pitch, pitch_ang_vel, yaw_ang_vel, wheel_l_vel, wheel_r_vel]

    def __distance(self, v1, v2):
        return sqrt(sum([(x - y) * (x - y) for x, y in zip(v1, v2)]))
