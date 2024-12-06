import random
import sys
import os

from robot.register import Registers
from robot.scene import RegisterScenes

path = os.path.dirname(__file__)
path = "/".join(path.split("\\")[:-1])
sys.path.append(path)
from robot.utils import *


# 基类中载入了一些设置在scene_parameters.yaml中的通用参数
class BaseScene(object):
    def __init__(self, physicsClientId: int = 0):
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        self._physics_client_id = physicsClientId
        param_path = os.path.join(os.path.dirname(__file__), "config/scene_parameters.yaml")
        param_dict = load(open(param_path, "r", encoding="utf-8"), Loader=Loader)
        for key, value in param_dict.items():
            setattr(self, key, value)
        self.is_built = False  # 是否已经调用过construct函数了
        self.load_items = {}  # 所有载入的物件的id
        self.debug_items = {}  # 所有的debug组件

    def construct(self):
        """
            use loading function in pybullet to load and assign the entity into the scene
            the function must set necessary variable as the attributes of the class
        """
        pass


@Registers.scenes.register("random_scene_with_path")
class RandomSceneWithPath(BaseScene):
    def __init__(self, physicsClientId: int = 0, obstacle_density=0.3):
        super(RandomSceneWithPath, self).__init__(physicsClientId=physicsClientId)
        self.obstacle_density = obstacle_density  # 障碍物的密度
        self.matrix_size = 2

    def dfs(self, matrix, x, y, n, visited):
        """深度优先搜索，检查是否从 (x, y) 到达 (n-1, n-1)"""
        if x < 0 or x >= n or y < 0 or y >= n or matrix[x][y] == 1 or visited[x][y]:
            return False
        if (x, y) == (n - 1, n - 1):  # 到达终点
            return True

        visited[x][y] = True

        # 上下左右四个方向
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            if self.dfs(matrix, x + dx, y + dy, n, visited):
                return True

        return False

    def generate_random_matrix(self, n, obstacle_density):
        """生成一个随机矩阵，并确保有通路"""
        while True:
            matrix = [[0] * n for _ in range(n)]

            # 先保证起点和终点没有障碍物
            matrix[0][0] = 0
            matrix[n - 1][n - 1] = 0

            # 随机生成障碍物，比例控制
            for i in range(n):
                for j in range(n):
                    if (i, j) not in [(0, 0), (n - 1, n - 1)] and random.random() < obstacle_density:
                        matrix[i][j] = 1

            # 确保有路径
            visited = [[False] * n for _ in range(n)]
            if self.dfs(matrix, 0, 0, n, visited):
                return matrix  # 如果有通路，返回矩阵
            print("没有通路，重新生成矩阵")

    def construct(self):
        # 计算障碍物的尺寸，确保整个地图可以完整地填充到环境范围内
        obstacle_size_x = self.INTERNAL_LENGTH / self.matrix_size  # 每个障碍物在X轴方向的大小
        obstacle_size_y = self.INTERNAL_WIDTH / self.matrix_size  # 每个障碍物在Y轴方向的大小
        obstacle_size_z = 1  # 假设障碍物的高度是固定的，或者可以自定义

        # 设置起点和终点
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.load_items["plane"] = p.loadURDF("plane.urdf", useMaximalCoordinates=self.USE_MAX_COOR,
                                              physicsClientId=self._physics_client_id)

        # 随机生成一个有效的矩阵
        matrix = self.generate_random_matrix(self.matrix_size, self.obstacle_density)

        # 随机生成围栏
        self.load_items["fence"] = addFence(
            center_pos=self.CENTER_POS,
            internal_length=self.INTERNAL_LENGTH,
            internal_width=self.INTERNAL_WIDTH,
            height=self.HEIGHT,
            thickness=self.THICKNESS,
            mass=self.FENCE_MASS,
            rgba=self.FENCE_COLOR,
            physicsClientId=self._physics_client_id
        )

        # 计算起点和终点的位置，确保它们不会超出限制范围
        start_marker_pos = [
            self.CENTER_POS[0] - self.INTERNAL_LENGTH / 2 + obstacle_size_x / 2,  # 起点的X位置
            self.CENTER_POS[1] - self.INTERNAL_WIDTH / 2 + obstacle_size_y / 2,  # 起点的Y位置
            self.CENTER_POS[2]  # 起点的Z位置，确保接地
        ]
        self.load_items["start_marker"] = addSphere(
            pos=start_marker_pos,
            radius=0.5,  # 起点的尺寸固定为 0.5
            rgba=[1, 0, 0, 1],  # 红色
            physicsClientId=self._physics_client_id
        )

        end_marker_pos = [
            self.CENTER_POS[0] + self.INTERNAL_LENGTH / 2 - obstacle_size_x / 2,  # 终点的X位置
            self.CENTER_POS[1] + self.INTERNAL_WIDTH / 2 - obstacle_size_y / 2,  # 终点的Y位置
            self.CENTER_POS[2]  # 终点的Z位置，确保接地
        ]
        self.load_items["end_marker"] = addSphere(
            pos=end_marker_pos,
            radius=0.5,  # 终点的尺寸固定为 0.5
            rgba=[0, 1, 0, 1],  # 绿色
            physicsClientId=self._physics_client_id
        )

        # 根据生成的矩阵来添加障碍物，确保它们完全填充到环境内
        for i in range(self.matrix_size):
            for j in range(self.matrix_size):
                if matrix[i][j] == 1:  # 如果是障碍物
                    # 计算障碍物的位置，确保它们位于围栏的范围内
                    obstacle_pos = [
                        self.CENTER_POS[0] - self.INTERNAL_LENGTH / 2 + (i + 0.5) * obstacle_size_x,  # 障碍物的X位置
                        self.CENTER_POS[1] - self.INTERNAL_WIDTH / 2 + (j + 0.5) * obstacle_size_y,  # 障碍物的Y位置
                        self.CENTER_POS[2]  # 障碍物的Z位置，确保接地
                    ]
                    self.load_items[f"obstacle_{i}_{j}"] = addBox(
                        pos=obstacle_pos,
                        halfExtents=[obstacle_size_x / 2, obstacle_size_y / 2, obstacle_size_z / 2],  # 障碍物的尺寸
                        physicsClientId=self._physics_client_id
                    )


if __name__ == "__main__":
    cid = p.connect(p.GUI)
    register_scenes = RegisterScenes()
    scene = register_scenes.construct("random_scene_with_path")

    btn_id = p.addUserDebugParameter("reset", 1, 0, 0)
    previous = p.readUserDebugParameter(btn_id)

    while True:
        key_dict = p.getKeyboardEvents()
        if RIGHT in key_dict:
            p.resetSimulation(physicsClientId=cid)
            scene.construct()
            if scene.matrix_size < scene.INTERNAL_LENGTH//1.4:
                scene.matrix_size += 1
            else:
                scene.obstacle_density +=0.01
                print(scene.obstacle_density)
    p.disconnect(cid)
