import numpy as np
from enum import IntEnum
from gym import spaces

# left, right, up, down
ACTIONS = [(0, -1), (0, 1), (1, 0), (-1, 0)]


def point_in_rect(xmin, xmax, ymin, ymax):
    def fn(x, y):
        return x >= xmin and x <= xmax and y >= ymin and y <= ymax

    return fn


def fill_coords(img, fn, color):
    """
    Fill pixels of an image with coordinates matching a filter function
    """
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            yf = (y + 0.5) / img.shape[0]
            xf = (x + 0.5) / img.shape[1]
            if fn(xf, yf):
                img[y, x] = color

    return img


class GridWorld2a:

    # Enumeration of possible actions
    class Actions(IntEnum):
        left = 0
        right = 1
        up = 2
        down = 3
        stay = 4

    def __init__(self, height=11, width=11, use_image=False):

        self.actions = GridWorld2a.Actions
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(height, width, 3),
            dtype='uint8'
        )

        self.width = width
        self.height = height
        self.max_steps = 1000  #(width + height) * 2
        self.step_count = 0

        self.agent_pos = [np.zeros(2), np.zeros(2)]
        self.goal = np.zeros(2)
        self.reach_goal = [False, False]

        self.tile_size = 30
        self.img = None
        self.cur_img = None
        self.window = None
        self.use_image = use_image

        self.reset()

    def reset(self):
        self.step_count = 0
        self.agent_pos[0] = np.array([0, 0])
        self.agent_pos[1] = np.array([self.height - 1, self.width - 1])
        self.goal = np.array([(self.height - 1) / 2, (self.width - 1)/2], dtype=int)
        self.reach_goal = [False, False]
        self.img = self.initialize_img()
        self.cur_img = self.update_img()
        return self.get_obs()

    def step(self, action):
        self.step_count += 1

        reach_goal_pre = np.array(self.reach_goal)

        new_poses = [None, None]
        for i in range(len(self.agent_pos)):
            if self.reach_goal[i]:
                new_poses[i] = self.goal
            else:
                pos_delta = ACTIONS[action[i]]
                # boundary check
                new_x = max(0, min(self.width - 1, self.agent_pos[i][1] + pos_delta[1]))
                new_y = max(0, min(self.height - 1, self.agent_pos[i][0] + pos_delta[0]))
                new_poses[i] = np.array([new_y, new_x])
                self.reach_goal[i] = (self.goal[0] == new_y) and (self.goal[1] == new_x)

        # Update position
        if self.reach_goal[0] and self.reach_goal[1]:
            self.agent_pos = new_poses
        elif np.array_equal(self.agent_pos[0], new_poses[1]) and np.array_equal(self.agent_pos[1], new_poses[0]):
            pass
        elif np.array_equal(self.agent_pos[0], new_poses[0]) and np.array_equal(self.agent_pos[0], new_poses[1]):
            pass
        elif np.array_equal(self.agent_pos[1], new_poses[1]) and np.array_equal(self.agent_pos[1], new_poses[0]):
            pass
        elif np.array_equal(new_poses[0], new_poses[1]):
            if np.random.rand() < 0.5:
                self.agent_pos[0] = new_poses[0]
            else:
                self.agent_pos[1] = new_poses[1]
        else:
            self.agent_pos = new_poses

        if self.use_image:
            self.cur_img = self.update_img()

        reward = self._reward() * (1 - reach_goal_pre)
        # if reach_goal:
        #     reward = self._reward()
        # else:
        #     reward = 0
        done = (self.reach_goal[0] and self.reach_goal[1]) or self.step_count == self.max_steps

        return self.get_obs(), reward, done, self.reach_goal

    def _reward(self):
        rewards = [0, 0]
        for i in range(len(self.agent_pos)):
            y, x = self.agent_pos[i]
            if (0 <= y <= 4 and 1 <= x <= 5) or (6 <= y <= 10 and 5 <= x <= 9):
                rewards[i] = -2
            if (self.goal[0] == y) and (self.goal[1] == x):
                rewards[i] = 1 - 0.9 * (self.step_count / self.max_steps)
        return rewards

        # version 1: discrete reward only at goal location
        # return 1 - 0.9 * (self.step_count / self.max_steps)
        # version 2: euclidean distance
        # return -np.linalg.norm(self.goal - self.agent_pos)
        # version 3: manhattan distance
        # return -abs(self.goal - self.agent_pos).sum()

    def initialize_img(self):
        img = np.zeros([self.height * self.tile_size, self.width * self.tile_size, 3], dtype=int)
        tile = np.zeros([self.tile_size, self.tile_size, 3])
        fill_coords(tile, point_in_rect(0, 0.04, 0, 1), (100, 100, 100))
        fill_coords(tile, point_in_rect(0.96, 1, 0, 1), (100, 100, 100))
        fill_coords(tile, point_in_rect(0, 1, 0, 0.04), (100, 100, 100))
        fill_coords(tile, point_in_rect(0, 1, 0.96, 1), (100, 100, 100))
        for i in range(self.height):
            for j in range(self.width):
                x = i * self.tile_size
                y = j * self.tile_size
                img[x:x + self.tile_size, y:y + self.tile_size] = tile

        goal_tile = np.zeros([self.tile_size, self.tile_size, 3])
        fill_coords(goal_tile, point_in_rect(0, 1, 0, 1), (0, 255, 0))

        x = (self.height - self.goal[0] - 1) * self.tile_size
        y = self.goal[1] * self.tile_size
        img[x:x + self.tile_size, y:y + self.tile_size] = goal_tile

        return img

    def update_img(self):

        agent_tile = np.zeros([self.tile_size, self.tile_size, 3])
        fill_coords(agent_tile, point_in_rect(0, 1, 0, 1), (255, 0, 0))

        img = self.img.copy()
        for i in range(len(self.agent_pos)):
            if not self.reach_goal[i]:
                x = (self.height - self.agent_pos[i][0] - 1) * self.tile_size
                y = self.agent_pos[i][1] * self.tile_size
                img[x:x + self.tile_size, y:y + self.tile_size] = agent_tile if i == 0 else agent_tile/2

        return img

    def render(self):
        if not self.window:
            from envs import window
            self.window = window.Window('Grid World')
            self.window.show(block=False)

        if not self.use_image:
            self.cur_img = self.update_img()
        self.window.show_img(self.cur_img)

        return self.cur_img

    def get_obs(self):
        vec = np.hstack([self.agent_pos[0] - self.goal, self.agent_pos[0] - self.goal])
        cur_obs = {
            'vec': vec
        }

        if self.use_image:
            arr = np.zeros([self.height, self.width])
            for agent_pos in self.agent_pos:
                arr[agent_pos[0], agent_pos[1]] = 1
            arr[self.goal[0], self.goal[1]] = 2
            cur_obs = {
                'image': self.cur_img,
                'encode': arr,
                'vec': vec
            }

        return cur_obs
