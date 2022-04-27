import numpy as np
import torch
import json
import random
import gym
from enum import IntEnum
from envs.rendering import fill_coords, point_in_circle
from envfiles.create_world import create_tile, colors

# left, right, up, down
ACTIONS = [(0, -1), (0, 1), (1, 0), (-1, 0)]


class AppleDoorEnv(gym.Env):

    # Enumeration of possible actions
    class Actions(IntEnum):
        left = 0
        right = 1
        up = 2
        down = 3
        stay = 4

    def __init__(self, json_file, visualize=False):

        with open(json_file) as infile:
            args = json.load(infile)
        self.grid = np.load(args["grid_file"])
        self.reward_mat = np.load(args["reward_file"])
        self.img = np.load(args["img_file"])
        self.height, self.width = self.grid.shape

        self.agent_num = args["agent_num"]
        self.starts = np.array(args["starts"])
        self.goals = np.array(args["goals"])
        self.door = np.array_equal(args["door"])
        self.agents = self.starts.copy()
        self.agents_pre = self.starts.copy()
        self.collide = False

        self.actions = AppleDoorEnv.Actions

        self.state_space = gym.spaces.Box(low=0, high=self.height-1, shape=(2 * self.agent_num, ), dtype='uint8')
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.height, self.width, 3), dtype='uint8')
        self.action_space = gym.spaces.Discrete(4) if self.agent_num == 1 else gym.spaces.Discrete(5)

        self.max_steps = 100
        self.step_count = 0

        self.visualize = visualize
        self.initialize_img()
        self.cur_img = None
        self.window = None

        self.reset()

    def reset(self):
        self.step_count = 0
        self.agents = self.starts.copy()
        self.agents_pre = self.starts.copy()
        if self.visualize:
            self.cur_img = self.img.copy()
            self.update_img()
        return self.state

    @property
    def state(self):
        cur_state = {
            'vec': self.agents.copy()
        }
        if self.visualize:
            cur_state = {
                'image': np.flip(self.cur_img, axis=0),
                'vec': self.agents.copy()
            }
        return cur_state

    @property
    def done(self):
        if np.array_equal(self.agents, self.goals) or (self.step_count >= self.max_steps):
            done = True
        else:
            done = False
        return done

    def _occupied_by_grid(self, i, j):
        if self.grid[i, j] == 1:
            return True
        return False

    def _occupied_by_agent(self, cur_id, i, j):
        for aid in range(self.agent_num):
            if aid == cur_id:
                pass
            if np.array_equal(self.agents[aid], [i, j]):
                self.collide = True
                return True
        return False

    def _available_actions(self, agent_pos):
        available_actions = set()
        available_actions.add(self.actions.stay)
        i, j = agent_pos

        assert (0 <= i <= self.height - 1) and (0 <= j <= self.width - 1), \
            'Invalid indices'

        if (i > 0) and not self._occupied_by_grid(i - 1, j):
            available_actions.add(self.actions.down)
        if (i < self.height - 1) and not self._occupied_by_grid(i + 1, j):
            available_actions.add(self.actions.up)
        if (j > 0) and not self._occupied_by_grid(i, j - 1):
            available_actions.add(self.actions.left)
        if (j < self.width - 1) and not self._occupied_by_grid(i, j + 1):
            available_actions.add(self.actions.right)

        return available_actions

    def _transition(self, actions):
        self.agents_pre = self.agents.copy()
        for aid in random.sample(range(self.agent_num), self.agent_num):
        # for aid in range(self.agent_num):
            action = actions[aid]
            if torch.is_tensor(action):
                action = action.item()
            if action not in self._available_actions(self.agents[aid]):
                pass
            else:
                i, j = self.agents[aid]
                if action == self.actions.up:
                    i += 1
                if action == self.actions.down:
                    i -= 1
                if action == self.actions.left:
                    j -= 1
                if action == self.actions.right:
                    j += 1
                if not self._occupied_by_agent(aid, i, j):
                    self.agents[aid] = [i, j]

    def step(self, actions):
        if not isinstance(actions, list):
            actions = [actions]

        self.step_count += 1
        self._transition(actions)

        if self.visualize:
            self.update_img()

        reward = self._reward()
        self.collide = False
        done = self.done or -2 in reward
        if -2 in reward:
            for i in range(len(reward)):
                reward[i] = 0

        return self.state, reward, done, {}

    def _reward(self):
        rewards = [0] * self.agent_num
        reach_goal = [False] * self.agent_num
        for aid in range(self.agent_num):
            i, j = self.agents[aid]
            if (self.goals[aid][0] == i) and (self.goals[aid][1] == j):
                reach_goal[aid] = True
            else:
                rewards[aid] = -10 if self.collide else self.reward_mat[i, j]
        if np.all(reach_goal):
            for aid in range(self.agent_num):
                rewards[aid] = 10 - 9 * (self.step_count / self.max_steps)
        return rewards

        # version 1: discrete reward only at goal location
        # return 1 - 0.9 * (self.step_count / self.max_steps)
        # version 2: euclidean distance
        # return -np.linalg.norm(self.goal - self.agent_pos)
        # version 3: manhattan distance
        # return -abs(self.goal - self.agent_pos).sum()

    def initialize_img(self, tile_size=30):
        for i in range(len(self.goals)):
            goal_tile = create_tile(tile_size, colors[i])
            x = self.goals[i][0] * tile_size
            y = self.goals[i][1] * tile_size
            self.img[x:x + tile_size, y:y + tile_size] = goal_tile / 2

    def update_img(self, tile_size=30):
        for i in range(self.agent_num):
            x = self.agents_pre[i][0] * tile_size
            y = self.agents_pre[i][1] * tile_size
            self.cur_img[x:x + tile_size, y:y + tile_size] = self.img[x:x + tile_size, y:y + tile_size].copy()

        for i in range(self.agent_num):
            x = self.agents[i][0] * tile_size
            y = self.agents[i][1] * tile_size
            agent_tile = self.cur_img[x:x + tile_size, y:y + tile_size]
            fill_coords(agent_tile, point_in_circle(0.5, 0.5, 0.31), colors[i])
            self.cur_img[x:x + tile_size, y:y + tile_size] = agent_tile

    def render(self, mode="human"):
        if not self.window:
            from envs import window
            self.window = window.Window('Grid World')
            self.window.show(block=False)

        if self.visualize:
            self.update_img()
        self.window.show_img(np.flip(self.cur_img, axis=0))

        return np.flip(self.cur_img, axis=0)
