import contextlib
import numpy as np
from envs.mpe.core import World, Agent, Landmark
from envs.mpe.scenario import BaseScenario

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


class Scenario(BaseScenario):
    def make_world(self, args=None):
        world = World()
        # set any world properties first
        world.dim_c = 2
        if args is not None:
            num_agents = args.num_agents  #3
        else:
            num_agents = 3
        num_landmarks = 3
        if num_agents == 1:
            self.target_type = "random"  # nearest
            if self.target_type == "random":
                self.target_id = args.mpe_tid   # np.random.randint(num_landmarks)

        self.aid = args.mpe_aid
        self.use_fixed_map = args.mpe_fixed_map
        self.use_sparse_reward = args.mpe_sparse_reward

        world.collaborative = True
        if args.mpe_not_share_reward:
            world.collaborative = False
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def _reset_world_landmarks(self, world):
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def reset_landmarks(self, world):
        if self.use_fixed_map:
            with temp_seed(101):
                self._reset_world_landmarks(world)
        else:
            self._reset_world_landmarks(world)
        # print([l.state.p_pos for l in world.landmarks])

    def _reset_world_agents(self, world, poses=None):
        for aid in range(len(world.agents)):
            agent = world.agents[aid]
            if poses is not None:
                agent.state.p_pos = poses[aid]
            else:
                agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

    def reset_agents(self, world):
        if self.use_fixed_map:
            agent_poses = np.array([[0.0976270078546495, 0.43037873274483895],
                                    [0.7468588055836325, 0.937081325641864],
                                    [-0.5534178416929223, 0.046326682801352215]])
            if len(world.agents) == 1:
                self._reset_world_agents(world, [agent_poses[self.aid]])
            else:
                self._reset_world_agents(world, agent_poses)
        else:
            self._reset_world_agents(world)

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set (random) initial states
        self.reset_agents(world)
        self.reset_landmarks(world)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def _occupied_all_landmarks(self, world):
        dones = []
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            dones.append(min(dists) < world.agents[0].size - world.landmarks[0].size)
        if np.all(dones):
            return True
        return False

    def _occupied_one_landmark(self, agent, world):
        dists = [np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos))) for l in world.landmarks]
        if min(dists) < agent.size - world.landmarks[0].size:
            return True
        return False

    def done(self, agent, world):
        if len(world.agents) == 1:
            return self._occupied_one_landmark(agent, world)
        else:
            return self._occupied_all_landmarks(world)

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        if len(world.agents) == 1:
            if self.target_type == "nearest":
                dists = [np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos))) for l in world.landmarks]
                rew -= min(dists)
            elif self.target_type == "random":
                dist = np.sqrt(np.sum(np.square(agent.state.p_pos - world.landmarks[self.target_id].state.p_pos)))
                rew -= dist
            return rew
        if self.use_sparse_reward:
            if self._occupied_all_landmarks(world):
            # if self._occupied_one_landmark(agent, world):
                rew += 1
        else:
            for l in world.landmarks:
                dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
                rew -= min(dists)
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
