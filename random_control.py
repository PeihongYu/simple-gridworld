from envs.gridworld import GridWorld
import random

env = GridWorld(10, 10)

done = False
while not done:
    action = random.randrange(0, 4)
    obs, reward, done, info = env.step(action)
    env.render()
