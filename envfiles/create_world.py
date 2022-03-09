import numpy as np
import json
import matplotlib.pyplot as plt
from envs.rendering import fill_coords, point_in_rect, point_in_line, point_in_circle

colors = [
    (255, 0, 0),  # red
    (0, 255, 0),  # green
    (0, 0, 255),  # blue
    (112, 39, 195),  # purple
    (255, 255, 0)  # yellow
]


def create_floor(tile_size):
    tile = np.zeros([tile_size, tile_size, 3])
    fill_coords(tile, point_in_rect(0, 0.04, 0, 1), (100, 100, 100))
    fill_coords(tile, point_in_rect(0.96, 1, 0, 1), (100, 100, 100))
    fill_coords(tile, point_in_rect(0, 1, 0, 0.04), (100, 100, 100))
    fill_coords(tile, point_in_rect(0, 1, 0.96, 1), (100, 100, 100))
    return tile


def create_lava(tile_size):
    tile = np.zeros([tile_size, tile_size, 3])
    # Background color
    fill_coords(tile, point_in_rect(0, 1, 0, 1), (255, 128, 0))
    # Little waves
    for i in range(3):
        ylo = 0.3 + 0.2 * i
        yhi = 0.4 + 0.2 * i
        fill_coords(tile, point_in_line(0.1, ylo, 0.3, yhi, r=0.03), (0, 0, 0))
        fill_coords(tile, point_in_line(0.3, yhi, 0.5, ylo, r=0.03), (0, 0, 0))
        fill_coords(tile, point_in_line(0.5, ylo, 0.7, yhi, r=0.03), (0, 0, 0))
        fill_coords(tile, point_in_line(0.7, yhi, 0.9, ylo, r=0.03), (0, 0, 0))
    return tile


def create_agent(tile_size, color):
    tile = np.zeros([tile_size, tile_size, 3])
    fill_coords(tile, point_in_circle(0.5, 0.5, 0.31), color)
    return tile


def create_tile(tile_size, color):
    tile = np.zeros([tile_size, tile_size, 3])
    fill_coords(tile, point_in_rect(0, 1, 0, 1), color)
    return tile


def initialize_img(grid, reward, tile_size=30):
    height, width = grid.shape
    img = np.zeros([height * tile_size, width * tile_size, 3], dtype=int)
    floor_tile = create_floor(tile_size)
    wall_tile = create_tile(tile_size, (100, 100, 100))
    lava_tile = create_lava(tile_size)
    reward_min = reward.min()

    for i in range(height):
        for j in range(width):
            x = i * tile_size
            y = j * tile_size
            if grid[i, j] == 1:
                img[x:x + tile_size, y:y + tile_size] = wall_tile
            else:
                if reward[i, j] < 0:
                    img[x:x + tile_size, y:y + tile_size] = lava_tile / (reward[i, j] / reward_min)
                else:
                    img[x:x + tile_size, y:y + tile_size] = floor_tile
    return img


def update_img(img, agents, goals, tile_size=30):

    for i in range(len(goals)):
        goal_tile = create_tile(tile_size, colors[i])
        x = goals[i][0] * tile_size
        y = goals[i][1] * tile_size
        img[x:x + tile_size, y:y + tile_size] = goal_tile

    for i in range(len(agents)):
        # agent_tile = create_agent(tile_size, colors[i])
        x = agents[i][0] * tile_size
        y = agents[i][1] * tile_size
        agent_tile = img[x:x + tile_size, y:y + tile_size]
        fill_coords(agent_tile, point_in_circle(0.5, 0.5, 0.31), colors[i])
        img[x:x + tile_size, y:y + tile_size] = agent_tile

    return img


def create_setup_0():
    file_name = "empty"
    env_name = "./envfiles/" + file_name

    width = 10
    height = 10
    agent_poses = [[0, 0]]
    goal_poses = [[height - 1, width - 1]]
    grid = np.zeros([height, width])
    reward = np.zeros([height, width])

    for i in range(height):
        for j in range(width):
            reward[i, j] = -np.linalg.norm(np.array(goal_poses[0]) - np.array([i, j]))

    img = initialize_img(grid, reward)

    json_dict = {
        "agent_num": 1,
        "starts": agent_poses,
        "goals": goal_poses,
        "grid_file": env_name + "_grid.npy",
        "reward_file": env_name + "_reward.npy",
        "img_file": env_name + "_img.npy"
    }

    np.save(file_name + "_grid.npy", grid)
    np.save(file_name + "_reward.npy", reward)
    np.save(file_name + "_img.npy", img)

    with open(file_name + "_1a.json", 'w') as outfile:
        json.dump(json_dict, outfile, indent=4)

    img = update_img(img, agent_poses, goal_poses)
    plt.imshow(np.flip(img, axis=0))
    plt.axis('off')
    plt.savefig(file_name + "_1a.jpg")
    plt.close()


def create_setup_1():
    # single agent: 10x10 grid map
    # top-left 8x8 grids have -2 as reward
    file_name = "upperLeftSquare"
    env_name = "./envfiles/" + file_name

    width = 10
    height = 10
    agent_poses = [[0, 0]]
    goal_poses = [[height - 1, width - 1]]
    grid = np.zeros([height, width])
    reward = np.zeros([height, width])

    for i in range(height):
        for j in range(width):
            if i > 1 and j < 8:
                reward[i, j] = -2

    img = initialize_img(grid, reward)

    json_dict = {
        "agent_num": 1,
        "starts": agent_poses,
        "goals": goal_poses,
        "grid_file": env_name + "_grid.npy",
        "reward_file": env_name + "_reward.npy",
        "img_file": env_name + "_img.npy"
    }

    np.save(file_name + "_grid.npy", grid)
    np.save(file_name + "_reward.npy", reward)
    np.save(file_name + "_img.npy", img)

    with open(file_name + "_1a.json", 'w') as outfile:
        json.dump(json_dict, outfile, indent=4)

    img = update_img(img, agent_poses, goal_poses)
    plt.imshow(np.flip(img, axis=0))
    plt.axis('off')
    plt.savefig(file_name + "_1a.jpg")
    plt.close()


def create_setup_2():
    # single agent: 10x10 grid map
    # center 8x8 grids have -2 as reward
    file_name = "centerSquare"
    env_name = "./envfiles/" + file_name

    width = 10
    height = 10
    agent_poses = [[0, 0]]
    goal_poses = [[height - 1, width - 1]]
    grid = np.zeros([height, width])
    reward = np.zeros([height, width])

    for i in range(height):
        for j in range(width):
            if 0 < i < 9 and 0 < j < 9:
                reward[i, j] = -2

    img = initialize_img(grid, reward)

    json_dict = {
        "agent_num": 1,
        "starts": agent_poses,
        "goals": goal_poses,
        "grid_file": env_name + "_grid.npy",
        "reward_file": env_name + "_reward.npy",
        "img_file": env_name + "_img.npy"
    }

    np.save(file_name + "_grid.npy", grid)
    np.save(file_name + "_reward.npy", reward)
    np.save(file_name + "_img.npy", img)

    with open(file_name + "_1a.json", 'w') as outfile:
        json.dump(json_dict, outfile, indent=4)

    img = update_img(img, agent_poses, goal_poses)
    plt.imshow(np.flip(img, axis=0))
    plt.axis('off')
    plt.savefig(file_name + "_1a.jpg")
    plt.close()


def create_setup_3():
    # two agent: 10x10 grid map
    # center 8x8 grids have -2 as reward
    file_name = "centerSquare"
    env_name = "./envfiles/" + file_name

    width = 10
    height = 10
    agent_poses = [[0, 0], [height - 1, width - 1]]
    goal_poses = [[height - 1, width - 1], [0, 0]]
    img = np.load(file_name + "_img.npy")

    json_dict = {
        "agent_num": 2,
        "starts": agent_poses,
        "goals": goal_poses,
        "grid_file": env_name + "_grid.npy",
        "reward_file": env_name + "_reward.npy",
        "img_file": env_name + "_img.npy"
    }

    with open(file_name + "_2a.json", 'w') as outfile:
        json.dump(json_dict, outfile, indent=4)

    img = update_img(img, agent_poses, goal_poses)
    plt.imshow(np.flip(img, axis=0))
    plt.axis('off')
    plt.savefig(file_name + "_2a.jpg")
    plt.close()


def create_setup_4():
    # four agent: 10x10 grid map
    # center 8x8 grids have -2 as reward
    file_name = "centerSquare"
    env_name = "./envfiles/" + file_name

    width = 10
    height = 10
    agent_poses = [[0, 0], [0, width - 1], [height - 1, width - 1], [height - 1, 0]]
    goal_poses = [[height - 1, width - 1], [height - 1, 0], [0, 0], [0, width - 1]]
    img = np.load(file_name + "_img.npy")

    json_dict = {
        "agent_num": 4,
        "starts": agent_poses,
        "goals": goal_poses,
        "grid_file": env_name + "_grid.npy",
        "reward_file": env_name + "_reward.npy",
        "img_file": env_name + "_img.npy"
    }

    with open(file_name + "_4a.json", 'w') as outfile:
        json.dump(json_dict, outfile, indent=4)

    img = update_img(img, agent_poses, goal_poses)
    plt.imshow(np.flip(img, axis=0))
    plt.axis('off')
    plt.savefig(file_name + "_4a.jpg")
    plt.close()



if __name__ == '__main__':
    # create_setup_0()
    # create_setup_1()
    # create_setup_2()
    # create_setup_3()
    create_setup_4()