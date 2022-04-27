import numpy as np
import matplotlib.pyplot as plt


def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed


def plot_multi_runs(folder, run_num):
    fig, axs = plt.subplots(2, 5, figsize=(30, 8))

    for i in range(run_num):
        file_name = folder + "_run" + str(i) + "/log.csv"
        data = np.genfromtxt(file_name, delimiter=',')
        x = int(i / 5)
        y = i % 5
        axs[x, y].plot(data[:, 0], smooth(data[:, 1], 0.99))

    plt.show()


def plot_single_run(folder):
    data = np.genfromtxt(folder + "/log.csv", delimiter=',')
    plt_num = data.shape[1] - 1
    fig, axs = plt.subplots(1, plt_num, figsize=(30, 8))
    for i in range(plt_num):
        axs[i].plot(data[:, 0], smooth(data[:, i + 1], 0.99))
        axs[i].grid()
    plt.show()


if __name__ == "__main__":
    # folder = "outputs_batch/centerSquare6x6_3a_PPO_wprior0.998_N1000_gradNoise"
    # run_num = 9
    # plot_multi_runs(folder, run_num)

    folder = 'outputs_lava/centerSquare6x6_2a_PPO_wprior0.995_N1000_gradNoise_run1r0'
    plot_single_run(folder)


