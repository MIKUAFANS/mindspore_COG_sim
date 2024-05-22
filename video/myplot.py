import ast
import os

from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm


def read_log_file():
    with open("test.txt", "r") as file:
        data = file.readlines()

    all_trajectories = []
    for idx, line in enumerate(data):
        # print(idx)
        # if line.strip() == "==========Done!" or idx != 4:
        if line.strip() == "==========Done!":
            # break
            continue
        try:
            array_list = ast.literal_eval(line)
        except:
            # print(line)
            raise ValueError("Error in reading the log file")
            # array_expr = eval(line)
            # print(array_expr)
        all_trajectories.append(array_list)

    return all_trajectories

def plot_and_save_trajectory(trajectories, output_dir):
    for i, trajectory in enumerate(trajectories):
        path = output_dir + f"/{i+1}"
        if not os.path.exists(path):
            os.makedirs(path)
        print(f"plot{i}..")
        x_values = [step[0] for step in trajectory]
        y_values = [step[1] for step in trajectory]

        # 0-17 0, 1, 2| 3, 4, 5| 6, 7, 8| 9, 10, 11| 12, 13, 14| 15, 16, 17|
        x1, y1 = trajectory[0][3], trajectory[0][4]  # goal1
        x2, y2 = trajectory[0][6], trajectory[0][7]  # goal1
        x3, y3 = trajectory[0][9], trajectory[0][10]  # goal1
        x4, y4 = trajectory[0][12], trajectory[0][13]  # goal1
        x5, y5 = trajectory[0][15], trajectory[0][16]  # goal1
        reached = []


        for idx, step in tqdm(enumerate(trajectory)):
            plt.figure()
            # plt.scatter(x_values, y_values, marker='o', label='Trajectory',color='blue', alpha = 0.7)
            x, y = step[0], step[1]
            plt.plot(x, y, marker='o', label='Trajectory', color='blue', alpha=0.7)
            plt.scatter(x1, y1, color='red', label='Goal1')
            plt.scatter(x2, y2, color='yellow', label='Goal2')
            plt.scatter(x3, y3, color='green', label='Goal3')
            plt.scatter(x4, y4, color='orange', label='Goal4')
            plt.scatter(x5, y5, color='pink', label='Goal5')
            plt.scatter(x_values[0], y_values[0], color='black', label='Start Position')
            if step[5]:
                if len(reached) == 0:
                    reached.append((x, y))
                plt.scatter(reached[0][0], reached[0][1], color='red', marker="*", label='Goal1 Reached')
            else:
                plt.scatter(x, y, color='white', marker='*', label='Goal1 Reached', alpha=1.0)
            if step[8]:
                if len(reached) == 1:
                    reached.append((x, y))
                plt.scatter(reached[1][0], reached[1][1], color='yellow', marker="*", label='Goal2 Reached')
            else:
                plt.scatter(x, y, color='white', marker='*', label='Goal2 Reached', alpha=1.0)
            if step[11]:
                if len(reached) == 2:
                    reached.append((x, y))
                plt.scatter(reached[2][0], reached[2][1], color='green', marker="*", label='Goal3 Reached')
            else:
                plt.scatter(x, y, color='white', marker='*', label='Goal3 Reached', alpha=1.0)
            if step[14]:
                if len(reached) == 3:
                    reached.append((x, y))
                plt.scatter(reached[3][0], reached[3][1], color='orange', marker="*", label='Goal4 Reached')
            else:
                plt.scatter(x, y, color='white', marker='*', label='Goal4 Reached', alpha=1.0)
            if step[17]:
                if len(reached) == 4:
                    reached.append((x, y))
                plt.scatter(reached[4][0], reached[4][1], color='pink', marker="*", label='Goal5 Reached')
            else:
                plt.scatter(x, y, color='white', marker='*', label='Goal5 Reached', alpha=1.0)
            plt.xlabel('X position')
            plt.ylabel('Y position')
            plt.xticks(np.arange(-1, 10, 1))
            plt.yticks(np.arange(-1, 6, 1))
            # plt.title(f'Training Trajectory {i + 1}')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            # plt.grid(True)
            # plt.savefig(os.path.join(output_dir, f'trajectory_{i + 1}.png'))
            # plt.savefig(os.path.join(output_dir, f'trajectory_{idx + 1}.png'))
            plt.savefig(os.path.join(path, f'{idx + 1}.png'), bbox_inches='tight')
            plt.close()

if __name__ == "__main__":
    # output_dir = './plot_result'
    output_dir = './pics'
    trajectories = read_log_file()
    plot_and_save_trajectory(trajectories, output_dir)
    # for step in trajectories[0]:
    #     print(step[5], step[8], step[11], step[14], step[17])