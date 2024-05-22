import ast
import os

from matplotlib import pyplot as plt
import numpy as np


def read_log_file():
    with open("logs/test.txt", "r") as file:
        data = file.readlines()

    all_trajectories = []
    for line in data:
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
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, trajectory in enumerate(trajectories[:2]):
        print(f"plot{i}..")
        x_values = [step[0] for step in trajectory]
        y_values = [step[1] for step in trajectory]

        # 0-17 0, 1, 2| 3, 4, 5| 6, 7, 8| 9, 10, 11| 12, 13, 14| 15, 16, 17|
        x1, y1 = trajectory[0][3], trajectory[0][4]  # goal1
        x2, y2 = trajectory[0][6], trajectory[0][7]  # goal1
        x3, y3 = trajectory[0][9], trajectory[0][10]  # goal1
        x4, y4 = trajectory[0][12], trajectory[0][13]  # goal1
        x5, y5 = trajectory[0][15], trajectory[0][16]  # goal1

        plt.figure()
        plt.scatter(x_values, y_values, marker='o', label='Trajectory',color='blue', alpha = 0.7)
        plt.scatter(x1, y1, color='red', label='Goal1')
        plt.scatter(x2, y2, color='yellow', label='Goal2')
        plt.scatter(x3, y3, color='green', label='Goal3')
        plt.scatter(x4, y4, color='orange', label='Goal4')
        plt.scatter(x5, y5, color='pink', label='Goal5')
        plt.scatter(x_values[0], y_values[0], color='black', label='Start Position')
        plt.xlabel('X position')
        plt.ylabel('Y position')
        # plt.title(f'Training Trajectory {i + 1}')
        # plt.legend()
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        # plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'trajectory_{i + 1}.pdf'), format='pdf', bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    output_dir = './plot_result'
    trajectories = read_log_file()
    plot_and_save_trajectory(trajectories, output_dir)