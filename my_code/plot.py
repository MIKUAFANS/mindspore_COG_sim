import matplotlib.pyplot as plt
import os


# def read_training_data(file_path):
#     with open(file_path, 'r') as file:
#         data = file.readlines()

#     all_trajectories = []
#     current_trajectory = []

#     for line in data:
#         if line.strip() == '==========':
#             if current_trajectory:
#                 all_trajectories.append(current_trajectory)
#                 current_trajectory = []
#         else:
#             parts = line.strip().strip('[]').split(', ')
#             if len(parts) == 5:
#                 x, y, yaw, x0, y0 = map(float, parts)
#                 current_trajectory.append((x, y, yaw, x0, y0))

#     if current_trajectory:  # Add the last trajectory if the file doesn't end with '=========='
#         all_trajectories.append(current_trajectory)

#     return all_trajectories
def read_training_data(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    
    all_trajectories = []
    current_trajectory = []
    
    for line in data:
        if line.strip() == '==========':
            if current_trajectory:
                all_trajectories.append(current_trajectory)
                current_trajectory = []
        else:
            parts = line.strip().strip('[]').split(', ')
            if len(parts) == 13:
                x, y, yaw, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5 = map(float, parts)
                current_trajectory.append((x, y, yaw, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5))
    
    if current_trajectory:  # Add the last trajectory if the file doesn't end with '=========='
        all_trajectories.append(current_trajectory)
    
    return all_trajectories


# def plot_and_save_trajectory(trajectories, output_dir):
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     for i, trajectory in enumerate(trajectories):
#         x_values = [step[0] for step in trajectory]
#         y_values = [step[1] for step in trajectory]
#         x0, y0 = trajectory[0][3], trajectory[0][4]  # Assuming the target is the same for the entire trajectory

#         plt.figure()
#         plt.scatter(x_values, y_values, marker='o', label='Trajectory', alpha=0.7)
#         plt.scatter(x0, y0, color='red', label='Target Position')
#         plt.scatter(x_values[0], y_values[0], color='black', label='Start Position')
#         plt.xlabel('X position')
#         plt.ylabel('Y position')
#         plt.title(f'Training Trajectory {i + 1}')
#         plt.legend()
#         # plt.grid(True)
#         plt.savefig(os.path.join(output_dir, f'trajectory_{i + 1}.png'))
#         plt.close()
def plot_and_save_trajectory(trajectories, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, trajectory in enumerate(trajectories):
        print(f"plot{i}..")
        x_values = [step[0] for step in trajectory]
        y_values = [step[1] for step in trajectory]

        x1, y1 = trajectory[0][3], trajectory[0][4]  # goal1
        x2, y2 = trajectory[0][5], trajectory[0][6]  # goal1
        x3, y3 = trajectory[0][7], trajectory[0][8]  # goal1
        x4, y4 = trajectory[0][9], trajectory[0][10]  # goal1
        x5, y5 = trajectory[0][11], trajectory[0][12]  # goal1

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
        plt.title(f'Training Trajectory {i + 1}')
        plt.legend()
        # plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'trajectory_{i + 1}.png'))
        plt.close()


if __name__ == "__main__":
    file_path = './test.txt'
    output_dir = './plot_result'
    trajectories = read_training_data(file_path)
    plot_and_save_trajectory(trajectories, output_dir)
