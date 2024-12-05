import sys

import matplotlib.pyplot as plt
import numpy as np
# Initialize lists to store time steps and rewards
frontier_time_steps = []
frontier_ratios = []
frontier_global_eps_mean = []
frontier_global_step_mean = []
frontier_area = []
# Read the file and parse the data
if len(sys.argv) != 3:
    print("Usage: python evalplot.py <path_to_frontier_eval.txt> <path_to_benchmark_eval.txt>")
    sys.exit(1)

file_path = sys.argv[1]
with open(file_path, 'r') as file:
    lines = file.readlines()
    for line in lines[1:-1]:  # Ignore the first row and last 4 rows, which are starting arguments and final results
        if 'num timesteps' in line:
            parts = line.split(',')
            time_step = int(parts[1].split(" ")[3])
            reward_line = lines[lines.index(line) + 1].strip()
            # print(reward_line)
            if 'Rewards:' in reward_line:
                if 'Global' in reward_line:
                    eps_reward_parts = reward_line.split(',')[1].split(':')[1]
                    step_reward_parts = reward_line.split(',')[0].split(':')[2]
                    # print(step_reward_parts)
                    eps_reward_parts = eps_reward_parts.strip()
                    step_reward_parts = step_reward_parts.strip()
                    if len(eps_reward_parts) > 1:
                        frontier_global_eps_mean.append(float(eps_reward_parts.split('/')[0]))
                        frontier_global_step_mean.append(float(step_reward_parts.split('/')[0]))
                        frontier_time_steps.append(time_step)
        if 'Final Exp Ratio' in line:
            ratio_line = lines[lines.index(line) + 1].strip()
            ratios = ratio_line.split(',')
            for ratio in ratios:
                if ratio != '':
                    frontier_ratios.append(float(ratio.strip()))
        if 'Area' in line:
            area_line = lines[lines.index(line) + 1].strip()
            areas = area_line.split(',')
            for area in areas:
                if area != '':
                    frontier_area.append(float(area.strip()))

benchmark_time_steps = []
benchmark_ratio = []
benchmark_global_eps_mean = []
benchmark_global_step_mean = []
benchmark_area = []
file_path = sys.argv[2]
with open(file_path, 'r') as file:
    lines = file.readlines()
    for line in lines[1:-1]:  # Ignore the first row and last 4 rows, which are starting arguments and final results
        if 'num timesteps' in line:
            parts = line.split(',')
            time_step = int(parts[1].split(" ")[3])
            reward_line = lines[lines.index(line) + 1].strip()
            # print(reward_line)
            if 'Rewards:' in reward_line:
                if 'Global' in reward_line:
                    eps_reward_parts = reward_line.split(',')[1].split(':')[1]
                    step_reward_parts = reward_line.split(',')[0].split(':')[2]
                    # print(step_reward_parts)
                    eps_reward_parts = eps_reward_parts.strip()
                    step_reward_parts = step_reward_parts.strip()
                    if len(eps_reward_parts) > 1:
                        benchmark_global_eps_mean.append(float(eps_reward_parts.split('/')[0]))
                        benchmark_global_step_mean.append(float(step_reward_parts.split('/')[0]))
                        benchmark_time_steps.append(time_step)
        if 'Final Exp Ratio' in line:
            ratio_line = lines[lines.index(line) + 1].strip()
            ratios = ratio_line.split(',')
            for ratio in ratios:
                if ratio != '':
                    benchmark_ratio.append(float(ratio.strip()))
        
        if 'Area' in line:
            area_line = lines[lines.index(line) + 1].strip()
            areas = area_line.split(',')
            for area in areas:
                if area != '':
                    benchmark_area.append(float(area.strip()))

# Plot the data
plt.figure()
plt.plot(frontier_time_steps, frontier_global_eps_mean, marker='o',label='Frontier Global Eps Mean')
plt.plot(frontier_time_steps, frontier_global_step_mean, marker='o',label='Frontier Global Step Mean')
plt.plot(frontier_time_steps, benchmark_global_eps_mean, marker='o',label='Benchmark Global Eps Mean')
plt.plot(frontier_time_steps, benchmark_global_step_mean, marker='o',label='Benchmark Step Mean')
plt.legend()
plt.xlabel('Time Steps')
plt.ylabel('Rewards')
plt.title('Reward vs Time Step')
plt.grid(True)


x = np.arange(len(frontier_ratios))
plt.figure()
plt.plot(x, frontier_ratios, marker='o',label='Frontier Exp Ratio')
plt.plot(x, benchmark_ratio, marker='o',label='Benchmark Exp Ratio')
plt.legend()
plt.xlabel('eval intervals')
plt.ylabel('Exp Ratio')
plt.title('Exp Ratio vs eval intervals')
plt.grid(True)

plt.figure()
plt.plot(x, frontier_area, marker='o',label='Frontier Area')
plt.plot(x, benchmark_area, marker='o',label='Benchmark Area')
plt.legend()
plt.xlabel('eval intervals')
plt.ylabel('Exp Area')
plt.title('Exp Area vs eval intervals')
plt.grid(True)

plt.show()
