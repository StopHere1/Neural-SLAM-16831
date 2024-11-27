import sys

import matplotlib.pyplot as plt

# Initialize lists to store time steps and rewards
time_steps = []
rewards = []
global_eps_mean = []
global_step_mean = []
# Read the file and parse the data
if len(sys.argv) != 2:
    print("Usage: python evalplot.py <path_to_eval2.txt>")
    sys.exit(1)

file_path = sys.argv[1]
with open(file_path, 'r') as file:
    lines = file.readlines()
    for line in lines[1:-4]:  # Ignore the first row and last 4 rows, which are starting arguments and final results
        if 'num timesteps' in line:
            parts = line.split(',')
            time_step = int(parts[1].split(" ")[3])
            reward_line = lines[lines.index(line) + 1].strip()
            # print(reward_line)
            if 'Rewards:' in reward_line:
                if 'Global' in reward_line:
                    eps_reward_parts = reward_line.split(',')[1].split(':')[1]
                    step_reward_parts = reward_line.split(',')[0].split(':')[2]
                    print(step_reward_parts)
                    eps_reward_parts = eps_reward_parts.strip()
                    step_reward_parts = step_reward_parts.strip()
                    if len(eps_reward_parts) > 1:
                        # print(eps_reward_parts.split('/'))
                        global_eps_mean.append(float(eps_reward_parts.split('/')[0]))
                        global_step_mean.append(float(step_reward_parts.split('/')[0]))
                        # reward = float(reward_parts[0].split()[4])
                        time_steps.append(time_step)
                        # rewards.append(reward)

# print('Time Steps:', time_steps)
# for i in range(len(time_steps)):
#     print(time_steps[i], global_eps_mean[i])

# Plot the data
plt.figure()
plt.plot(time_steps, global_eps_mean, marker='o',label='Global Eps Mean')
plt.plot(time_steps, global_step_mean, marker='o',label='Global Step Mean')
plt.legend()
plt.xlabel('Time Steps')
plt.ylabel('Rewards')
plt.title('Reward vs Time Step')
plt.grid(True)
plt.show()