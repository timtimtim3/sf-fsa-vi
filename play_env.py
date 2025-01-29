import gym
import time
import envs  # Ensure envs/__init__.py is executed


# Initialize the environment using gym.make
env_name = "Office-v0"  # Using the same naming convention
env = gym.make(env_name, add_obj_to_start=False)

# # Initialize the environment using gym.make
# env_name = "OfficeAreas-v0"  # Using the same naming convention
# env = gym.make(env_name, add_obj_to_start=False)

# Reset the environment to get the initial state
state = env.reset()
print(f"Initial state: {state}")

# Number of steps to simulate
num_steps = 50

for step in range(num_steps):
    env.render()  # Render the environment

    # Take a random action
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)

    print(f"Step {step}: Action {action}, State {next_state}, Reward {reward}")

    if done:
        print("Agent reached an exit state!")
        break  # End if the task is completed

    time.sleep(0.3)  # Add delay for better visualization

env.close()  # Close the environment when done
