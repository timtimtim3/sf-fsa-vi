import gym
import time
import envs  # Ensure envs/__init__.py is executed
from fsa.tasks_specification import load_fsa
from types import SimpleNamespace

cfg = SimpleNamespace(
    fsa_name="task1",
    env_name="OfficeAreas-v0",
)

# # Initialize the environment using gym.make
# env_name = "Office-v0"  # Using the same naming convention
# env = gym.make(env_name, add_obj_to_start=False)

# Initialize the environment using gym.make
env_name = cfg.env_name  # Using the same naming convention
env = gym.make(env_name, add_obj_to_start=False)

# Reset the environment to get the initial state
state = env.reset()
print(f"Initial state: {state}")

# Number of steps to simulate
num_steps = 50

print(f"env.coords_to_state: {env.coords_to_state}")
print(f"env.object_ids: {env.object_ids}")
print(f"env.exit_states: {env.exit_states}")

# exit_states_idxs = list(map(lambda x: env.coords_to_state[env.exit_states[x]], range(len(env.exit_states))))
# print(f"exit_states_idxs: {exit_states_idxs}")

# Create the FSA env wrapper, to evaluate the FSA
fsa, T = load_fsa('-'.join([env_name, cfg.fsa_name]), env) # Load FSA


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
