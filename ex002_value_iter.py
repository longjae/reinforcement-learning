import numpy as np
import gymnasium as gym

env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=True)

def value_iteration(env):
    num_iterations = 1000
    threshold = 1e-20
    gamma = 1.0
    
    value_table = np.zeros(env.observation_space.n)
    
    for i in range(num_iterations):
        updated_value_table = np.copy(value_table)
        
        for s in range(env.observation_space.n):
            Q_values = [sum([prob*(r + gamma * updated_value_table[s_])
                             for prob, s_, r, _ in env.P[s][a]])
                                for a in range(env.action_space.n)]
            value_table[s] = max(Q_values)
            
        if np.sum(np.fabs(updated_value_table - value_table)) <= threshold:
            break
        
    return value_table

def extract_policy(value_table):
    gamma = 1.0
    policy = np.zeros(env.observation_space.n)
    for s in range(env.observation_space.n):
        Q_values = [sum([prob*(r + gamma * value_table[s_])
                         for prob, s_, r, _ in env.P[s][a]])
                            for a in range(env.action_space.n)]
        policy[s] = np.argmax(np.array(Q_values))
    
    return policy

optimal_value_function = value_iteration(env)
optimal_policy = extract_policy(optimal_value_function)
print(optimal_policy)

def evaluate_policy(policy):
    num_episodes = 1000
    num_timesteps = 1000
    total_reward = 0
    total_timestep = 0
    
    for i in range(num_timesteps):
        state = env.reset()
        
        for t in range(num_timesteps):
            if policy is None:
                action = env.action_space.sample()
            else:
                action = policy[state]
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        total_timestep += t
        
    print(f"Number of successful episodes: {total_reward} / {num_episodes}")
    print(f"Average number of timesteps per episode {total_timestep/num_episodes}")
    
optimal_value_function = value_iteration(env)
optimal_policy = extract_policy(optimal_value_function)
evaluate_policy(None)
evaluate_policy(optimal_policy)