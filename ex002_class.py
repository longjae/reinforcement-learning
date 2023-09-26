import numpy as np
import gymnasium as gym

# 글로벌 변수를 안써서 코딩해보자

def value_iteration(env):
    num_iteration = 2000
    threshold = 1e-20
    gamma = 0.99
    
    value_table = np.zeros(env.observation_space.n) # state list의 개수
    
    for i in range(num_iteration):
        updated_value_table = np.copy(value_table) # 이전 테이블과 비교해서 차이가 없으면 종료하기 위함
        
        for s in range(env.observation_space.n):
            Q_values = np.zeros(env.action_space.n)
            for a in range(env.action_space.n):
                sum_value = 0
                for prob, s_, r, _ in env.P[s][a]:
                    sum_value += prob * (r + gamma * updated_value_table[s_])
                    Q_values[a] = sum_value
            value_table[s] = max(Q_values)
            
        # np.fabs = absolute value
        if np.sum(np.fabs(updated_value_table - value_table)) <= threshold:
            break
        
    return value_table

def extract_policy(env, value_table):
    gamma = 0.99
    policy = np.zeros(env.observation_space.n)
    
    for s in range(env.observation_space.n):
        Q_values = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            sum_value = 0
            for prob, s_, r, _ in env.P[s][a]:
                sum_value += prob * (r + gamma * value_table[s_])
                Q_values[a] = sum_value
        policy[s] = np.argmax(np.array(Q_values))
    
    return policy

def evaluate_policy(env, policy):
    num_episodes = 1000
    num_timesteps = 1000
    total_reward = 0
    total_timestep = 0
    
    for i in range(num_episodes):
        state, _ = env.reset()
        
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
        
    print(f"number of successful episodes: {total_reward} / {num_episodes}")
    print(f"average number of timesteps per episodes: {total_timestep/num_episodes}")
    
    

if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", render_mode="ansi", is_slippery=False)
    
    optimal_value_function = value_iteration(env)
    optimal_policy = extract_policy(env, optimal_value_function)
    print(optimal_policy) # 0:left, 1: down, 2: right, 3: up
    
    evaluate_policy(env, None) # 랜덤으로 샘플링하는 경우
    evaluate_policy(env, optimal_policy)