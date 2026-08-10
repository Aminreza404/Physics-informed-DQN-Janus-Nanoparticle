import numpy as np 
import matplotlib.pyplot as plt 
import torch 
import torch.nn as nn 
import torch.optim as optim 
import random 
from collections import deque 

plt.rcParams['text.usetex'] = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
print(f'Using device: {device}') 

KB = 1.38e-23
T_0 = 298.0          # Room temperature (K)
eta_0 = 0.001        # Water viscosity at 25°C (Pa·s)
R = 33e-9            # Janus particle radius (m)

# 1. Rotational Dynamics
tau_R = 0.22e-3     
D_R = 1.0 / tau_R   

delta_T = 6.0     
T_eff = T_0 + (5.0/12.0) * delta_T

eta_eff = 0.95 * eta_0 
D_HBM = (KB * T_eff) / (6.0 * np.pi * eta_eff * R)

# 3. Active Self-Propulsion & Peclet Number
Pe_target = 0.30     # Target Peclet number (Active vs Passive)
D_T = D_HBM / (1.0 - (Pe_target**2) / 4.0)
v_max = Pe_target * np.sqrt(D_T * D_R) 
print(f"Active Transitional diffusion coefficient {D_T}")
print(f"Hot Brownian motion diffusion coefficient {D_HBM}")

class JanusNanoParticles: 
    def __init__(self):
        self.D_T = D_T 
        self.D_R = D_R 
        self.x = 0.0; self.y = 0.0; self.theta = 0.0  
        self.v_propulsion = 0.0 
        self.max_speed = v_max 

    def step(self, dt, laser_power, action=None): 
        if action == 0:  
            self.v_propulsion = 0.0
        elif action == 1:  
            self.v_propulsion = self.max_speed * (laser_power / 81.0)
        
        dw_trans = np.sqrt(2 * self.D_T * dt) * np.random.randn(2) 
        dw_rot = np.sqrt(2 * self.D_R * dt) * np.random.randn() 

        self.theta += dw_rot
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))

        dx = self.v_propulsion * np.cos(self.theta) * dt 
        dy = self.v_propulsion * np.sin(self.theta) * dt 

        self.x += dx + dw_trans[0] 
        self.y += dy + dw_trans[1] 
        return self.x, self.y, self.theta 

class NanoPropellerEnv:
    def __init__(self, target_x=2e-6, target_y=2e-6):  
        self.particle = JanusNanoParticles() 
        # The particle constructed. Now the environment has a particle.
        self.target = np.array([target_x, target_y])  
        self.dt = 0.0001  # This is the ti,e step. It means that if the agent make decision we will move for 0.1ms.
        self.max_steps = 5000  
        self.step_count = 0 # Number of the movement.
        self.initial_dist = np.sqrt((target_x ** 2 + target_y ** 2))

    def get_state(self): # When the agent wants to make a decision we have to call this function.
        # The state of the particle contains [distance, angle].
        dist = np.sqrt((self.particle.x - self.target[0])**2 + (self.particle.y - self.target[1])**2)
        target_angle = np.arctan2(self.target[1] - self.particle.y, self.target[0] - self.particle.x)
        angle_diff = np.arctan2(np.sin(target_angle - self.particle.theta), np.cos(target_angle - self.particle.theta))

        # Here we are normalizing the distance and the angle. 
        # Because the Neural network trains with normal numbers better. 
        norm_dist = np.clip(dist / self.initial_dist, 0, 1.0) 
        norm_angle = angle_diff / np.pi 
        return np.array([norm_dist, norm_angle], dtype=np.float32)  

    # Start new game. Put the particle in the initial position. 
    def reset(self): 
        self.particle.x = 0.0 
        self.particle.y = 0.0 
        self.particle.theta = 0.0 
        self.step_count = 0
        return self.get_state()    

    def step(self, action): 
        prev_dist = np.sqrt((self.particle.x - self.target[0])**2 + (self.particle.y - self.target[1])**2)

        # Now we have to update the position by solving the langevin equation. 
        x, y, theta = self.particle.step(self.dt, laser_power=81.0, action=action) 
        # Move the particle for one step. 

        # Then new distance of the particle from gaol is computed. 
        new_dist = np.sqrt((x - self.target[0])**2 + (y - self.target[1])**2)

        # Put the condition
        if new_dist < 0.5e-6:
            reward, done = 100.0, True 
            # In RL if we have done = True then we don't have another step.
        else: 
            target_angle = np.arctan2(self.target[1] - y, self.target[0] - x)
            # The line above ask that if the particle wants to move toward the target which angle it has to choose.
            angle_diff = np.arctan2(np.sin(target_angle - theta), np.cos(target_angle - theta))
            cos_diff = np.cos(angle_diff)

            dist_reward = (prev_dist - new_dist) * 100.0 # Distance's reward

            # These are the actions we must do to navigate the particle to reach the goal.
            if action == 1:  # Laser ON
                reward = cos_diff * 0.2 - 0.005 + dist_reward
            else:  # Laser OFF
                reward = -0.002
            done = False

        self.step_count += 1 
        if self.step_count >= self.max_steps and not done: 
            done, reward = True, reward - 20.0 

        return self.get_state(), reward, done, {'distance': new_dist, 'x': x, 'y': y} 
            
class ReplayBuffer:  
    def __init__(self, capacity=50000): 
        self.buffer = deque(maxlen=capacity) 

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        # We have to save the informations in to the buffer.

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size) # Choose some information randomly
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(next_states), np.array(dones, dtype=np.float32))
    
    def __len__(self): 
        return len(self.buffer)

class DQN(nn.Module): # This is our Neural Network.
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__() 
        self.net = nn.Sequential(nn.Linear(state_dim, 128), 
                                 nn.ReLU(), 
                                 nn.Linear(128, 128), 
                                 nn.ReLU(), 
                                 nn.Linear(128, action_dim))

    def forward(self, x): return self.net(x) 

class DQNAgent: 
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99, epsilon=1.0, 
                 epsilon_decay=0.995, epsilon_min=0.01, batch_size=256, target_update=500):  
        
        self.action_dim, self.gamma, self.epsilon = action_dim, gamma, epsilon
        self.epsilon_decay, self.epsilon_min = epsilon_decay, epsilon_min
        self.batch_size, self.target_update, self.step_count = batch_size, target_update, 0
        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device) 
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() 
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, weight_decay=1e-5)  
        self.memory = ReplayBuffer(capacity=50000)  
        self.loss_fn = nn.SmoothL1Loss() 

    def select_action(self, state): 
        if np.random.random() < self.epsilon: # Choose an action randomly (Exploration)
            return np.random.randint(0, self.action_dim)
        with torch.no_grad(): 
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            return self.policy_net(state_tensor).argmax(dim=1).item() # Exploitation

    def update(self): 
        if len(self.memory) < self.batch_size: return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
        
        current_q = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_actions = self.policy_net(next_states).max(dim=1, keepdim=True)[1]
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (self.gamma * next_q * (1 - dones))
        
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        self.optimizer.step()
        
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self): 
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def train_dqn(n_episodes=500, max_steps_per_episode=5000):  
    env = NanoPropellerEnv(target_x=2e-6, target_y=2e-6) 
    agent = DQNAgent(state_dim=2, action_dim=2, lr=1e-4, epsilon_decay=0.995, batch_size=256, target_update=500)  
    
    reward_per_episode, success_history, total_successes = [], [], 0
    print("Starting Training ...")
    
    for episode in range(n_episodes): 
        state, total_reward, episode_success, step_in_episode = env.reset(), 0, 0, 0

        for _ in range(max_steps_per_episode): 
            action = agent.select_action(state) 
            next_state, reward, done, info = env.step(action) 
            agent.memory.push(state, action, reward, next_state, done) 
            
            if step_in_episode % 4 == 0 and len(agent.memory) >= agent.batch_size:
                agent.update() 
            
            total_reward += reward 
            state = next_state 
            step_in_episode += 1 

            if done:
                if info['distance'] < 0.5e-6:
                    episode_success, total_successes = 1, total_successes + 1
                break 

        agent.decay_epsilon() 
        reward_per_episode.append(total_reward)
        success_history.append(episode_success)
        
        if (episode + 1) % 10 == 0:
            recent_success_rate = (sum(success_history[-50:]) / 50.0) * 100
            avg_reward = np.mean(reward_per_episode[-50:])
            print(f"Ep {episode+1:4d} | Reward: {avg_reward:7.1f} | Eps: {agent.epsilon:.3f} | Success Rate: {recent_success_rate:5.1f}% | Total: {total_successes}") 

    return agent, env, reward_per_episode, success_history


def compare_all_strategies(env_template, trained_agent, n_trials=100):
    print("\n" + "="*70)
    print(" CRITICAL EXPERIMENT: Comparing All Strategies (100 Trials Each) ")
    print("="*70)
    
    strategies = {
        'Always OFF': lambda state: 0,
        'Always ON': lambda state: 1,
        'Random ON/OFF': lambda state: random.randint(0, 1),
    }
    
    results = {}
    for name, func in strategies.items():
        successes, final_dists, steps_list = 0, [], []
        for _ in range(n_trials):
            env = NanoPropellerEnv(target_x=env_template.target[0], target_y=env_template.target[1])
            env.max_steps = 5000
            state = env.reset()
            for step in range(env.max_steps):
                next_state, _, done, info = env.step(func(state))
                state = next_state
                if done:
                    final_dists.append(info['distance'] * 1e6)
                    steps_list.append(step + 1)
                    if info['distance'] < 0.5e-6: successes += 1
                    break
        
        results[name] = {
            'success': (successes/n_trials)*100, 
            'dist': np.mean(final_dists) if final_dists else 0, 
            'std': np.std(final_dists) if final_dists else 0
        }
        print(f" {name:15} | Success: {results[name]['success']:5.1f}% | Mean Dist: {results[name]['dist']:.2f} ± {results[name]['std']:.2f} μm")

    print(f" Testing DQN Trained Agent...")
    successes, final_dists, steps_list = 0, [], []
    trained_agent.epsilon = 0.0
    
    for _ in range(n_trials):
        env = NanoPropellerEnv(target_x=env_template.target[0], target_y=env_template.target[1])
        env.max_steps = 5000
        state = env.reset()
        for step in range(env.max_steps):
            next_state, _, done, info = env.step(trained_agent.select_action(state))
            state = next_state 
            if done:
                final_dists.append(info['distance'] * 1e6)
                steps_list.append(step + 1)
                if info['distance'] < 0.5e-6: successes += 1
                break
                
    results['DQN Trained'] = {
        'success': (successes/n_trials)*100, 
        'dist': np.mean(final_dists) if final_dists else 0, 
        'std': np.std(final_dists) if final_dists else 0
    }
    print(f" {'DQN Trained':15} | Success: {results['DQN Trained']['success']:5.1f}% | Mean Dist: {results['DQN Trained']['dist']:.2f} ± {results['DQN Trained']['std']:.2f} μm")
    
    print("\n" + "="*70)
    print(" SCIENTIFIC CONCLUSION ")
    print("="*70)
    if results['DQN Trained']['success'] > results['Always ON']['success'] * 2:
        print(" DQN is MORE THAN 2x better than Always ON.")
        print("   → PROOF: DQN learned a SMART temporal strategy, not just 'keep laser ON'.")
    elif results['DQN Trained']['success'] > results['Always ON']['success'] * 1.2:
        print(" DQN is better than Always ON, but not dramatically.")
    else:
        print(" DQN is NOT significantly better than Always ON.")
        
    return results

if __name__ == "__main__":
    print("Training the Agent") 
    agent, env, rewards, success_history = train_dqn(n_episodes=500, max_steps_per_episode=8000)
    
    model_path = 'Janus_dqn_model.pth' 
    torch.save(agent.policy_net.state_dict(), model_path) 
    print(f"\n Model saved to: {model_path}") 
    
    print("\n--- Plotting Learning Metrics ---")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    window = 50
    ax1.plot(range(window-1, len(rewards)), np.convolve(rewards, np.ones(window)/window, mode='valid'), color='blue', linewidth=2)
    ax1.set_title("Smoothed Total Reward"); ax1.set_xlabel("Episode"); ax1.set_ylabel("Reward"); ax1.grid(True)
    ax2.plot(range(window-1, len(success_history)), np.convolve(success_history, np.ones(window)/window, mode='valid') * 100, color='green', linewidth=2.5)
    ax2.set_title("Success Rate"); ax2.set_xlabel("Episode"); ax2.set_ylabel("Success Rate (%)"); ax2.set_ylim(0, 100); ax2.grid(True)
    plt.tight_layout(); plt.show()

    compare_all_strategies(env, agent, n_trials=100)
    
    print("\n All phases completed successfully! ")