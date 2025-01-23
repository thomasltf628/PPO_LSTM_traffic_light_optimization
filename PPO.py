# AA00E7RFUH
import gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import deque
from torch.distributions import Categorical
import traci

df = pd.DataFrame({'Action actually taken': [], 'Action': []})
step_to_finish = []

class TrafficEnv(gym.Env):
    def __init__(self, sumo_cmd, sequence_length=7):
        super().__init__()
        self.sumo_cmd = sumo_cmd
        self.sequence_length = sequence_length
        self.num_lanes = 12
        self.num_features = 5
        self.max_steps = 1000
        self.lane_ids = ["NC_0", "NC_1", "NC_2", "EC_0", "EC_1", "EC_2", "SC_0", "SC_1", "SC_2", "WC_0", "WC_1", "WC_2"]

        # Define the main and transition phases from the tlLogic
        self.main_phases = [
            (31, "rGGrrgrGGrrg"),
            (31, "rrgrGGrrgrGG"),
            (31, "GrgrrgGrgrrg"),
            (31, "rrgGrgrrgGrg"),
            (31, "rrgrrgGGGrrg"),
            (31, "GGGrrgrrgrrg"),
            (31, "rrgGGGrrgrrg"),
            (31, "rrgrrgrrgGGG"),
        ]
        self.transition_phases = [
            (4, "ryyrryryyrry"),
            (4, "rryryyrryryy"),
            (4, "yryrryyryrry"),
            (4, "rryyryrryyry"),
            (4, "rryrryyyyrry"),
            (4, "yyyrryrryrry"),
            (4, "rryyyyrryrry"),
            (4, "rryrryrryyyy"),
        ]

        # Observation and action spaces
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(sequence_length, self.num_lanes, self.num_features),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(len(self.main_phases))  # 8 main actions

        self.current_step = 0
        self.observation_history = deque(maxlen=sequence_length)
        self.reset()

    def _get_current_observation(self):
        """
        Gather lane-level traffic data from SUMO via TraCI.
        """
        obs = np.zeros((self.num_lanes, self.num_features), dtype=np.float32)
        for i, lane_id in enumerate(self.lane_ids):
            e2_detector_id = f"e2_detector_{lane_id}"
            obs[i, 0] = traci.lanearea.getLastStepVehicleNumber(e2_detector_id)  # Vehicle count
            obs[i, 1] = traci.lanearea.getLastStepMeanSpeed(e2_detector_id)  # Mean speed
            obs[i, 2] = traci.lanearea.getLastStepHaltingNumber(e2_detector_id)  # Queue length
            obs[i, 3] = traci.lanearea.getLastStepOccupancy(e2_detector_id)  # Occupancy
            obs[i, 4] = np.mean([
                traci.vehicle.getWaitingTime(vehicle_id)
                for vehicle_id in traci.lanearea.getLastStepVehicleIDs(e2_detector_id)
            ]) if traci.lanearea.getLastStepVehicleIDs(e2_detector_id) else 0  # Avg waiting time

        return obs

    def reset(self):
        """
        Reset the SUMO simulation and pre-fill the observation history.
        """
        if is_traci_connected():
            print("Closing existing connection...")
            traci.close()
        try:
            traci.start(self.sumo_cmd)
        except Exception as e:
            print(f"Error processing: {e}")
            return None 
        self.current_step = 0
        self.observation_history.clear()

        # Pre-fill the observation history
        initial_obs = self._get_current_observation()
        for _ in range(self.sequence_length):
            self.observation_history.append(initial_obs)

        return np.array(self.observation_history)

    def step(self, action):
        """
        Apply an action (main phase), simulate the environment, and return the results.
        """
        self.current_step += 1

        # Apply the main phase
        main_phase_duration, main_phase_state = self.main_phases[action]
    
        traci.trafficlight.setRedYellowGreenState("C", main_phase_state)
        for _ in range(main_phase_duration * 10):  # Assuming 0.1s per step
            traci.simulationStep()

        # Apply the transition phase
        transition_phase_duration, transition_phase_state = self.transition_phases[action]
        traci.trafficlight.setRedYellowGreenState("C", transition_phase_state)
        for _ in range(transition_phase_duration * 10):  # Assuming 0.1s per step
            traci.simulationStep()

        # Gather observation
        current_obs = self._get_current_observation()
        self.observation_history.append(current_obs)

        # Compute reward (negative average waiting time)
        avg_speed = np.mean([
            traci.vehicle.getSpeed(vehicle_id)
            for vehicle_id in traci.vehicle.getIDList()
        ]) if traci.vehicle.getIDList() else 0
        queue_length = sum(traci.lanearea.getLastStepHaltingNumber(f"e2_detector_{lane_id}") for lane_id in self.lane_ids)
        #reward = avg_speed - 0.005*queue_length # Max speed PLUS QUEUE LENGTH
        reward = 0

        print(f"Step {self.current_step}, reward: {reward}")
        
        # Check if done
        done = self.current_step >= self.max_steps or traci.simulation.getMinExpectedNumber() == 0
        
        

        return np.array(self.observation_history), reward, done, {}
    
    def close(self):
        """
        Close the SUMO simulation.
        """
        traci.close()

class ModifiedLSTMPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=12 * 5, hidden_size=64, batch_first=True)
        self.fc1 = nn.Linear(64, 200)
        self.fc2 = nn.Linear(200, 8)
        self.relu = nn.ReLU()

    def forward(self, x, hidden=None):
        batch_size, seq_len, rows, cols = x.size()
        x = x.view(batch_size, seq_len, -1)  # Flatten spatial dimensions
        if hidden is None:
            lstm_out, hidden = self.lstm(x)
        else:
            lstm_out, hidden = self.lstm(x, hidden)
        lstm_out = lstm_out[:, -1]
        x = self.relu(self.fc1(lstm_out))
        action_logits = self.fc2(x)
        return action_logits, hidden

def is_traci_connected():
    try:
        # Check if the default connection exists
        traci.getConnection("default")
        return True
    except traci.TraCIException:
        return False

def train_loop():
    #sumoCmd = ["sumo-gui", "--start", "--quit-on-end", "-c", r"C:\Users\super\traffic_simulation_v2\complex.sumocfg", "--no-warnings"]
    sumoCmd = ["sumo", "-c", r"C:\Users\super\traffic_simulation_v2\complex.sumocfg", "--verbose", "--no-warnings"]
    env = TrafficEnv(sumo_cmd = sumoCmd)
    policy = ModifiedLSTMPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.0005)
    
    gamma = 0.99
    lam = 0.95
    ppo_clip = 0.2
    epochs = 700
    batch_size = 32

    for episode in range(20):
        obs = env.reset()
        if obs is None:
            continue
        hidden = None

        log_probs = []
        states = []
        actions = []
        rewards = []

        while True:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action_logits, hidden = policy(obs_tensor, hidden)
            action_probs = torch.softmax(action_logits, dim=-1)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            log_probs.append(log_prob)
            actions.append(action)
            states.append(obs_tensor)

            next_obs, reward, done, _, current_step = env.step(action.item())            
            rewards.append(reward)
            
            if done:
                step_required = current_step
                break

            obs = next_obs

        discounted_rewards = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            discounted_rewards.insert(0, G)
        discounted_rewards = torch.FloatTensor(discounted_rewards)

        advantages = discounted_rewards  # Placeholder, improve with value network if needed

        for _ in range(epochs):
            for i in range(0, len(states), batch_size):
                batch_states = torch.cat(states[i:i + batch_size])
                batch_actions = torch.cat(actions[i:i + batch_size]).unsqueeze(1)
                batch_log_probs = torch.cat(log_probs[i:i + batch_size]).detach()
                batch_advantages = advantages[i:i + batch_size]

                action_logits, _ = policy(batch_states)
                action_probs = torch.softmax(action_logits, dim=-1)
                dist = Categorical(action_probs)

                new_log_probs = dist.log_prob(batch_actions.squeeze(-1))
                ratio = (new_log_probs - batch_log_probs).exp()
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                optimizer.zero_grad()
                policy_loss.backward()
                optimizer.step()

        print(f"Episode {episode}, Step required: {step_required}")
        step_to_finish.append(step_required)
        print(step_to_finish)

if __name__ == "__main__":
    train_loop()

