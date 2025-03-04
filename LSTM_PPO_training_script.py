import gym
from gym import spaces
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import traci
from collections import deque
import torch.optim as optim
import matplotlib.pyplot as plt

NUM_OF_LANES = 12
NUM_OF_FEATURES = 5
STEP_LENGTH = 0.1
step_to_finish = []

# ---------------------------------------------------------
# 1) SUMO-like Gym Environment
# ---------------------------------------------------------
class SumoGymEnv(gym.Env):
    """
    A Gym-style environment that simulates interacting with SUMO.
    Now:
      - Action space: 8 discrete actions.
      - Observation space: shape (7, 12, 5) meaning:
          7 "historical time points"
         12 "lanes"
          5 "features" per lane
    """
    def __init__(self, sumoCmd, signal_phase_interval=30, sequence_length=7):
        super(SumoGymEnv, self).__init__()

        self.signal_phase_interval = signal_phase_interval
        self.current_step = 0
        self.sumo_cmd = sumoCmd
        self.max_steps = 103

        self.num_lanes = NUM_OF_LANES
        self.num_features = NUM_OF_FEATURES
        self.sequence_length = sequence_length
        # --- 2. Define the observation space
        # Shape: (7, 12, 5), and we allow any float values for simplicity
        obs_shape = (self.sequence_length, self.num_lanes, self.num_features)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
            dtype=np.float32
        )
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

        # --- 1. Define the action space as 8 discrete actions
        self.action_space = spaces.Discrete(len(self.main_phases))
        self.observation_history = deque(maxlen=sequence_length)
        self.reset()

    def get_queue_length(self, detector_id):
        queue_length = 0
        vehicle_ids = traci.inductionloop.getLastStepVehicleIDs(detector_id)
        for vehicle_id in vehicle_ids:
            if traci.vehicle.getSpeed(vehicle_id) < 0.1:
                queue_length += 1
        return queue_length

    def _get_current_observation(self):
        """
        Gather lane-level traffic data from both E2 and E1 detectors via TraCI.
        """
        obs = np.zeros((self.num_lanes, self.num_features), dtype=np.float32)

        for i, lane_id in enumerate(self.lane_ids):
            # E2 detector
            e2_detector_id = f"e2_detector_{lane_id}"
            obs[i, 0] = traci.lanearea.getLastStepVehicleNumber(e2_detector_id)  # Vehicle count
            obs[i, 1] = traci.lanearea.getLastStepMeanSpeed(e2_detector_id)  # Mean speed
            obs[i, 2] = traci.lanearea.getLastStepOccupancy(e2_detector_id)  # Lane occupancy

            # E1 detector
            e1_detector_id = f"e1_detector_{lane_id}"
            queue_length = self.get_queue_length(e1_detector_id)
            obs[i, 3] = queue_length  # Queue length near stop line
            obs[i, 4] = np.mean([
                traci.vehicle.getWaitingTime(vehicle_id)
                for vehicle_id in traci.inductionloop.getLastStepVehicleIDs(e1_detector_id)
            ]) if traci.inductionloop.getLastStepVehicleIDs(e1_detector_id) else 0  # Avg waiting time

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
        if initial_obs is None:  # Check if observation is valid
            print("Initial observation is None!")
            return None
        for _ in range(self.sequence_length):
            self.observation_history.append(initial_obs)

        return np.array(self.observation_history)

    def calculate_reward(self):
        """
        Calculate the reward as the average speed of all vehicles in the road network.

        Returns:
            float: The average speed of all vehicles.
        """
        # Get the list of all vehicle IDs in the simulation
        vehicle_ids = traci.vehicle.getIDList()

        # If there are no vehicles, return a reward of 0
        if len(vehicle_ids) == 0:
            return 0.0

        # Calculate the total speed of all vehicles
        total_speed = sum(traci.vehicle.getSpeed(vehicle_id) for vehicle_id in vehicle_ids)

        # Calculate the average speed
        avg_speed = total_speed / len(vehicle_ids)

        return avg_speed
    
    def step(self, action):
        """
        Applies the chosen action (switch light phases, etc.),
        then advances the simulation by 'signal_phase_interval' seconds.

        Returns:
            - next_state: shape (7, 12, 5)
            - reward: float
            - done: bool
            - info: dict
        """
        self.current_step += 1

        # Apply the main phase
        main_phase_duration, main_phase_state = self.main_phases[action]

        traci.trafficlight.setRedYellowGreenState("C", main_phase_state)
        for _ in range(main_phase_duration * int(1/STEP_LENGTH)):  # Assuming 0.1s per step
            traci.simulationStep()

        # Apply the transition phase
        transition_phase_duration, transition_phase_state = self.transition_phases[action]
        traci.trafficlight.setRedYellowGreenState("C", transition_phase_state)
        for _ in range(transition_phase_duration  * int(1/STEP_LENGTH)):  # Assuming 0.1s per step
            traci.simulationStep()
        
        # Gather observation
        current_obs = self._get_current_observation()
        self.observation_history.append(current_obs)

        # Average speed as reward
        reward = self.calculate_reward()

        # Done if we exceed 3600
        done = self.current_step >= self.max_steps or traci.simulation.getMinExpectedNumber() == 0
        if done:
            step_to_finish.append(self.current_step)
            print(step_to_finish)

        return np.array(self.observation_history), reward, done, {}

def is_traci_connected():
    try:
        # Check if the default connection exists
        traci.getConnection("default")
        return True
    except traci.TraCIException:
        return False


# ---------------------------------------------------------
# 2) Replay Memory / Experience Buffer
# ---------------------------------------------------------
class ReplayMemory:
    def __init__(self, capacity=1200):
        self.capacity = capacity
        self.buffer = []

    def add(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer.pop(0)
            self.buffer.append(transition)

    def sample(self, batch_size=32):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# ------------------------------------------------------------------
# 3) LSTM-based Actor-Critic Networks
#    We'll build "Actor" and "Critic" as separate but similar LSTM
#    networks. Alternatively, you could keep them in one network
#    that outputs both policy and value.
# ------------------------------------------------------------------
class LSTMActor(nn.Module):
    """
    The 'Actor' network: given state -> produce action probabilities.
    """
    def __init__(self, input_dim=NUM_OF_LANES*NUM_OF_FEATURES, lstm_units=64, fc_dims=[200,8], action_dim=8):
        super(LSTMActor, self).__init__()
        self.lstm = nn.LSTM(input_dim, lstm_units, batch_first=True)
        fc_layers = []
        prev_dim = lstm_units
        for dim in fc_dims:
            fc_layers.append(nn.Linear(prev_dim, dim))
            fc_layers.append(nn.ReLU())
            prev_dim = dim
        self.fc = nn.Sequential(*fc_layers)
        self.actor_head = nn.Linear(prev_dim, action_dim)

    def forward(self, x, hidden_state=None):
        # x shape: (batch_size, seq_len=7, input_dim=60)
        lstm_out, lstm_hidden = self.lstm(x, hidden_state)
        # take the last time-step
        last_timestep = lstm_out[:, -1, :]
        fc_out = self.fc(last_timestep)
        logits = self.actor_head(fc_out)
        return logits, lstm_hidden


class LSTMCritic(nn.Module):
    """
    The 'Critic' network: given state -> produce state-value V(s).
    """
    def __init__(self, input_dim=NUM_OF_LANES*NUM_OF_FEATURES, lstm_units=64, fc_dims=[200,8]):
        super(LSTMCritic, self).__init__()
        self.lstm = nn.LSTM(input_dim, lstm_units, batch_first=True)
        fc_layers = []
        prev_dim = lstm_units
        for dim in fc_dims:
            fc_layers.append(nn.Linear(prev_dim, dim))
            fc_layers.append(nn.ReLU())
            prev_dim = dim
        self.fc = nn.Sequential(*fc_layers)
        self.critic_head = nn.Linear(prev_dim, 1)

    def forward(self, x, hidden_state=None):
        lstm_out, lstm_hidden = self.lstm(x, hidden_state)
        last_timestep = lstm_out[:, -1, :]
        fc_out = self.fc(last_timestep)
        state_value = self.critic_head(fc_out)
        return state_value, lstm_hidden

# ------------------------------------------------------------------
# 4) PPO-LSTM Agent with Actor-Old / Actor-New
#     - RL target (Eq. (1))
#     - Policy Gradient Update (Eq. (2), (3))
#     - Advantage function (Eq. (4))
#     - KL divergence (Eq. (5)) for controlling distribution shift
# ------------------------------------------------------------------
class PPO_LSTM_Agent:
    def __init__(
        self,
        lstm_units=64,
        fc_dims=[200, 8],
        max_memory=1200,
        sample_size=300,
        learning_rate=0.0005,
        batch_size=32,
        ppo_epochs=100,
        gamma=0.99,
        lam=0.95,
        clip_epsilon=0.2,
    ):
        
        self.action_dim = 8
        self.batch_size = batch_size
        self.ppo_epochs = ppo_epochs
        self.sample_size = sample_size
        self.gamma = gamma        # discount
        self.lam = lam            # GAE-lambda (not fully shown, but could incorporate)
        self.clip_epsilon = clip_epsilon
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build Actor-New and Actor-Old
        self.actor_new = LSTMActor(
            input_dim=NUM_OF_LANES*NUM_OF_FEATURES,
            lstm_units=lstm_units,
            fc_dims=fc_dims,
            action_dim=self.action_dim
        )
        self.actor_old = LSTMActor(
            input_dim=NUM_OF_LANES*NUM_OF_FEATURES,
            lstm_units=lstm_units,
            fc_dims=fc_dims,
            action_dim=self.action_dim
        )
        # Copy parameters so that initially Actor-Old = Actor-New
        self.actor_old.load_state_dict(self.actor_new.state_dict())

        # Build Critic-New and Critic-Old
        self.critic_new = LSTMCritic(
            input_dim=NUM_OF_LANES*NUM_OF_FEATURES,
            lstm_units=lstm_units,
            fc_dims=fc_dims
        )
        self.critic_old = LSTMCritic(
            input_dim=NUM_OF_LANES*NUM_OF_FEATURES,
            lstm_units=lstm_units,
            fc_dims=fc_dims
        )
        self.critic_old.load_state_dict(self.critic_new.state_dict())

        # Separate optimizers (or a combined one if you prefer)
        self.actor_optimizer = optim.Adam(self.actor_new.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic_new.parameters(), lr=learning_rate)

        # Replay memory
        self.memory = ReplayMemory(capacity=max_memory)
    
    def save_policy(self, save_path):
        """
        Save the actor and critic policies to disk.

        Args:
            save_path (str): Directory path to save the policies.
        """
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Save actor and critic new policies
        actor_path = os.path.join(save_path, "actor_new.pth")
        critic_path = os.path.join(save_path, "critic_new.pth")
        
        torch.save(self.actor_new.state_dict(), actor_path)
        torch.save(self.critic_new.state_dict(), critic_path)

        print(f"Policies saved to {save_path}")

    def _preprocess_state(self, state):
        """ 
        Flatten from (7, 12, 5) → (7, 60).
        """
        if state is None:
            print("Warning: Received None state. Filling with random numbers for debugging.")
            # Generate a random state with the correct shape
            state = np.random.rand(7, NUM_OF_LANES * NUM_OF_FEATURES).astype(np.float32)
        return state.reshape(7, NUM_OF_LANES*NUM_OF_FEATURES).astype(np.float32)

    def select_action(self, state):
        """
        Use Actor-New to select an action (sampling from distribution).
        """
        # Convert shape
        s = self._preprocess_state(state)
        s_t = torch.FloatTensor(s).unsqueeze(0)  # (1,7,60)
        with torch.no_grad():
            logits, _ = self.actor_new(s_t)
        probs = torch.softmax(logits, dim=1)
        action = np.random.choice(self.action_dim, p=probs.numpy()[0]) # probability by weighted chance
        return action

    def store_transition(self, transition):
        """
        transition = (state, action, reward, next_state, done)
        """
        self.memory.add(transition)

    def _compute_returns_and_advantages(self, batch):
        """
        For each transition, we compute:
            G_t = r_t + gamma * V(s_{t+1})
        Then Advantage:
            A_t = G_t - V(s_t)
        (A simplified version of Eq. (4).)
        In a real PPO, you might use multi-step returns or GAE-lambda.
        """
        states, actions, rewards, next_states, dones = batch
        # We'll compute values using Critic-New for simplicity.
        # (Some variants do "Critic-Old," but the paper suggests we can do new.)
        with torch.no_grad():
            values_s, _      = self.critic_new(states)      # V(s_t)
            values_s_next, _ = self.critic_new(next_states) # V(s_{t+1})

        # Flatten
        values_s      = values_s.squeeze(-1)      # shape (batch_size,)
        values_s_next = values_s_next.squeeze(-1)

        # Discounted returns
        returns = rewards + self.gamma * values_s_next * (1 - dones)

        # Advantage (Eq. (4)):  A(s_t,a_t) = (returns) - V(s_t)
        advantages = returns - values_s
        return returns, advantages

    def update(self):
        """
        The main PPO update routine referencing Eqs. (2), (3), (4), (5):
            - Collect a batch from memory
            - Evaluate old & new policy (for ratio, KL, etc.)
            - Compute the clipped surrogate objective
            - Compute advantage-based returns
            - Update actor & critic
            - Finally, set Actor-Old = Actor-New, Critic-Old = Critic-New
        """
        if len(self.memory) < self.sample_size:
            return

        # 1) Sample transitions
        transitions = self.memory.sample(self.sample_size)

        # Convert transitions to tensors
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for (s, a, r, s_next, d) in transitions:
            states.append(self._preprocess_state(s))
            actions.append(a)
            rewards.append(r)
            next_states.append(self._preprocess_state(s_next))
            dones.append(float(d))

        states_t      = torch.tensor(states, dtype=torch.float32, device = self.device)      # (sample_size, 7, 60)
        actions_t     = torch.tensor(actions, dtype=torch.long, device=self.device)      # (sample_size,)
        rewards_t     = torch.tensor(rewards, dtype=torch.float32, device = self.device)     # (sample_size,)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device = self.device) # (sample_size, 7, 60)
        dones_t       = torch.tensor(dones, dtype=torch.float32, device=self.device)       # (sample_size,)

        # 2) We'll split into mini-batches for multiple PPO epochs
        num_batches = self.sample_size // self.batch_size

        # Precompute returns & advantages (Eq. (4))
        returns, advantages = self._compute_returns_and_advantages(
            (states_t, actions_t, rewards_t, next_states_t, dones_t)
        )

        for _ in range(self.ppo_epochs):

            # Shuffle indices
            indices = np.arange(self.sample_size)
            np.random.shuffle(indices)

            for i in range(num_batches):
                batch_idx = indices[i*self.batch_size : (i+1)*self.batch_size]

                b_states      = states_t[batch_idx]      # shape (B, 7, 60)
                b_actions     = actions_t[batch_idx]      # shape (B,)
                b_returns     = returns[batch_idx]        # shape (B,)
                b_advantages  = advantages[batch_idx]     # shape (B,)
                b_dones       = dones_t[batch_idx]

                # -------------------------------------------------
                # A) Compute old & new policy distribution (log probs)
                #    This implements importance sampling per Eqs. (2)(3).
                # -------------------------------------------------
                # Actor-Old
                old_logits, _ = self.actor_old(b_states)
                old_dist = torch.distributions.Categorical(logits=old_logits)
                old_log_probs = old_dist.log_prob(b_actions)

                # Actor-New
                new_logits, _ = self.actor_new(b_states)
                new_dist = torch.distributions.Categorical(logits=new_logits)
                new_log_probs = new_dist.log_prob(b_actions)

                # Ratio: r(θ) = p_θ(a|s) / p_θ_old(a|s)  [Eq. (3)]
                ratio = torch.exp(new_log_probs - old_log_probs.detach())

                # (Optional) KL divergence measure for early stopping or penalty (Eq. (5))
                # KL(p_old || p_new) = sum p_old * (log p_old - log p_new)
                with torch.no_grad():
                    kl = torch.distributions.kl.kl_divergence(old_dist, new_dist).mean()

                # Advantage
                # In practice, we might normalize advantage, e.g.:
                # b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

                # -------------------------------------------------
                # B) Clipped Surrogate Objective
                #    L^CLIP = E[ min( ratio * A, clip(ratio,1-ε,1+ε)*A ) ]
                # -------------------------------------------------
                unclipped = ratio * b_advantages
                clipped   = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * b_advantages
                policy_loss = -torch.mean(torch.min(unclipped, clipped))

                # -------------------------------------------------
                # C) Critic loss: MSE( V(s), returns )
                #    We use Critic-New for learning the value function.
                # -------------------------------------------------
                value_pred, _ = self.critic_new(b_states)
                value_pred = value_pred.squeeze(-1)
                critic_loss = nn.MSELoss()(value_pred, b_returns)

                # Combine losses. We might also add a KL penalty if desired:
                # total_loss = policy_loss + critic_loss + beta * kl
                # For simplicity, let's do standard PPO: two separate optim steps
                # (one for actor, one for critic).
                
                # 1) Update Actor
                self.actor_optimizer.zero_grad()
                policy_loss.backward(retain_graph=True)
                self.actor_optimizer.step()

                # 2) Update Critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

        # After finishing all epochs on this batch, we update:
        # Actor-Old <- Actor-New, Critic-Old <- Critic-New
        self.actor_old.load_state_dict(self.actor_new.state_dict())
        self.critic_old.load_state_dict(self.critic_new.state_dict())

    # end update()

# ------------------------------------------------------------------
# 5) Main loop that uses the LSTM-PPO agent
# ------------------------------------------------------------------
def main():
    # Example hyperparams
    NUM_EPOCHS = 700
    BATCH_SIZE = 32
    LSTM_UNITS = 64
    FC_DIMS = [200, 8]
    MAX_MEMORY_CAPACITY = 1200
    SAMPLED_TRAINING_SAMPLES = 300
    NUM_ITERATIONS_TRAINING = 13   # for demonstration
    GAMMA = 0.99
    LAM   = 0.95
    LR    = 0.0005
    CLIP_EPS = 0.2
    rolling_window = 4
    satisfactory_reward_rolling = 9.35
    satisfactory_reward_one_time = 9.7

    sumoCmd = ["sumo",  "-c", r"C:\Users\super\traffic_simulation_v2\complex.sumocfg", "--verbose", "--no-warnings", "--step-length", f"{STEP_LENGTH}"]
    #sumoCmd = ["sumo-gui", "--start", "--quit-on-end",  "-c", r"C:\Users\super\traffic_simulation_v2\complex.sumocfg", "--step-length", f"{STEP_LENGTH}"]
    # Suppose you define your environment with 8 discrete actions,
    # obs shape = (7,12,5). For brevity, assume it's named 'env' here.
    # from your_sumo_env_module import SumoGymEnv
    env = SumoGymEnv(sumoCmd = sumoCmd)

    agent = PPO_LSTM_Agent(
        lstm_units=LSTM_UNITS,
        fc_dims=FC_DIMS,
        max_memory=MAX_MEMORY_CAPACITY,
        sample_size=SAMPLED_TRAINING_SAMPLES,
        learning_rate=LR,
        batch_size=BATCH_SIZE,
        ppo_epochs=NUM_ITERATIONS_TRAINING,
        gamma=GAMMA,
        lam=LAM,
        clip_epsilon=CLIP_EPS
    )

    # Lists to track performance
    average_rewards_per_step = []  # Total reward per episode
    rolling_avg_rewards = []  # Rolling average for smoother visualization

    for epoch in range(NUM_EPOCHS):
        # Run one episode per epoch (or more, if you prefer)
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        try:
            while not done:
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                agent.store_transition((state, action, reward, next_state, done))
                state = next_state
                total_reward += reward
                steps += 1

            # Record episode reward
            avg_reward_per_step = total_reward / steps if steps > 0 else 0
            average_rewards_per_step.append(avg_reward_per_step)

            # Compute rolling average reward
            if len(average_rewards_per_step) >= rolling_window:
                rolling_avg = sum(average_rewards_per_step[-rolling_window:]) / rolling_window
            else:
                rolling_avg = sum(average_rewards_per_step) / len(average_rewards_per_step)
            rolling_avg_rewards.append(rolling_avg)
            print(rolling_avg_rewards)

            # Check for early stopping
            if rolling_avg >= satisfactory_reward_rolling or avg_reward_per_step >= satisfactory_reward_one_time:
                print(f"Early stopping at epoch {epoch + 1}. Rolling avg reward: {rolling_avg:.2f}. Average speed last episode: {avg_reward_per_step:.2f}")
                break

            # After each episode, update PPO
            agent.update()

            print(f"[Epoch {epoch+1}/{NUM_EPOCHS}] Update done.")
        except Exception as e:
            print(e)
            continue
    
    save_path = r"C:\Users\super\traffic_simulation_v2"
    agent.save_policy(save_path)

    # Plot training performance
    plt.figure(figsize=(10, 6))
    plt.plot(average_rewards_per_step, label="Episode Reward", alpha=0.7)
    plt.plot(rolling_avg_rewards, label="Rolling Average Reward (10 episodes)", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Reward (Average Speed)")
    plt.title("PPO-LSTM Training Performance")
    plt.legend()
    plt.grid()
    plt.show()

    print("Training completed.")

if __name__ == "__main__":
    main()
