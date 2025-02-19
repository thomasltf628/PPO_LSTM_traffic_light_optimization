import os
import traci
import sumolib

# Paths to your SUMO files
SUMO_BINARY = "sumo"  # Use "sumo-gui" for GUI simulation
SUMO_CONFIG = r"C:\Users\super\traffic_simulation_v2\complex.sumocfg"  # Path to your SUMO config file

def calculate_reward():
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

def run_simulation():
    """Run a basic SUMO simulation."""
    # Start SUMO with the given config
    traci.start([SUMO_BINARY, "-c", SUMO_CONFIG])

    steps = 0
    total_reward = 0

    try:
        while steps < 3600 and traci.simulation.getMinExpectedNumber() > 0:  # Run for 1000 simulation steps
            traci.simulationStep()  # Advance the simulation
            print(f"Step {steps}:")
            reward = calculate_reward()
            total_reward += reward
            steps += 1
        avg_reward_per_step = total_reward / steps if steps > 0 else 0
        print(avg_reward_per_step)
    except Exception as e:
        print(f"Error during simulation: {e}")

    finally:
        # Close the simulation
        traci.close()
        print("Simulation finished.")

if __name__ == "__main__":
    run_simulation()
