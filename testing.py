import os
import traci
import sumolib

# Paths to your SUMO files
SUMO_BINARY = "sumo"  # Use "sumo-gui" for GUI simulation
SUMO_CONFIG = r"C:\Users\super\traffic_simulation_v2\complex.sumocfg"  # Path to your SUMO config file

def run_simulation():
    """Run a basic SUMO simulation."""
    # Start SUMO with the given config
    traci.start([SUMO_BINARY, "-c", SUMO_CONFIG])

    step = 0
    try:
        while step < 1000:  # Run for 1000 simulation steps
            traci.simulationStep()  # Advance the simulation
            print(f"Step {step}:")
            lane_id = "NC_0"
            e1_detector_id = f"e1_detector_{lane_id}"
            # Example: Retrieve vehicle information
            vehicles = traci.inductionloop.getLastStepVehicleIDs(e1_detector_id)
            print(f"  Number of vehicles: {len(vehicles)}")
            for veh_id in vehicles:
                speed = traci.vehicle.getSpeed(veh_id)
                position = traci.vehicle.getPosition(veh_id)
                print(f"    Vehicle {veh_id} -> Speed: {speed:.2f}, Position: {position}")
            
            step += 1

    except Exception as e:
        print(f"Error during simulation: {e}")

    finally:
        # Close the simulation
        traci.close()
        print("Simulation finished.")

if __name__ == "__main__":
    run_simulation()
