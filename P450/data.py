import numpy as np
import os

# Load distance data from XVG file
def load_xvg(filename):
    time, distance = [], []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith(('#', '@')):
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                time.append(float(parts[0]))
                distance.append(float(parts[1]))
    return np.array(time), np.array(distance)

# Main parameters
input_files = ["distance0.xvg", "distance1.xvg", "distance2.xvg"]
num_trajectories = 1000
noise_min, noise_max = 0.0001, 0.001
main_output_folder = "synthetic_trajectories"
os.makedirs(main_output_folder, exist_ok=True)

# Process each file
for file in input_files:
    file_base = os.path.splitext(os.path.basename(file))[0]
    time, original_distance = load_xvg(file)

    # Folder for this file's synthetic trajectories
    file_output_folder = os.path.join(main_output_folder, file_base)
    os.makedirs(file_output_folder, exist_ok=True)

    # Save time array (once per source)
    np.save(os.path.join(file_output_folder, "time.npy"), time)

    # Generate and save synthetic trajectories
    for i in range(num_trajectories):
        noise = np.random.uniform(noise_min, noise_max, size=original_distance.shape) * np.random.choice([-1, 1], size=original_distance.shape)
        synthetic = original_distance + noise
        np.save(os.path.join(file_output_folder, f"traj_{i:03d}.npy"), synthetic)

    print(f"Saved {num_trajectories} trajectories for {file} in '{file_output_folder}'")

print("All files processed successfully.")

