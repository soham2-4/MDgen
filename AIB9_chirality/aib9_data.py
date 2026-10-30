import os
import numpy as np
import pandas as pd

replicas = [
    "aib9_chirality_analysis_data_replica_1.csv",
    "aib9_chirality_analysis_data_replica_2.csv",
    "aib9_chirality_analysis_data_replica_3.csv",
    "aib9_chirality_analysis_data_replica_4.csv",
    "aib9_chirality_analysis_data_replica_5.csv",
]

output_folder = "synthetic_aib9_trajectories"
os.makedirs(output_folder, exist_ok=True)

traj_counter = 0

for replica_id, csv_file in enumerate(replicas):

    df = pd.read_csv(csv_file)

    time_ps = df["Time_ps"].values.astype(np.float32)
    zeta = df["Zeta_rad"].values.astype(np.float32)

    np.save(
        os.path.join(
            output_folder,
            f"time_replica_{replica_id+1}.npy"
        ),
        time_ps
    )

    np.save(
        os.path.join(
            output_folder,
            f"original_replica_{replica_id+1}.npy"
        ),
        zeta
    )

    for i in range(200):

        noise = np.random.uniform(
            0.0001,
            0.001,
            size=zeta.shape
        ) * np.random.choice(
            [-1, 1],
            size=zeta.shape
        )

        synthetic = zeta + noise

        np.save(
            os.path.join(
                output_folder,
                f"traj_{traj_counter:04d}.npy"
            ),
            synthetic.astype(np.float32)
        )

        traj_counter += 1

print(f"Generated {traj_counter} synthetic trajectories")