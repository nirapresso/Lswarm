import os
from the_matrix import init_comsol
from sentinels import run_pso
from sentinel_config import DB_FILE, metric, dist


if __name__ == "__main__":
    client, model = init_comsol()
    
    best_position, best_ppi = run_pso(model, distance_metric=metric, db_file=DB_FILE,distribution=dist)

client.clear()  # Ensure COMSOL session is cleared
#client.disconnect()  # Fully disconnect from COMSOL
