import numpy as np
from mph import Client
import sys

# Define parameter names
param_names = ["loadb", "Es", "rhos", "nus", "cs", "phis", 
               "epss", "thetass", "hks", "H1", "H2", "x6"]

def init_comsol():
    """Starts COMSOL and loads the model."""
    client = Client()
    model = client.load("C:\\Users\\....\\Model1.mph") #load comsol model file
    return client, model

def get_latest_refinement_dataset(model):
    """Finds the latest adaptive mesh refinement dataset in COMSOL."""
    try:
        # Retrieve all available datasets
        dataset_names = model.datasets()

        # Filter only datasets related to Adaptive Mesh Refinement
        refinement_datasets = [ds for ds in dataset_names if "Adaptive Mesh Refinement Solutions" in ds]

        if not refinement_datasets:
            print("WARNING: No adaptive refinement datasets found! Using 'Solution 1'.")
            return "Study 1//Solution 1"

        # Sort datasets numerically to get the latest one
        refinement_datasets.sort(key=lambda ds: int(ds.split()[-1]))  # Extract last number
        latest_dataset = refinement_datasets[-1]

        #print(f"Using latest dataset: {latest_dataset}")
        return latest_dataset

    except Exception as e:
        print(f"ERROR retrieving datasets: {e}")
        return "Study 1//Solution 1"  # Fallback if datasets cannot be retrieved

def evaluate_particle(model, position):
    """Evaluates PPI using COMSOL for a given particle's parameters."""
    #sys.stdout = open("comsol_log.txt", "w")  # Redirect output to log file

    try:
        for i, name in enumerate(param_names):
            model.parameter(name, str(position[i]))

            # Solve model
        model.solve("Study 1")
        # Get latest dataset
        latest_dataset = get_latest_refinement_dataset(model)
        # Extract results
        sigma_1 = model.evaluate("maxop1(solid.sp1Gp)", dataset=latest_dataset)
        sigma_3 = model.evaluate("maxop1(solid.sp3Gp)", dataset=latest_dataset)
        e_dev = model.evaluate("maxop1(solid.edeve)", dataset=latest_dataset)
        e_plst = model.evaluate("maxop1(solid.epeGp)", dataset=latest_dataset)
        # Ensure sigma_1 and sigma_3 are scalars
        sigma_1 = sigma_1[0] if isinstance(sigma_1, (list, np.ndarray)) else sigma_1
        sigma_3 = sigma_3[0] if isinstance(sigma_3, (list, np.ndarray)) else sigma_3
        e_dev = e_dev[0] if isinstance(e_dev, (list, np.ndarray)) else e_dev
        e_plst = e_plst[0] if isinstance(e_plst, (list, np.ndarray)) else e_plst
        # Compute normal stress on the failure plane
        phi = position[5]
        cohesion = position[4]
        sigma_n = (sigma_1 + sigma_3) / 2 - ((sigma_1 - sigma_3) / 2) * np.sin(phi)
        # Compute maximum shear stress
        tau_max = (sigma_1 - sigma_3) / 2
        # Compute Objective Function (PPI, or FI in the manuscript)
        ppi = tau_max / (cohesion + sigma_n * np.tan(phi))
        #print(f"✅ Computed PPI: {ppi:.2f}, e_dev: {e_dev:.5f}, e_plst: {e_plst:.5f}, sigma_1: {sigma_1:.1f}")  # Debug output
        
        #return float(ppi), float(e_dev), float(e_plst)  # ✅ Return strain values
        return float(ppi), float(sigma_1), float(sigma_3), float(e_dev), float(e_plst)

    except Exception as e:
        return np.inf, np.inf, np.inf, np.inf, np.inf

    #finally:
     #   sys.stdout = sys.__stdout__  # Restore normal output


