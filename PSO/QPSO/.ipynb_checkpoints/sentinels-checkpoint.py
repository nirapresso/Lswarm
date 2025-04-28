import numpy as np
import os
import time
import sqlite3
from scipy.spatial.distance import mahalanobis, cosine
from scipy.stats import levy, cauchy
from scipy.stats.qmc import LatinHypercube
import sys
import random
from the_matrix import evaluate_particle 
from sentinel_config import num_particles, num_iterations, w, c1, c2, alpha
from exos import print_total_computation_time, print_ppi_chart, print_iteration_summary, print_itrsum_divrs

# Define parameter names, bounds, and units
param_names = ["loadb", "Es", "rhos", "nus", "cs", "phis",
               "epss", "thetass", "hks", "H1", "H2", "x6"]
param_units = ["Pa", "Pa", "kg/m³", "", "Pa", "rad", "", "", "m/s", "m", "m", "m"]
param_bounds = np.array([
    (0, 50000), (10e6, 200e6), (1000, 1800), (0.25, 0.45),
    (0, 50e3), (25 * np.pi/180, 40 * np.pi/180), (0.2, 0.7),
    (0.3, 0.8), (1e-9, 1e-1), (10, 30), (1, 5), (3, 12)
])

log_scale_indices = [i for i, (low, high) in enumerate(param_bounds) if low > 0 and high / low > 1e3]
# ==============================================
# Database Setup
# ==============================================
from sentinel_config import DB_FILE

# Create the 'runs' table if missing
# Create the 'particles' table if missing

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    # runs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_name TEXT UNIQUE NOT NULL,
            algorithm TEXT NOT NULL,
            alpha REAL NOT NULL,
            num_particles INTEGER NOT NULL,
            num_iterations INTEGER NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # particles table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS particles (
            particle_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            iteration INTEGER NOT NULL,
            loadb REAL, Es REAL, rhos REAL, nus REAL, cs REAL, phis REAL,
            epss REAL, thetass REAL, hks REAL, H1 REAL, H2 REAL, x6 REAL,
            ppi REAL, sigma_1 REAL, sigma_3 REAL, e_dev REAL, e_plst REAL,
            FOREIGN KEY(run_id) REFERENCES runs(run_id)
        )
    """)

    conn.commit()
    conn.close()

def create_or_get_run(run_name="PSO_Run_X", algorithm="PSO"):
    """Inserts (or retrieves) a row from 'runs' table and returns run_id."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO runs (run_name, algorithm, alpha, num_particles, num_iterations)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(run_name) DO NOTHING
    """, (run_name, algorithm, alpha, num_particles, num_iterations))
    conn.commit()

    # If newly inserted, lastrowid != 0, else we fetch existing row
    run_id = cursor.lastrowid
    if run_id == 0:  # conflict, row existed
        cursor.execute("SELECT run_id FROM runs WHERE run_name = ?", (run_name,))
        row = cursor.fetchone()
        if row:
            run_id = row[0]
        else:
            raise ValueError("Failed to retrieve existing run_id.")

    conn.close()
    return run_id

def get_last_iteration(run_id):
    """Returns the max iteration stored for this run_id, or -1 if none."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(iteration) FROM particles WHERE run_id=?", (run_id,))
    val = cursor.fetchone()[0]
    conn.close()
    return val if val is not None else -1

def load_iteration(run_id, iteration):
    """Loads particle array and their ppi from DB for the specified iteration."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT loadb, Es, rhos, nus, cs, phis, epss, thetass, hks, H1, H2, x6,
               ppi, sigma_1, sigma_3, e_dev, e_plst
        FROM particles
        WHERE run_id=? AND iteration=?
        ORDER BY particle_id ASC
    """, (run_id, iteration))
    rows = cursor.fetchall()
    conn.close()
    if not rows:
        return None

    # Build arrays
    # each row: [loadb, Es, rhos, nus, cs, phis, epss, thetass, hks, H1, H2, x6, ppi, sigma_1, sigma_3, e_dev, e_plst]
    # we only want the first 12 for 'particles' if we want to reconstruct them.
    particles = np.array([row[:12] for row in rows])
    # we can also parse ppi, etc. if needed
    ppi_vals = [row[12] for row in rows]
    return particles, ppi_vals, rows

def save_iteration(run_id, iteration, particles, iteration_data):
    """Saves all particle data for this iteration: positions + ppi, sigma_1, etc."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    for i, particle in enumerate(particles):
        score, s1, s3, dev, plst = iteration_data[i]  # [score, sigma_1, sigma_3, e_dev, e_plst]
        cursor.execute("""
            INSERT INTO particles (run_id, iteration,
                loadb, Es, rhos, nus, cs, phis, epss, thetass, hks, H1, H2, x6,
                ppi, sigma_1, sigma_3, e_dev, e_plst)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id, iteration, *particle,
            score, s1, s3, dev, plst
        ))
    conn.commit()
    conn.close()

def get_best_ppi_from_db(run_id):
    """Fetches the best (maximum) PPI per iteration from the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Ensure only one (max) PPI value is returned per iteration
    cursor.execute("""
        SELECT iteration, MAX(ppi) FROM particles
        WHERE run_id = ?
        GROUP BY iteration
        ORDER BY iteration ASC
    """, (run_id,))
    
    best_ppi = [row[1] for row in cursor.fetchall()]
    
    conn.close()
    return np.tanh(best_ppi)

def compute_lambda(param_bounds):
    """
    Compute lambda scaling factors for each parameter based on their range.
    Ensures velocity updates respect parameter scale.
    """
    param_ranges = param_bounds[:, 1] - param_bounds[:, 0]  # Normal range differences

    # Identify parameters that need log scaling
    log_scale_indices = [i for i, (low, high) in enumerate(param_bounds) if low > 0 and high / low > 1e3]

    # Replace linear range difference with log difference for log-scaled parameters
    for idx in log_scale_indices:
        log_min = np.log10(max(param_bounds[idx, 0], 1e-12))
        log_max = np.log10(param_bounds[idx, 1])
        param_ranges[idx] = log_max - log_min  # Log-space range difference

    # Compute lambda scaling factors
    lambda_factors = param_ranges / np.sum(param_ranges)
    return lambda_factors

def sample_lhs_with_logscale(num_samples, param_bounds):
    """
    Generate LHS samples and automatically apply log-scaling for parameters with extreme ranges.
    Avoids divide-by-zero errors when detecting log-scaled parameters.
    """
    lhs = LatinHypercube(d=len(param_bounds))
    samples = lhs.random(num_samples)  # Generate LHS in [0,1] range

    # Identify parameters that need log scaling (avoiding division by zero)
    log_scale_indices = [i for i, (low, high) in enumerate(param_bounds) if low > 0 and high / low > 1e3]

    # Scale normal parameters linearly
    scaled_samples = samples * (param_bounds[:, 1] - param_bounds[:, 0]) + param_bounds[:, 0]

    # Apply log scaling to detected parameters
    for idx in log_scale_indices:
        log_min = np.log10(max(param_bounds[idx, 0], 1e-12))  # Ensure log10(0) is avoided
        log_max = np.log10(param_bounds[idx, 1])
        scaled_samples[:, idx] = 10 ** (samples[:, idx] * (log_max - log_min) + log_min)

    return scaled_samples

def denormalize_particle(particle, param_bounds):
    """
    Convert a single particle from scaled values back to real-world values.
    Handles log-scaled parameters correctly.
    """
    real_particle = np.copy(particle)

    for idx in log_scale_indices:
        real_particle[idx] = 10 ** real_particle[idx]  # Convert log-space back to real values

    return real_particle


def run_pso(model, distance_metric="vector", db_file=DB_FILE,distribution="log-uniform"):

    # 1) Ensure DB schema
    init_db()
    run_id = create_or_get_run(run_name=DB_FILE, algorithm="QPSO")

    # Check last iteration from DB
    last_iter = get_last_iteration(run_id)
    if last_iter < 0:
        print("No DB iteration found -> fresh start")
        # lhs = LatinHypercube(d=len(param_names))
        # particles = lhs.random(num_particles) * (param_bounds[:,1] - param_bounds[:,0]) + param_bounds[:,0]
        particles = sample_lhs_with_logscale(num_particles, param_bounds)
        velocities = np.random.uniform(-1e-16, 1e-16, (num_particles, len(param_names)))
        pbest_positions = np.copy(particles)
        pbest_scores = np.array([evaluate_particle(model, p)[0] for p in particles])  
    else:
        print(f"DB indicates last iteration = {last_iter}, resuming...")
        # load last iteration's particles. If we want velocities, we'd store them, but let's keep minimal.
        loaded = load_iteration(run_id, last_iter)
        if not loaded:
            print("No data found in DB for iteration, resetting.")
            lhs = LatinHypercube(d=len(param_names))
            particles = lhs.random(num_particles) * (param_bounds[:,1] - param_bounds[:,0]) + param_bounds[:,0]
            velocities = np.random.uniform(-1e-16, 1e-16, (num_particles, len(param_names)))
            pbest_positions = np.copy(particles)
            pbest_scores = np.array([evaluate_particle(model, p)[0] for p in particles])
            last_iter = -1
        else:
            # we have partial data, minimal approach: we just restore positions
            # user might want to store velocities if needed for exact resume.
            particles, ppis, _ = loaded
            velocities = np.random.uniform(-1e-16, 1e-16,(num_particles,len(param_names)))
            # build pbest from these positions
            pbest_positions = np.copy(particles)
            pbest_scores = np.array(ppis)
    
    # Now compute or restore the global best from personal bests
    valid_indices = np.where(np.isfinite(pbest_scores))[0]
    if len(valid_indices) > 0:
        gbest_index = valid_indices[np.argmax(pbest_scores[valid_indices])]
        gbest_position = pbest_positions[gbest_index].copy()
        gbest_score = pbest_scores[gbest_index]
    else:
        print("⚠️ WARNING: No valid PPI values found. Defaulting global best to zeros.")
        gbest_position = np.zeros_like(pbest_positions[0])
        gbest_score = -np.inf
    
    lambda_factors = compute_lambda(param_bounds)  # Compute λ scaling factors
    mbest = np.mean(pbest_positions, axis=0)
    
    start_time = time.time()
    # We'll do the same iteration approach from original code
    # but offset by last_iter+1 if we want to resume seamlessly.

    # e.g. if last_iter=5, next iteration starts from 6
    iteration_start = last_iter + 1
    iteration_end = iteration_start + num_iterations  # original code had num_iterations as total, you might want to adjust.

    for iteration in range(iteration_start, iteration_end):
        iter_particle_data = []
        iter_ppi = []
        iter_sigma_1, iter_sigma_3, iter_e_dev, iter_e_plst = [], [], [], []

        for i in range(num_particles):
            r1, r2 = np.random.rand(), np.random.rand()
            u = np.random.uniform(0, 1, len(param_bounds))
            sign = np.where(np.random.rand(len(param_bounds)) > 0.5, 1, -1)
            # do the distance-based velocity update as original code
            from_sent = particles[i]
            #from_pbest = pbest_positions[i]
            from_gbest = gbest_position

            #Original:
            pbest_diff = np.abs(pbest_positions[i] - mbest)            
            #pbest_diff = from_pbest - from_sent
            #gbest_diff = from_gbest - from_sent
            
            if distribution == "log-uniform":
                quantum_step = sign * alpha * pbest_diff * np.log(1 / u)
            elif distribution == "gaussian":
                quantum_step = sign * alpha * pbest_diff * np.random.normal(0, 1, len(param_bounds))
            elif distribution == "cauchy":
                quantum_step = sign * alpha * pbest_diff * cauchy.rvs(size=len(param_bounds))
            elif distribution == "levy":
                quantum_step = sign * alpha * pbest_diff * levy.rvs(size=len(param_bounds))
            else:
                raise ValueError("Invalid distribution. Choose 'log-uniform', 'gaussian', 'cauchy', or 'levy'.")

            particles[i] = mbest + quantum_step
                       
            particles[i] = np.clip(particles[i], param_bounds[:,0], param_bounds[:,1])

            # Clip the ones with log-scaled parameters
            for idx in log_scale_indices:
                particles[i, idx] = 10 ** np.clip(np.log10(particles[i, idx]), 
                                                  np.log10(param_bounds[idx, 0]), 
                                                  np.log10(param_bounds[idx, 1]))
            
            real_particle = denormalize_particle(particles[i], param_bounds)
            
            #new_score, sigma_1, sigma_3, e_dev, e_plst = evaluate_particle(model, particles[i])
            new_score, sigma_1, sigma_3, e_dev, e_plst = evaluate_particle(model, real_particle)
            #new_score= np.tanh(new_score)
            
            if np.isinf(new_score):
                print(f"⚠️ Particle {i} got inf PPI, resetting with perturbation")
                perturbation = np.random.uniform(-0.05, 0.05, size=len(param_names)) * (param_bounds[:, 1] - param_bounds[:, 0])
                particles[i] = np.clip(particles[i] + perturbation, param_bounds[:, 0], param_bounds[:, 1])
                new_score, sigma_1, sigma_3, e_dev, e_plst = evaluate_particle(model, particles[i])
                print(f"🔄 New score after reset: {new_score}")
            
            if not np.isinf(new_score):
                if new_score > pbest_scores[i]:
                    pbest_scores[i] = new_score
                    pbest_positions[i] = particles[i].copy()
                if new_score > gbest_score:
                    gbest_score = new_score
                    gbest_position = particles[i].copy()

            iter_particle_data.append(list(particles[i]) + [new_score, sigma_1, sigma_3, e_dev, e_plst])
            iter_ppi.append(new_score)
            iter_sigma_1.append(sigma_1)
            iter_sigma_3.append(sigma_3)
            iter_e_dev.append(e_dev)
            iter_e_plst.append(e_plst)

        # print_iteration_summary(iteration+1, num_iterations, gbest_score, particles, gbest_position)
        print_itrsum_divrs(iteration+1, num_iterations, gbest_score, particles, gbest_position)
        #if (iteration+1) % 3 == 0:
        #    print_ppi_chart(get_best_ppi_from_db(run_id))   #now return tanh version

        # also save to DB now
        iteration_data = []
        for i in range(num_particles):
            row = iter_particle_data[i]
            # row = [pos0, pos1, ..., pos11, score, s1, s3, e_dev, e_plst]
            score = row[len(param_names)]
            s1 = row[len(param_names)+1]
            s3 = row[len(param_names)+2]
            dev= row[len(param_names)+3]
            plst=row[len(param_names)+4]
            iteration_data.append([score, s1, s3, dev, plst])
        save_iteration(run_id, iteration, particles, iteration_data)

    end_time = time.time()
    print_total_computation_time(start_time, end_time)

    return gbest_position, gbest_score
