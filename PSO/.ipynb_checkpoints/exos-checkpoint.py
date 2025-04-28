import numpy as np
import pandas as pd
import sys

# Define parameter names, bounds, and units
param_names = ["loadb", "Es", "rhos", "nus", "cs", "phis",
               "epss", "thetass", "hks", "H1", "H2", "x6"]
param_units = ["Pa", "Pa", "kg/m³", "", "Pa", "rad", "", "", "m/s", "m", "m", "m"]

def print_total_computation_time(start_time, end_time):
    """Prints the total computation time."""
    total_time = end_time - start_time
    print(f"\n⏱️ Total Computation Time: {total_time:.2f} seconds")

def print_ppi_chart(ppi_values, width=50, height=10):
    """Displays an ASCII chart of PPI values over iterations with correct X-axis labels."""
    if not ppi_values:
        print("No data to display.")
        return

    min_val, max_val = 0.0, 1.0  # Fixed Y-axis range
    num_iterations = len(ppi_values)

    # Ensure only one PPI per iteration is taken
    scaled_values = [(ppi_values[i] - min_val) / (max_val - min_val + 1e-9) for i in range(num_iterations)]

    # Initialize empty chart grid
    chart = [[" "] * width for _ in range(height)]

    for i, val in enumerate(scaled_values):
        x_pos = int((i / max(1, num_iterations - 1)) * (width - 1))
        y_pos = height - 1 - int(val * (height - 1))
        chart[y_pos][x_pos] = "◯"  # Use circle marker

    # Print chart with Y-axis labels
    for row_idx, row in enumerate(chart):
        y_label = f"{(1 - row_idx / (height - 1)):.1f}".ljust(4)
        print(y_label + "|" + "".join(row))

    # Print X-axis line
    print("    " + "-" * width)

    # ✅ Fix: Properly Space Iteration Labels and Avoid `IndexError`
    num_ticks = min(5, num_iterations)  # Set max 5 labels
    tick_positions = [int(i * (num_iterations - 1) / (num_ticks - 1)) for i in range(num_ticks)]

    # Ensure tick_positions does not exceed the width of the plot
    tick_labels = {int((pos / max(1, num_iterations - 1)) * (width - 1)): str(pos) for pos in tick_positions}

    # Scale the labels to match the width
    scaled_x_labels = [tick_labels.get(i, " ") for i in range(width)]

    print("    " + "".join(scaled_x_labels))
    print(" " * (width // 2 - 5) + "Iterations")


def print_iteration_summary(iteration, num_iterations, best_ppi, particles, best_position):
    """Prints a clean summary of the PSO iteration to the terminal."""
    sys.stdout.write(f"\rIteration {iteration}/{num_iterations} | Best PPI: {best_ppi:.5f} ")
    sys.stdout.flush()

    # Display best solution in table format
    print("\n\n📊 Best Solution So Far 📊")
    for i, param in enumerate(param_names):
        print(f"{param.ljust(6)}: {best_position[i]:.2f} {param_units[i]}")

    #print("\n📌 Particles Being Evaluated:")
    #for i, particle in enumerate(particles):
    #    particle_str = ", ".join(f"{p:.2f} {u}" for p, u in zip(particle, param_units))
    #    print(f"Particle {i+1}: [{particle_str}]")

    print("-" * 50)

def print_itrsum_divrs(iteration, num_iterations, best_ppi, particles, best_position):
    """Prints a clean summary of the PSO iteration to the terminal."""
    diversity = compute_particle_diversity(particles)
    sys.stdout.write(f"\rIteration {iteration}/{num_iterations} | Best PPI: {best_ppi:.5f} | Diversity = {diversity:.4f} ")
    sys.stdout.flush()
    # Display best solution in table format
    #print("\n\n📊 Particle diversity 📊")
    #diversity = compute_particle_diversity(particles)
    #print(f"Iteration {iteration}: Diversity = {diversity:.4f}")
    print("-" * 20)


def compute_particle_diversity(particles):
    """
    Computes diversity of a swarm based on the standard deviation of particle positions.
    
    Args:
        particles (numpy.ndarray): Shape (N, D) where N = number of particles, D = dimensions.

    Returns:
        float: Average standard deviation across all dimensions.
    """
    if len(particles.shape) != 2:
        raise ValueError("Input particles must be a 2D NumPy array (N, D).")
    
    # Standardize each parameter (Z-score normalization)
    mean_vals = np.mean(particles, axis=0)  # Mean per parameter
    std_vals = np.std(particles, axis=0)  # Std per parameter

    # Avoid division by zero for constant parameters
    std_vals[std_vals == 0] = 1  

    standardized_particles = (particles - mean_vals) / std_vals  # Standardized values

    # Compute standard deviation over all particles for each parameter
    std_devs = np.std(standardized_particles, axis=0)

    # Compute mean diversity across parameters
    diversity = np.mean(std_devs)
    
    return diversity

