# PCB Design Optimization using Grid Search and Monte Carlo

This Python script performs an optimization of Printed Circuit Board (PCB) design parameters (Length, Width, Component Area, and Number of Layers). It aims to maximize a defined utility function, which balances functional board space against manufacturing cost and efficiency, subject to various constraints. The script implements both Grid Search and Monte Carlo optimization algorithms and visualizes the results.

## Project Overview

The script helps in making early-stage PCB design decisions by:

- Defining a mathematical model of the PCB design problem including:
  - Design Variables: PCB Length (L), Width (W), target Component Area ($A_{comp}$), Number of Layers ($N_{layers}$).
  - Objective Function: Maximize Utility $U = \frac{((L \times W) - A_{comp}) \times N_{per\_sheet}}{Cost_{sheet}(N_{layers})}$.
  - Constraints: Functional area, manufacturing quantity from a fixed number of sheets, and variable bounds.
- Implementing two search algorithms:
  1. **Grid Search:** Systematically evaluates a predefined grid of parameter combinations.
  2. **Monte Carlo Search:** Explores the design space through random sampling.
- Reporting the best solutions found by each method.
- Visualizing the utility landscape using contour plots derived from the Grid Search results, with Monte Carlo samples overlaid for comparison.

## Setup and Execution

Follow these steps to set up a virtual environment and run the script:

### 1. Prerequisites

- Python 3.6 or higher installed on your system.
- `pip` (Python package installer).

### 2. Create a Virtual Environment (Recommended)

A virtual environment helps manage project dependencies separately.

   **On macOS and Linux:**

```bash
   python3 -m venv venv_pcb_opt
   source venv_pcb_opt/bin/activate
```

**On Windows:**

**Bash**

```
python -m venv venv_pcb_opt
.\venv_pcb_opt\Scripts\activate
```

After activation, your terminal prompt should change to indicate you are in the virtual environment (e.g., `(venv_pcb_opt)`).

### 3. Install Dependencies

With your virtual environment activated, install the required libraries:

**Bash**

```
pip install -r requirements.txt
```

This command will read the `requirements.txt` file and install `numpy` and `matplotlib`.

### 4. Prepare the Script

Save the Python code (e.g., from artifact `python_mc_grid_contour_v1` or your latest version) as a `.py` file (e.g., `pcb_optimization_script.py`) in your project directory.

### 5. Run the Script

Execute the script from your terminal:

**Bash**

```
python pcb_optimization_script.py
```

The script will:

* Print the fixed problem parameters being used.
* Run the Grid Search optimization and print its results.
* Run the Monte Carlo optimization and print its results.
* Generate and display separate plot windows for each layer configuration (e.g., 2-layer, 4-layer), showing the utility contour from the Grid Search and the overlaid Monte Carlo samples.

## Understanding the Output

* **Console Output:** Provides details on the fixed parameters, progress of each search algorithm, total points evaluated, feasible points found, and the best overall solution (Utility, L, W, A_comp, N_layers) identified by both Grid Search and Monte Carlo methods.
* **Plots:** Each figure visualizes the L-W design space for a specific number of layers.
  * The colored contours represent the maximum utility found by the Grid Search for each (L,W) pair (considering variations in A_comp).
  * Monte Carlo samples are overlaid:
    * Light grey 'x' markers: Infeasible random samples.
    * Green circles: Feasible random samples.
    * Magenta pentagrams with text: Best feasible sample found by Monte Carlo  *for that specific layer* .
    * Cyan hexagrams with text: Best feasible sample found by Grid Search  *for that specific layer* .

## Customization

You can modify the script to explore different scenarios:

* **`params` dictionary:** Adjust fixed parameters like sheet quantity (`S_sheets`), required PCBs (`N_total_req`), spacing (`S_x`, `S_y`), margins (`M`), component connectivity (`N_conn`), routing factor (`k_route`), and cost parameters (`C_base`, `C_add_layer`).
* **Algorithm Settings:**
  * `L_range`, `W_range`, `A_comp_range`: Modify the search bounds for the design variables.
  * `N_layers_options`: Change the discrete layer counts to evaluate (e.g., `[2, 4, 6]`).
  * `num_samples_mc`: Increase for a more thorough (but slower) Monte Carlo search.
  * `grid_points_L`, `grid_points_W`, `grid_points_A_comp`: Increase for a denser (more accurate but much slower) Grid Search.

## Deactivating the Virtual Environment

When you are done working on the project, you can deactivate the virtual environment:

**Bash**

```
deactivate
```
