import numpy as np
import matplotlib.pyplot as plt
import time
import random

# --- Helper Functions ---
def calculate_A_rout(N_layers, A_comp, N_conn, k_route):
    """Calculates estimated routing area."""
    if N_layers <= 0:
        return np.inf
    if A_comp < 0: 
        A_comp = 0 # Ensure A_comp is not negative for sqrt
    return k_route * (N_conn * np.sqrt(max(0, A_comp))) / N_layers

def calculate_N_per_sheet(L, W, W_usable, H_usable, S_x, S_y):
    """Calculates number of PCBs per sheet, considering orientation."""
    if L <= 0 or W <= 0 or (L + S_x) <= 0 or (W + S_y) <= 0 or \
       (W + S_x) <= 0 or (L + S_y) <= 0: # Check all potential divisors
        return 0

    # Orientation 1 (L horizontal)
    N_x1 = np.floor((W_usable + S_x) / (L + S_x))
    N_y1 = np.floor((H_usable + S_y) / (W + S_y))
    N1 = N_x1 * N_y1

    # Orientation 2 (W horizontal)
    N_x2 = np.floor((W_usable + S_x) / (W + S_x))
    N_y2 = np.floor((H_usable + S_y) / (L + S_y))
    N2 = N_x2 * N_y2
    
    return max(0, N1, N2) # Ensure non-negative

def calculate_cost_sheet_modified(N_layers, C_base, C_add_layer):
    """Calculates cost per sheet based on layer count (modified formula)."""
    if N_layers < 2: # Invalid layer count
        return np.inf 
    if N_layers == 2:
        return C_base
    # For N_layers > 2, cost of additional layers beyond the first two
    additional_cost = (N_layers - 2) * C_add_layer
    return C_base + additional_cost

def check_constraints_pcb(L, W, A_comp, N_layers, params):
    """Checks if a design point satisfies all constraints."""
    A_comp_min_abs = params['A_comp_min_abs']
    N_conn = params['N_conn']
    k_route = params['k_route']
    W_usable = params['W_usable']
    H_usable = params['H_usable']
    S_x = params['S_x']
    S_y = params['S_y']
    S_sheets = params['S_sheets']
    N_total_req = params['N_total_req']

    if L <= 0 or W <= 0 or A_comp < A_comp_min_abs: # Basic validity
        return False

    # 1. Functional Area Constraint
    A_rout = calculate_A_rout(N_layers, A_comp, N_conn, k_route)
    A_req = A_comp + A_rout
    if (L * W) < A_req:
        return False

    # 2. Manufacturing Quantity Constraint
    N_per_sheet = calculate_N_per_sheet(L, W, W_usable, H_usable, S_x, S_y)
    N_produced = N_per_sheet * S_sheets
    if N_produced < N_total_req:
        return False
        
    return True

def calculate_objective_pcb(L, W, A_comp, N_layers, params):
    """Calculates the Utility objective function U."""
    # Check feasibility first
    if not check_constraints_pcb(L, W, A_comp, N_layers, params):
        return -np.inf # Infeasible

    C_base = params['C_base']
    C_add_layer = params['C_add_layer']
    W_usable = params['W_usable']
    H_usable = params['H_usable']
    S_x = params['S_x']
    S_y = params['S_y']

    N_per_sheet = calculate_N_per_sheet(L, W, W_usable, H_usable, S_x, S_y)
    cost_sheet = calculate_cost_sheet_modified(N_layers, C_base, C_add_layer)

    # Check for valid intermediate values
    if N_per_sheet <= 0 or cost_sheet <= 0 or cost_sheet == np.inf:
        return -np.inf

    area_margin = (L * W) - A_comp
    if area_margin <= 0: # Should be caught by functional area constraint if A_req is positive
        return -np.inf
        
    utility = (area_margin * N_per_sheet) / cost_sheet
    return utility

# --- Grid Search Function ---
def perform_grid_search(num_L, num_W, num_A, L_r, W_r, A_comp_r, N_layers_opts, p_gs):
    """Performs Grid search for PCB optimization."""
    L_grid_vals = np.linspace(L_r[0], L_r[1], num_L)
    W_grid_vals = np.linspace(W_r[0], W_r[1], num_W)
    A_comp_grid_vals = np.linspace(A_comp_r[0], A_comp_r[1], num_A)

    best_U_overall_gs = -np.inf
    best_vars_overall_gs = {} # Using dict instead of struct
    
    all_evaluated_points_gs = [] # List of dicts
    
    total_evals = 0
    feasible_evals = 0

    for N_layers_val_gs in N_layers_opts:
        for L_val_gs in L_grid_vals:
            for W_val_gs in W_grid_vals:
                for A_comp_val_gs in A_comp_grid_vals:
                    total_evals += 1
                    current_eval_pt = {
                        'L': L_val_gs, 'W': W_val_gs, 'A_comp': A_comp_val_gs, 
                        'N_layers': N_layers_val_gs, 'U': -np.inf, 'feasible': False
                    }
                    if check_constraints_pcb(L_val_gs, W_val_gs, A_comp_val_gs, N_layers_val_gs, p_gs):
                        feasible_evals += 1
                        U_gs = calculate_objective_pcb(L_val_gs, W_val_gs, A_comp_val_gs, N_layers_val_gs, p_gs)
                        current_eval_pt['U'] = U_gs
                        current_eval_pt['feasible'] = True
                        if U_gs > best_U_overall_gs:
                            best_U_overall_gs = U_gs
                            best_vars_overall_gs = current_eval_pt.copy() # Store a copy
                    all_evaluated_points_gs.append(current_eval_pt)
    return best_vars_overall_gs, all_evaluated_points_gs, total_evals, feasible_evals

# --- Monte Carlo Search Function ---
def perform_monte_carlo_search(num_samples, L_r, W_r, A_comp_r, N_layers_opts, p_mc):
    """Performs Monte Carlo search for PCB optimization."""
    best_U_overall = -np.inf
    best_vars_overall = {} 
    all_sampled_points = []

    for _ in range(num_samples):
        L_val_mc = random.uniform(L_r[0], L_r[1])
        W_val_mc = random.uniform(W_r[0], W_r[1])
        A_comp_val_mc = random.uniform(A_comp_r[0], A_comp_r[1])
        N_layers_val_mc = random.choice(N_layers_opts)
        current_sample_mc = {
            'L': L_val_mc, 'W': W_val_mc, 'A_comp': A_comp_val_mc, 
            'N_layers': N_layers_val_mc, 'U': -np.inf, 'feasible': False
        }
        if check_constraints_pcb(L_val_mc, W_val_mc, A_comp_val_mc, N_layers_val_mc, p_mc):
            U_mc = calculate_objective_pcb(L_val_mc, W_val_mc, A_comp_val_mc, N_layers_val_mc, p_mc)
            current_sample_mc['U'] = U_mc
            current_sample_mc['feasible'] = True
            if U_mc > best_U_overall:
                best_U_overall = U_mc
                best_vars_overall = current_sample_mc.copy()
        all_sampled_points.append(current_sample_mc)
    return best_vars_overall, all_sampled_points

# --- Main Script Logic ---
if __name__ == "__main__":
    # --- Define Problem Parameters (Dummy Values) ---
    params = {
        'S_sheets': 7,
        'N_total_req': 50,
        'S_x': 2.0,  # Panel spacing X (mm)
        'S_y': 2.0,  # Panel spacing Y (mm)
        'M': 5.0,    # Panel margin (mm)
        'A_comp_min_abs': 50.0, # Absolute minimum component area
        'N_conn': 150.0,
        'k_route': 0.5, # Routing Constant (mm/connection)
        'C_base': 20.0, # Base cost for a 2-layer sheet
        'C_add_layer': 7.5, # Cost per *additional individual layer*
    }
    sheet_dim = 100.0 # mm
    params['W_usable'] = sheet_dim - 2 * params['M']
    params['H_usable'] = sheet_dim - 2 * params['M']

    # --- Print Fixed Parameters ---
    print("--- Fixed Problem Parameters ---")
    for key, value in params.items():
        print(f"  {key}: {value}")
    print(f"  Sheet Dimension: {sheet_dim} mm")
    print("---------------------------------")

    # --- Optimization Algorithm Settings ---
    L_range = [20, 50] # mm
    W_range = [20, 50] # mm
    A_comp_range = [max(params['A_comp_min_abs'], 200), 1000] # mm^2
    N_layers_options = [2, 4] # Discrete options

    # Monte Carlo Settings
    num_samples_mc = 10 

    # Grid Search Settings
    grid_points_L = 15 
    grid_points_W = 15
    grid_points_A_comp = 7 

    # --- Perform Grid Search Optimization ---
    print("\nStarting Grid Search Optimization...")
    gs_start_time = time.time()
    best_vars_overall_gs, gs_all_evaluated_points, gs_total_evals, gs_feasible_evals = perform_grid_search(
        grid_points_L, grid_points_W, grid_points_A_comp, L_range, W_range, A_comp_range, N_layers_options, params)
    gs_end_time = time.time()
    print("\nGrid Search Complete.")
    # print(f"  Grid Search Time taken: {gs_end_time - gs_start_time:.2f} seconds") # Time removed
    print(f"  Total grid points evaluated: {gs_total_evals}")
    print(f"  Feasible grid points found: {gs_feasible_evals}")

    if best_vars_overall_gs: # Check if dict is not empty
        print("\n--- Overall Best Solution Found by Grid Search ---")
        print(f"  Utility (U): {best_vars_overall_gs['U']:.4f} mm^2/$")
        print(f"  L: {best_vars_overall_gs['L']:.2f} mm, W: {best_vars_overall_gs['W']:.2f} mm, A_comp: {best_vars_overall_gs['A_comp']:.2f} mm^2, N_layers: {best_vars_overall_gs['N_layers']}")
    else:
        print("\nNo feasible solution found by Grid Search.")

    # --- Perform Monte Carlo Optimization ---
    print("\nStarting Monte Carlo Optimization...")
    mc_start_time = time.time()
    best_vars_overall_mc, mc_all_sampled_points = perform_monte_carlo_search(num_samples_mc, L_range, W_range, A_comp_range, N_layers_options, params)
    mc_end_time = time.time()
    print("\nMonte Carlo Search Complete.")
    # print(f"  Monte Carlo Time taken: {mc_end_time - mc_start_time:.2f} seconds") # Time removed
    print(f"  Total samples generated: {num_samples_mc}")
    mc_feasible_count = sum(1 for p in mc_all_sampled_points if p['feasible'])
    print(f"  Feasible samples found: {mc_feasible_count}")

    if best_vars_overall_mc:
        print("\n--- Overall Best Solution Found by Monte Carlo ---")
        print(f"  Utility (U): {best_vars_overall_mc['U']:.4f} mm^2/$")
        print(f"  L: {best_vars_overall_mc['L']:.2f} mm, W: {best_vars_overall_mc['W']:.2f} mm, A_comp: {best_vars_overall_mc['A_comp']:.2f} mm^2, N_layers: {best_vars_overall_mc['N_layers']}")
    else:
        print("\nNo feasible solution found by Monte Carlo.")

    # --- Find best MC result for each layer for plotting overlay ---
    best_vars_per_layer_mc = []
    for N_layer_opt_val in N_layers_options:
        layer_best_U_temp = -np.inf
        layer_best_vars_temp = {}
        for pt in mc_all_sampled_points:
            if pt['N_layers'] == N_layer_opt_val and pt['feasible']:
                if pt['U'] > layer_best_U_temp:
                    layer_best_U_temp = pt['U']
                    layer_best_vars_temp = pt.copy()
        best_vars_per_layer_mc.append(layer_best_vars_temp if layer_best_vars_temp else {'feasible': False, 'N_layers': N_layer_opt_val})


    # --- Find best Grid Search result for each layer for plotting ---
    best_vars_per_layer_gs = []
    if gs_feasible_evals > 0:
        for N_layer_opt_val in N_layers_options:
            layer_best_U_temp_gs = -np.inf
            layer_best_vars_temp_gs = {}
            for pt in gs_all_evaluated_points:
                if pt['N_layers'] == N_layer_opt_val and pt['feasible']:
                    if pt['U'] > layer_best_U_temp_gs:
                        layer_best_U_temp_gs = pt['U']
                        layer_best_vars_temp_gs = pt.copy()
            best_vars_per_layer_gs.append(layer_best_vars_temp_gs if layer_best_vars_temp_gs else {'feasible': False, 'N_layers': N_layer_opt_val})
    else: # Populate with non-feasible entries if no feasible GS points
        for N_layer_opt_val in N_layers_options:
             best_vars_per_layer_gs.append({'feasible': False, 'N_layers': N_layer_opt_val})


    # --- Visualization with contourf (using Grid Search data for contours) ---
    print("\nGenerating plots (Grid Search contour with MC overlay)...")

    L_grid_vector = np.linspace(L_range[0], L_range[1], grid_points_L)
    W_grid_vector = np.linspace(W_range[0], W_range[1], grid_points_W)

    for k_idx, N_layers_fixed in enumerate(N_layers_options):
        plt.figure(figsize=(8, 7)) # Create a new figure for each layer
        ax = plt.gca() 
        
        U_gs_grid_for_contour = np.full((len(W_grid_vector), len(L_grid_vector)), np.nan)

        for i_l, L_val in enumerate(L_grid_vector):
            for i_w, W_val in enumerate(W_grid_vector):
                max_U_for_LW_gs = -np.inf
                any_A_comp_feasible_gs = False
                for pt in gs_all_evaluated_points:
                    if pt['N_layers'] == N_layers_fixed and \
                       abs(pt['L'] - L_val) < 1e-6 and \
                       abs(pt['W'] - W_val) < 1e-6 and \
                       pt['feasible']:
                        any_A_comp_feasible_gs = True
                        if pt['U'] > max_U_for_LW_gs:
                            max_U_for_LW_gs = pt['U']
                if any_A_comp_feasible_gs:
                    U_gs_grid_for_contour[i_w, i_l] = max_U_for_LW_gs
        
        if np.sum(~np.isnan(U_gs_grid_for_contour)) > 0:
            contour_plot = ax.contourf(L_grid_vector, W_grid_vector, U_gs_grid_for_contour, levels=15, cmap='viridis', extend='min')
            plt.colorbar(contour_plot, ax=ax, label='Max Utility U (mm^2/$) from Grid Search')
        
        ax.set_title(f'{N_layers_fixed}-Layer Design (Grid Search Contour & MC Samples)')
        ax.set_xlabel('PCB Length L (mm)')
        ax.set_ylabel('PCB Width W (mm)')
        ax.set_xlim(L_range)
        ax.set_ylim(W_range)
        ax.set_aspect('equal', adjustable='box') # Square axis
        ax.grid(True, linestyle=':', alpha=0.6)

        # Overlay Monte Carlo Samples
        mc_layer_points = [p for p in mc_all_sampled_points if p['N_layers'] == N_layers_fixed]
        infeasible_L = [p['L'] for p in mc_layer_points if not p['feasible']]
        infeasible_W = [p['W'] for p in mc_layer_points if not p['feasible']]
        if infeasible_L:
            ax.scatter(infeasible_L, infeasible_W, marker='x', color='silver', s=50, alpha=0.7)

        feasible_L = [p['L'] for p in mc_layer_points if p['feasible']]
        feasible_W = [p['W'] for p in mc_layer_points if p['feasible']]
        if feasible_L:
            ax.scatter(feasible_L, feasible_W, marker='o', edgecolors='k', facecolors='lime', s=60, alpha=0.7)

        # Mark the layer-specific best MC point
        current_layer_best_mc_data = best_vars_per_layer_mc[k_idx]
        if current_layer_best_mc_data and current_layer_best_mc_data.get('feasible', False): # Check if dict is not empty and feasible
            ax.plot(current_layer_best_mc_data['L'], current_layer_best_mc_data['W'], 'p', 
                    markeredgecolor='k', markerfacecolor='magenta', markersize=12)
            ax.text(current_layer_best_mc_data['L'] + 0.5, current_layer_best_mc_data['W'] + 0.5, 
                    f"MC Best U={current_layer_best_mc_data['U']:.1f}\nA_c={current_layer_best_mc_data['A_comp']:.0f}",
                    fontsize=7, bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.3'))

        # Mark the layer-specific best Grid Search point
        current_layer_best_gs_data = best_vars_per_layer_gs[k_idx]
        if current_layer_best_gs_data and current_layer_best_gs_data.get('feasible', False): # Check if dict is not empty and feasible
            ax.plot(current_layer_best_gs_data['L'], current_layer_best_gs_data['W'], 'h', 
                    markeredgecolor='k', markerfacecolor='cyan', markersize=12)
            ax.text(current_layer_best_gs_data['L'] + 0.5, current_layer_best_gs_data['W'] - 1.5, # Adjusted Y for less overlap
                    f"Grid Best U={current_layer_best_gs_data['U']:.1f}\nA_c={current_layer_best_gs_data['A_comp']:.0f}",
                    fontsize=7, bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.3'))
        
        plt.tight_layout()
    
    plt.show()
    print('\nNote: Contour plot represents the utility landscape found by Grid Search.')


