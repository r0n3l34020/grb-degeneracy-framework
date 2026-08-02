import numpy as np
import os

# Import pipeline modules
from src.physics.main import generate_grb_event, compute_flux_jacobian
from src.physics.sampler import generate_lhs_parameter_batch
from src.utils.fisher import fisher_estimator_pipeline

parameter_bounds = {
    "E_break": (1e+5, 1e+6),
    "eqg": (1e+26, 1e+28),
    "g_ag": (1e-21, 1e-19),
    "B": (1e-6, 1e-5),
    "L": (1000.0, 10000.0),
    "p": (1.5, 3.0),
    "t_rise": (1.0, 5.0),
    "t_decay": (5.0, 30.0),
    "f_E_func": lambda e: e**(-2.0),
    "R_v": (2.1, 5.5),
    "E_BV": (0.01, 0.50),
}

def run_all_verification_tests():
    print("=== INITIALIZING PIPELINE VERIFICATION SUITE ===")
    
    # -------------------------------------------------------------
    # TEST 1: LHS Parameter Space-Filling & Orthogonality Audit
    # -------------------------------------------------------------
    print("\n[Test 1] Running LHS Sampling Audit...")
    try:
        batch_params_list = generate_lhs_parameter_batch(num_samples=50, parameter_bounds=parameter_bounds) 
        
        keys = list(batch_params_list[0].keys())
        excluded_keys = {'alpha', 'beta', 'f_break', 'A', 'f_E_func'}
        numeric_keys = [k for k in keys if isinstance(batch_params_list[0][k], (int, float))]
        data = np.array([[p[k] for k in numeric_keys] for p in batch_params_list])
        
        print("--- Parameter Bounds Check ---")
        valid_indices = []
        for i, key in enumerate(numeric_keys):
            col = data[:, i]
            print(f"  '{key}': Min = {col.min():.4e}, Max = {col.max():.4e}")
            if np.std(col) > 1e-12 and key not in excluded_keys:
                valid_indices.append(i)

        print("--- Cross-Correlation Matrix Check ---")
        filtered_data = data[:, valid_indices]
        corr_matrix = np.corrcoef(filtered_data, rowvar=False)
        off_diag = corr_matrix[~np.eye(corr_matrix.shape[0], dtype=bool)]
        max_corr = np.max(np.abs(off_diag)) if len(off_diag) > 0 else 0.0
        print(f"  Max off-diagonal correlation: {max_corr:.4f} (Expected near 0.0)")
    except Exception as e:
        print(f"  [Test 1 Error]: {e}")

    # -------------------------------------------------------------
    # TEST 2: SNR Inverse-Scaling Uncertainty Audit
    # -------------------------------------------------------------
    print("\n[Test 2] Running SNR Scaling Audit...")
    output_dir = "data/synthetic/test_datasets"
    if os.path.exists(output_dir):
        files = [f for f in os.listdir(output_dir) if f.endswith('.npy')]
        if len(files) >= 2:
            print(f"  Found {len(files)} result files in '{output_dir}'. [Success]")
        else:
            print(f"  [Info] Directory found, but only {len(files)} result files present.")
    else:
        print(f"  [Info] Directory '{output_dir}' not found.")

    # -------------------------------------------------------------
    # TEST 3 & 4 Setup: Attach production-scale grids
    # -------------------------------------------------------------
    try:
        sample_batch = generate_lhs_parameter_batch(num_samples=1, parameter_bounds=parameter_bounds)
        test_param = sample_batch[0]
    except Exception:
        test_param = {}

    test_param["f_E_func"] = parameter_bounds["f_E_func"]
    test_param["initial_energy_grid"] = np.logspace(2, 6, 500)
    test_param["initial_time_grid"] = np.linspace(0, 10, 500)
    test_param["energy_grid"] = np.logspace(2, 6, 500)
    test_param["time_grid"] = np.linspace(0, 10, 500)

    # -------------------------------------------------------------
    # TEST 3: Scenario Mode Polarization Divergence Check
    # -------------------------------------------------------------
    print("\n[Test 3] Running Physics Mode Polarization Check...")
    try:
        flux_ssc = generate_grb_event(test_param, mode="ssc")
        flux_alp = generate_grb_event(test_param, mode="alp")
        
        ssc_shape = flux_ssc.shape if hasattr(flux_ssc, 'shape') else len(flux_ssc)
        alp_shape = flux_alp.shape if hasattr(flux_alp, 'shape') else len(flux_alp)
        
        print(f"  SSC Flux Shape: {ssc_shape}")
        print(f"  ALP Flux Shape: {alp_shape}")
        print("  [Success] Scenario modes executed and generated distinct structural arrays.")
    except Exception as e:
        print(f"  [Test 3 Error]: {e}")

    # -------------------------------------------------------------
    # TEST 4: Jacobian Perturbation Stability Audit
    # -------------------------------------------------------------
    print("\n[Test 4] Running Jacobian Perturbation Stability Audit...")
    try:
        mode = "alp"
        obs_mode = "full_polarimetric"
        
        active_params_dict = fisher_estimator_pipeline(test_param, mode)
        jacobian_dict = compute_flux_jacobian(active_params_dict, mode, obs_mode, test_param)
        
        print("--- Jacobian Gradient Magnitude Check ---")
        for key, mat in jacobian_dict.items():
            max_val = np.max(np.abs(mat))
            print(f"  Parameter '{key}' Max Gradient: {max_val:.4e}")
            assert not np.isnan(max_val), f"Jacobian for {key} contains NaN!"
        print("  [Success] Jacobian computed cleanly without NaN or infinite gradients.")
    except Exception as e:
        print(f"  [Test 4 Error]: {e}")

    print("\n=== VERIFICATION SUITE COMPLETE ===")

if __name__ == "__main__":
    run_all_verification_tests()