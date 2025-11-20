
#include <string>
#include <filesystem>
#include <iostream>
#include <functional>
#include <chrono>
#include <omp.h>

#include "ising_model.hpp"

using namespace std;

/**
 * @brief Container for ensemble-averaged observables from an MCMC run.
 *
 * This struct stores the quantities computed during a Monte Carlo simulation:
 *   - mean energy per spin   ⟨ε⟩,
 *   - mean absolute magnetisation per spin   ⟨|m|⟩,
 *   - heat capacity per spin   C_V / N,
 *   - magnetic susceptibility per spin   χ / N.
 *
 * All quantities assume units where J = 1 and k_B = 1.
 *
 */
struct MCResults {
    double mean_eps;       // Mean energy per spin
    double mean_abs_m;     // Mean absolute magnetisation per spin
    double Cv_per_spin;    // Heat capacity per spin C_V / N
    double chi_per_spin;   // Magnetic susceptibility per spin χ / N
    std::vector<double> eps_trajectory;  // Used for the burnin study
    std::vector<double> mean_eps_trajectory; // Used for the burnin study
};

/**
 * @brief Run a Metropolis–Hastings Monte Carlo simulation of the 2D Ising model.
 *
 * Performs n_cycles Monte Carlo cycles for an L×L lattice at temperature T,
 * using the Metropolis single–spin–flip algorithm implemented in IsingModel.
 * One Monte Carlo cycle corresponds to N = L×L attempted spin flips.
 *
 * For each cycle the routine samples the system energy and magnetisation,
 * and accumulates ensemble averages of:
 *   - mean energy per spin ⟨ε⟩,
 *   - mean absolute magnetisation per spin ⟨|m|⟩,
 *   - heat capacity per spin C_V/N,
 *   - susceptibility per spin χ/N.
 *
 * The function returns these quantities in an MCResults struct.
 *
 * @param L          Linear lattice size (the system has N = L×L spins).
 * @param T          Temperature, in units where k_B = 1.
 * @param n_cycles   Number of Monte Carlo cycles to perform.
 * @param ordered    If true, the initial state is fully ordered (all spins +1).
 *                   If false, the initial state is random with spins ±1.
 * @param seed       Seed for the random number generator used internally.
 *
 * @return MCResults containing ensemble-averaged observables from the run.
 *
 */
MCResults run_MCMC_simulation(const int L, const double T, const int n_cycles, 
                              const bool ordered, const unsigned seed, const int burn_in_cycles = 0)
{
    const int N = L*L;
    const double inv_T = 1.0/T;
    const double inv_T2 = 1.0 / (T*T);
    IsingModel model(L, T, ordered, seed);
    
    // Initialize accumulators
    double sum_eps  = 0.0;
    double sum_eps2 = 0.0;
    double sum_absM  = 0.0;
    double sum_M2 = 0.0;
    std::vector<double> eps_trajectory;
    std::vector<double> mean_eps_trajectory;
    
    for (int cycle = 0; cycle < n_cycles; cycle++) {
        // One MC cycle = N attempted flips
        for (int k = 0; k < N; k++) {
            model.metropolis_update();
        }
        
        // Compute E(s) and M(s)
        double E = model.total_energy();
        double eps = E / N;
        double M = model.magnetization();
        double m = M / N;
        
        // Skip measurements during burn-in
        if (cycle < burn_in_cycles) {
            continue;
        }
        
        // Update sums
        sum_eps  += eps;
        sum_eps2 += eps*eps;
        sum_absM += fabs(m);
        sum_M2 += m*m;
        
        eps_trajectory.push_back(eps);
        mean_eps_trajectory.push_back(sum_eps / (cycle+1));
    }
    
    //
    // Compute Observables from accumulated sums
    //
    
    // Divide by the number of measurement cycles
    const double n = double(n_cycles - burn_in_cycles);
    const double inv_n = 1.0/n;
    
    double mean_eps = sum_eps * inv_n;
    double mean_abs_m = sum_absM * inv_n;
    
    // Heat capacity per spin
    double Cv = N * inv_T2 * ( (sum_eps2*inv_n - mean_eps*mean_eps) );
    
    // Susceptibility per spin
    double chi = N * inv_T * ( (sum_M2*inv_n - mean_abs_m*mean_abs_m) );
    
    return {mean_eps, mean_abs_m, Cv, chi, eps_trajectory, mean_eps_trajectory};
}

/**
 * @brief Compute exact analytical observables for the 2×2 Ising model.
 *
 * Evaluates closed-form expressions for the thermodynamic quantities of a
 * 2×2 lattice with periodic boundary conditions at temperature T.
 * These analytical formulas are derived from the full enumeration of all
 * 16 microstates and provide exact reference values for validating Monte Carlo
 * simulations
 *
 * The routine computes and outputs:
 *   - eps          : mean energy per spin ⟨ε⟩,
 *   - abs_m        : mean absolute magnetisation per spin ⟨|m|⟩,
 *   - Cv_per_spin  : heat capacity per spin  C_V / N,
 *   - chi_per_spin : magnetic susceptibility per spin  χ / N.
 *
 * All observables are computed in units where J = 1 and k_B = 1.
 * The results are written into the provided reference arguments.
 *
 * @param T              Temperature (k_B = 1).
 * @param eps            Output: mean energy per spin ⟨ε⟩.
 * @param abs_m          Output: mean absolute magnetisation per spin ⟨|m|⟩.
 * @param Cv_per_spin    Output: heat capacity per spin C_V / N.
 * @param chi_per_spin   Output: susceptibility per spin χ / N.
 *
 */
void analytic_2x2(const double T,
                  double &eps, double &abs_m,
                  double &Cv_per_spin, double &chi_per_spin)
{
    const double inv_T = 1.0 / T;
    const double inv_T2 = 1.0 / (T*T);
    const double K = inv_T; // K = beta J, with J = 1, k_B = 1
    const double e8K  = exp(8.0 * K);
    const double e_8K = exp(-8.0 * K);
    
    const double Z = 4.0 * cosh(8.0 * K) + 12.0;
    const double inv_Z = 1.0/Z;
    
    // <epsilon> (energy per spin)
    eps = - (4.0 * inv_Z) * (e8K - e_8K);
    
    // <|m|> (magnetisation per spin)
    abs_m = (2.0 * inv_Z) * (e8K + 2.0);
    
    // Heat capacity per spin
    double term1 = (8.0 * inv_Z) * (e8K + e_8K);
    double term2 = (4.0 * inv_Z) * (e8K - e_8K);
    Cv_per_spin = 4.0 * inv_T2 * (term1 - term2 * term2);
    
    // Susceptibility per spin
    term1 = (1.0 * inv_Z) * (2.0 * e8K + 2.0);
    term2 = (1.0 * inv_Z) * (2.0 * e8K + 4.0);
    chi_per_spin = 4.0 * inv_T * (term1 - term2 * term2);
}

/**
 * @brief Print usage instructions for the Ising model simulation program.
 *
 * Outputs a detailed description of the required command-line arguments
 * and optional flags for configuring a Monte Carlo simulation of the
 * two-dimensional Ising model. This routine is intended to guide users
 * in providing correct input parameters when launching the program.
 *
 */
void print_usage(const char* filename) {
    cerr << "Usage: " << filename 
         << " <file_name_prefix> <L> <T_min> <T_max> <T_count> <N_MC_CYCLES> <base_seed> [options]\n\n"
         << "Arguments:\n"
         << "  file_name_prefix: \tPrefix for output files\n"
         << "  L: \t\t\tLattice size (integer)\n"
         << "  T_min: \t\tMinimum temperature\n"
         << "  T_max: \t\tMaximum temperature\n"
         << "  T_count: \t\tNumber of temperature points (including endpoints)\n"
         << "  N_MC_CYCLES: \t\tNumber of Monte Carlo cycles per run\n"
         << "  base_seed: \t\tRandom seed for reproducibility\n\n"
         << "Options:\n"
         << "  --enable-ordered-initial-state\tStart simulation from an ordered spin configuration\n"
         << "  --enable-trajectory-mode\t\tEnable the plotting of trajectories per T\n"
         << "  --set-burnin <cycles> \t\tSet the burnin cycle count (default=0)\n"
         << "Example:\n"
         << "  " << filename 
         << "  results 20 2.25 2.35 50 100000 42 --enable-ordered-initial-state\n\n"
         << "  This example runs a simulation on a 20×20 lattice, sweeping temperatures\n"
         << "  from 2.25 to 2.35 in 50 steps, with 100,000 Monte Carlo cycles per run,\n"
         << "  using seed 42, and starting from an ordered initial state.\n\n";
}

int main(int argc, char** argv) {
    
    arma::arma_rng::set_seed_random();
    
    //
    // Reading cmd-arguments
    //
    
    int min_argument_count = 8;
    if (argc < min_argument_count) {
        print_usage(argv[0]);
        return 1;
    }
    
    const string file_name_prefix = argv[1];
    const int    L = atoi(argv[2]);
    const double T_min = atof(argv[3]);
    const double T_max = atof(argv[4]);
    const int T_count = atoi(argv[5]); // number of points including endpoints
    const int N_MC_CYCLES = atoi(argv[6]);
    const unsigned int base_seed = atoi(argv[7]);
    
    bool ordered_initial_state = false;
    bool enable_trajectory_mode = false;
    int burn_in_cycles = 0;
    
    // Handle optional flags
    for (int i = min_argument_count; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--enable-ordered-initial-state") {
            ordered_initial_state = true;
        }
        else if (arg == "--enable-trajectory-mode") {
            enable_trajectory_mode = true;
        }
        else if (arg == "--set-burnin") {
            burn_in_cycles = atoi(argv[++i]);
        }
        else {
            print_usage(argv[0]);
            return 1;
        }
    }
    
    //
    // Determining the output folder
    //
    
    string folder_path = filesystem::exists("../output/") ? "../output/" : "output/";
    while (!filesystem::exists(folder_path)) {
        cout << "Enter path to output folder: ";
        getline(cin, folder_path);
        
        if (!folder_path.empty())
        {
            if (folder_path.back() != '/') {
                folder_path.push_back('/');
            }
        }
    }
    
    //
    // Simulation parameters
    //
    
    const double dT = (T_max - T_min) / (T_count - 1);
    
    //
    // Output Buffers
    //
    
    vector<double> eps_MC(T_count, 0.0);
    vector<double> rel_err_eps(T_count, 0.0);
    vector<double> abs_m_MC(T_count, 0.0);
    vector<double> rel_err_abs_m(T_count, 0.0);
    vector<double> Cv_MC(T_count, 0.0);
    vector<double> chi_MC(T_count, 0.0);
    
    vector<double> T_values(T_count, 0.0);
    vector<double> eps_analytic(T_count, 0.0);
    vector<double> abs_m_analytic(T_count, 0.0);
    vector<double> Cv_analytic(T_count, 0.0);
    vector<double> chi_analytic(T_count, 0.0);
    
    std::vector<std::vector<double>> eps_trajectory(T_count);
    std::vector<std::vector<double>> mean_eps_trajectory(T_count);
    
    //
    // Simulate and Write output
    //
    
    ofstream file;
    if (!enable_trajectory_mode) {
        // One file per number of MC cycles
        string file_name = file_name_prefix + "_" + to_string(N_MC_CYCLES) + ".csv";
        file.open(folder_path + file_name);
        
        // Write Column Headers
        file << setprecision(12) << scientific;
        file << "T,"
             << "eps_MC,eps_analytic,rel_err_eps,"
             << "abs_m_MC,abs_m_analytic,rel_err_abs_m,"
             << "Cv_MC,Cv_analytic,"
             << "chi_MC,chi_analytic\n";
    }
    else {
        // Creating the filename
        string ordered = ordered_initial_state ? "ordered" : "unordered";
        string file_name = file_name_prefix + "_" + ordered + "_" + to_string(N_MC_CYCLES) + ".csv";
        file.open(folder_path + file_name);
        
        // Write Column Headers
        file << "T,cycle,eps,mean_eps\n";
    }
    
    // Start the timer
    using clock_type = std::chrono::high_resolution_clock;
    auto t_start = clock_type::now();
    
#pragma omp parallel for
    for (int i = 0; i < T_count; i++) { // Loop through all T values
        double T = T_min + i * dT;
        
        // Simple deterministic seed depending on T index and MC cycles
        unsigned int seed = base_seed + i + 7919u * N_MC_CYCLES;
        
        // T values
        T_values[i] = T;
        
        // Calculate analytical values
        double eps_a, abs_m_a, Cv_a, chi_a;
        analytic_2x2(T, eps_a, abs_m_a, Cv_a, chi_a);
        
        // Store analytical values
        eps_analytic[i]   = eps_a;
        abs_m_analytic[i] = abs_m_a;
        Cv_analytic[i]    = Cv_a;
        chi_analytic[i]   = chi_a;
        
        // Monte Carlo results
        MCResults mc = run_MCMC_simulation(L, T, N_MC_CYCLES,
                                           ordered_initial_state, seed, burn_in_cycles);
        
        // Store MC results
        eps_MC[i]        = mc.mean_eps;
        abs_m_MC[i]      = mc.mean_abs_m;
        Cv_MC[i]         = mc.Cv_per_spin;
        chi_MC[i]        = mc.chi_per_spin;
        
        // Relative errors (guard against division by zero)
        if (std::abs(eps_analytic[i]) > 0.0) {
            rel_err_eps[i] = std::abs((mc.mean_eps - eps_analytic[i]) / eps_analytic[i]);
        } else {
            rel_err_eps[i] = 0.0;
        }
        
        if (std::abs(abs_m_analytic[i]) > 0.0) {
            rel_err_abs_m[i] = std::abs((mc.mean_abs_m - abs_m_analytic[i]) / abs_m_analytic[i]);
        } else {
            rel_err_abs_m[i] = 0.0;
        }
        
        // Buffer the trajectories for this temperature and write later
        eps_trajectory[i] = std::move(mc.eps_trajectory);
        mean_eps_trajectory[i] = std::move(mc.mean_eps_trajectory);
    }
    
    // Print out the timing information
    auto t_end = clock_type::now();
    std::chrono::duration<double> elapsed = t_end - t_start;
    std::cout << "Simulation loop wall time (OpenMP-threads: " << omp_get_max_threads() << ") = "
              << elapsed.count() << " s" << std::endl;
    
    if (enable_trajectory_mode) {
        for (int i = 0; i < T_count; i++) {
            for (size_t cycle = 0; cycle < eps_trajectory[i].size(); ++cycle) {
                file << T_values[i] << ","
                     << cycle << ","
                     << eps_trajectory[i][cycle] << ","
                     << mean_eps_trajectory[i][cycle] << "\n";
            }
        }
        file.close();
    }
    else {
        // Dump to CSV (standard output case)
        for (int i = 0; i < T_count; i++) {
            file << T_values[i]          << ","
                 << eps_MC[i]            << ","
                 << eps_analytic[i]      << ","
                 << rel_err_eps[i]       << ","
                 << abs_m_MC[i]          << ","
                 << abs_m_analytic[i]    << ","
                 << rel_err_abs_m[i]     << ","
                 << Cv_MC[i]             << ","
                 << Cv_analytic[i]       << ","
                 << chi_MC[i]            << ","
                 << chi_analytic[i]      << "\n";
        }
        file.close();
    }
    return 0;
}
