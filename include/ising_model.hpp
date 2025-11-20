#if !defined(PENNING_TRAP_H)

#include <armadillo>
#include <random>
#include <array>
#include <cmath>

/**
 * @brief 2D square-lattice Ising model with Metropolis dynamics.
 *
 * The Hamiltonian is
 *   E = -J * sum_<⟨ij⟩> s_i s_j
 * with nearest-neighbour interactions on a periodic L×L lattice,
 * and spins s_i ∈ {−1, +1}.
 *
 * We work in units where (J = 1) and (k_B = 1),
 * so energy is in units of (J) and temperature is in units of (J/k_B).
 *
 * This class supports:
 *  - Initialising ordered or random spin configurations
 *  - Computing total energy E and energy per spin
 *  - Computing total magnetisation M and magnetisation per spin
 *  - Computing local ΔE for a proposed spin flip
 *  - Performing one Metropolis single-spin update
 *
 * The implementation uses:
 *  - arma::Mat<int> to store the spins
 *  - std::mt19937 as the random number generator
 *
 */
class IsingModel {
public:
    /**
     * @brief Construct an Ising model on an L×L lattice.
     *
     * @param L_       Linear system size (lattice has L×L spins).
     * @param T_       Temperature T in units where k_B = 1.
     * @param ordered  If true, initialise all spins to +1.
     *                 If false, initialise spins randomly to ±1.
     * @param seed     Seed for the internal RNG (std::mt19937).
     */
    IsingModel(int L_,
               double T_,
               bool ordered = false,
               unsigned int seed = std::random_device{}())
            : L(L_),
              N(L_ * L_),
              T(T_),
              beta(1.0 / T_),
              spins(L_, L_),
              rng(seed),
              dist_index(0, L_ - 1),
              dist_real(0.0, 1.0)
    {
        initialize_spins(ordered);
        compute_total_magnetization();
        compute_total_energy();
        precompute_boltzmann();
    }
    
    /**
     * @brief Reinitialise the spin configuration.
     *
     * Resets the spins to a new configuration (ordered or random),
     * and recomputes total energy and magnetisation.
     *
     * @param ordered  If true, all spins are set to +1.
     *                 If false, spins are initialised randomly to ±1.
     */
    void initialize(bool ordered = false);
    
    /**
     * @brief Set the temperature T and recompute Boltzmann factors.
     *
     * @param T_ New temperature (k_B = 1).
     */
    void set_temperature(double T_);
    
    /**
     * @brief Get the current total energy E.
     *
     * @return Total energy of the system.
     */
    inline double total_energy() const { return E; }
    
    /**
     * @brief Get the energy per spin ε = E / N.
     *
     * @return Energy per spin.
     */
    inline double energy_per_spin() const { return E / N; }
    
    /**
     * @brief Get the total magnetisation M = sum_i s_i.
     *
     * @return Total magnetisation of the system.
     */
    inline double magnetization() const { return M; }
    
    /**
     * @brief Get the magnetisation per spin m = M / N.
     *
     * @return Magnetisation per spin.
     */
    inline double magnetization_per_spin() const { return M / N; }
    
    /**
     * @brief Compute ΔE for flipping the spin at (i, j).
     *
     * @param i Row index (0 ≤ i < L).
     * @param j Column index (0 ≤ j < L).
     * @return Integer ΔE ∈ {−8, −4, 0, 4, 8}.
     */
    int delta_energy(int i, int j) const;
    
    /**
     * @brief Perform one Metropolis single-spin update.
     *
     *  - A random lattice site (i, j) is chosen uniformly.
     *  - The energy difference ΔE for flipping that spin is computed.
     *  - If ΔE ≤ 0, the flip is always accepted.
     *  - If ΔE > 0, the flip is accepted with probability exp(−ΔE / T).
     *
     * On acceptance, the internal energy E and magnetisation M are updated.
     *
     * @return true if the proposed spin flip was accepted, false otherwise.
     */
    bool metropolis_update();
    
    /**
     * @brief Read-only access to the spin configuration.
     *
     * @return Const reference to the underlying Armadillo matrix
     *         of spins, with entries ±1.
     */
    inline const arma::Mat<int>& get_spins() const { return spins; }
    
private:
    // --- Model parameters ---
    
    int L;        // Linear lattice size (L×L system).
    int N;        // Total number of spins (N = L * L).
    double T;     // Temperature (k_B = 1).
    double beta;  // Inverse temperature β = 1 / T.
    
    // --- State variables ---
    
    arma::Mat<int> spins;  // Spin configuration, entries ±1.
    
    double E = 0.0;  // Current total energy.
    double M = 0.0;  // Current total magnetisation.
    
    // --- Random number generation ---
    
    std::mt19937 rng;                                 // Mersenne Twister RNG.
    std::uniform_int_distribution<int> dist_index;    // Uniform index [0, L-1].
    std::uniform_real_distribution<double> dist_real; // Uniform real [0, 1).
    
    /**
     * @brief Precomputed Boltzmann weights for possible ΔE values.
     *
     * For a 2D Ising model with nearest neighbours, ΔE ∈ {−8, −4, 0, 4, 8}.
     * These are stored at indices [ΔE_scaled + 8] to map −8..8 → 0..16.
     * Unused entries remain 0.
     */
    std::array<double, 17> boltzmann{};
    
    /**
     * @brief (Re)initialise spins, either ordered or random.
     *
     * @param ordered If true, set all spins to +1, otherwise random ±1.
     */
    void initialize_spins(bool ordered);
    
    /**
     * @brief Apply periodic boundary conditions to a lattice index.
     *
     * @param idx Index (can be outside [0, L−1]).
     * @return Wrapped index in [0, L−1].
     */
    int pbc(int idx) const;
    
    /**
     * @brief Recompute the total magnetisation M = sum_i s_i.
     */
    void compute_total_magnetization();
    
    /**
     * @brief Recompute the total energy E.
     *
     * Uses periodic boundary conditions and counts each bond once by
     * summing interactions only with the right and down neighbours.
     * For L = 2 this effectively double-counts
     */
    void compute_total_energy();
    
    /**
     * @brief Precompute Boltzmann weights exp(−ΔE / T) for allowed ΔE.
     *
     * Only the possible ΔE ∈ {−8, −4, 0, 4, 8} are filled in.
     */
    void precompute_boltzmann();
    
    /**
     * @brief Accept a proposed spin flip at (i, j) and update E and M.
     *
     * @param i    Row index of flipped spin.
     * @param j    Column index of flipped spin.
     * @param s_ij Old spin value before flip (±1).
     * @param dE   Energy change ΔE associated with the flip.
     */
    void accept_flip(int i, int j, int s_ij, double dE);
};

#define PENNING_TRAP_H
#endif
