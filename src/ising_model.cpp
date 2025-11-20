
#include "ising_model.hpp"

using namespace std;

void IsingModel::initialize(bool ordered) {
    initialize_spins(ordered);
    compute_total_magnetization();
    compute_total_energy();
}

void IsingModel::set_temperature(double T_) {
    T = T_;
    beta = 1.0 / T;
    precompute_boltzmann();
}

int IsingModel::delta_energy(int i, int j) const {
    int s_ij = spins(i, j);
    
    int up    = spins(pbc(i - 1), j);
    int down  = spins(pbc(i + 1), j);
    int left  = spins(i, pbc(j - 1));
    int right = spins(i, pbc(j + 1));
    
    int neighbour_sum = up + down + left + right;
    
    return 2 * s_ij * neighbour_sum; // integer in {-8,-4,0,4,8}
}

bool IsingModel::metropolis_update() {
    // Choose a random lattice site
    int i = dist_index(rng);
    int j = dist_index(rng);
    
    // Computing delta_E
    int s_ij = spins(i, j);
    int dE = delta_energy(i, j);
    
    if (dE <= 0) {
        // Always accept if energy goes down
        accept_flip(i, j, s_ij, dE);
        return true;
    } else {
        // Metropolis criterion
        double w = boltzmann[dE + 8]; // index shift: -8..8 -> 0..16
        double r = dist_real(rng);
        if (r < w) {
            accept_flip(i, j, s_ij, dE);
            return true;
        } else {
            return false;
        }
    }
}

void IsingModel::initialize_spins(bool ordered) {
    if (ordered) {
        spins.fill(1);  // all spins up
    } else {
        uniform_int_distribution<int> dist_spin(0, 1);
        for (int i = 0; i < L; i++) {
            for (int j = 0; j < L; j++) {
                int r = dist_spin(rng);  // r in {0, 1}
                spins(i, j) = (r*2) - 1; // spin in {-1, 1}
            }
        }
    }
}

inline int IsingModel::pbc(int idx) const {
    // periodic boundary conditions
    return (idx + L) % L;
}

void IsingModel::compute_total_magnetization() {
    M = 0.0;
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            M += spins(i, j);
        }
    }
}

void IsingModel::compute_total_energy() {
    E = 0.0;
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            int s = spins(i, j);
            int s_right = spins(i, pbc(j + 1));
            int s_down  = spins(pbc(i + 1), j);
            E -= s * (s_right + s_down);
        }
    }
}

void IsingModel::precompute_boltzmann() {
    boltzmann.fill(0.0);
    
    // Only 5 possible dE values:
    const int possible_dE[5] = {-8, -4, 0, 4, 8};
    for (int k = 0; k < 5; k++) {
        int dE = possible_dE[k];
        boltzmann[dE + 8] = exp(-beta * dE);
    }
}

inline void IsingModel::accept_flip(int i, int j, int s_ij, double dE) {
    // Flip spin
    spins(i, j) = -s_ij;
    
    // Update total energy and magnetization
    E += dE;
    M += -2.0 * s_ij;
}
