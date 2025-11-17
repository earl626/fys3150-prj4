#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>
#include <string>

using namespace std;

//function declarations
double total_energy(const vector<vector<int>>& spins, int L, double J);
int total_magnetisation(const vector<vector<int>>& spins, int L);
void metropolis_step(vector<vector<int>>& spins, int L, double J, double T,
                     mt19937_64& rng, const vector<double>& w,
                     double& E, int& M);

int main(int argc, char *argv[])
{
    if (argc < 6) {
        cout << "Usage: ./ising_equilibration L T n_cycles ordered(0/1) output_file.csv\n";
        return 1; //for error if not enough arg
    }

    int L = stoi(argv[1]);
    double T = stod(argv[2]);
    long long n_cycles = stoll(argv[3]); //convert third argument to number of cycles
    bool ordered = stoi(argv[4]);
    string outfile = argv[5];

    double J = 1.0; //T = kB*T/J
    double kB = 1.0;
    int N = L * L; //total spins
    double beta = 1.0 / (kB * T);

    //RNG
    random_device rd;
    mt19937_64 rng(rd());
    uniform_real_distribution<double> uniform(0.0, 1.0);

    //initialize spins
    vector<vector<int>> spins(L, vector<int>(L));
    if (ordered) {
        for (int i = 0; i < L; ++i)
            for (int j = 0; j < L; ++j)
                spins[i][j] = 1;
    } else {
        for (int i = 0; i < L; ++i)
            for (int j = 0; j < L; ++j)
                spins[i][j] = (uniform(rng) < 0.5) ? 1 : -1;
    }

    //precompute boltzmann lookup for ΔE = 4J, 8J
    vector<double> w(17, 0.0);
    w[12] = exp(-beta * 4 * J);
    w[16] = exp(-beta * 8 * J);

    double E = total_energy(spins, L, J);
    int M = total_magnetisation(spins, L);

    //file output: cycle, energy per spin, mean energy per spin
    ofstream file(outfile); //open CSV file to write
    file << "cycle,eps,eps_mean\n";

    double E_sum = 0.0; //accumulate total energy over cycles to find running mean

    for (long long cycle = 1; cycle <= n_cycles; ++cycle) { //each iteration = 1 MC cycle
        metropolis_step(spins, L, J, T, rng, w, E, M); //N attempted flips and update
        E_sum += E; //add this cycle total energy to accumulator

        double eps = E / N; //energy per spin for current state
        double eps_mean = E_sum / (cycle * N); //running mean of energy per spin

        file << cycle << "," << eps << "," << eps_mean << "\n";
    }

    file.close();
    cout << "Simulation finished. Data saved to: " << outfile << endl;
    return 0; //great success
}


//function: total energy - hamiltonian
double total_energy(const vector<vector<int>>& spins, int L, double J)
{
    double E = 0.0;
    for (int i = 0; i < L; ++i)
        for (int j = 0; j < L; ++j)
            E -= J * spins[i][j] * (spins[i][(j+1)%L] + spins[(i+1)%L][j]); //periodic boundary conditions

    return E; //total energy (not per spin)
}


//function: total magnetisation
int total_magnetisation(const vector<vector<int>>& spins, int L)
{
    int M = 0;
    for (int i = 0; i < L; ++i)
        for (int j = 0; j < L; ++j)
            M += spins[i][j];

    return M;
}


//function: one cycle (N spin flips)
void metropolis_step(vector<vector<int>>& spins, int L, double J, double T,
                     mt19937_64& rng, const vector<double>& w,
                     double& E, int& M)
{
    uniform_real_distribution<double> uniform(0.0, 1.0);
    uniform_int_distribution<int> randint(0, L-1);

    for (int n = 0; n < L*L; ++n) {
        int i = randint(rng);
        int j = randint(rng);
        int S = spins[i][j];
        int nb = spins[i][(j+1)%L] + spins[i][(j-1+L)%L] +
                 spins[(i+1)%L][j] + spins[(i-1+L)%L][j];

        int dE = 2 * J * S * nb;

        if (dE <= 0 || uniform(rng) <= w[dE + 8]) {
            spins[i][j] = -S;
            E += dE;
            M += -2*S;
        }
    }
}

//outputs CSV file with MC cycle number, energy per spin at that cycle, running average energy per spin up to that cycle
//plot eps and esp_mean vs cycle