//g++ -O3 -std=c++17 ising.cpp -o ising
//./ising 2 1.0 1000000 1


#include <iostream>
#include <fstream>
#include <iomanip>
#include <random>
#include <vector>
#include <cmath>
#include <string>

using namespace std;

//function declatarions
double total_energy(const vector<vector<int>>& spins, int L, double J); //returns the total energy E of the lattice
int total_magnetisation(const vector<vector<int>>& spins, int L); //returns total magnetisation M
void metropolis_step(vector<vector<int>>& spins, int L, double J, //performs a single monte carlo cycle
    double T, mt19937_64& rng, const vector<double>& w,
    double& E, int& M); //modifies spins & updates E and M in place

//main
int main(int argc, char* argv[])
{
    if (argc < 5) {
        cout << "Usage: ./ising L T n_cycles ordered(0/1)" << endl;
        return 1;
    }

//parameters
int L = stoi(argv[1]); //lattice size
double T = stod(argv[2]); //temperature (J/k_B units)
int n_cycles = stoi(argv[3]); //monte carlo cycles
bool ordered = stoi(argv[4]); //1 = all spins up, 0 = random
double J = 1.0;
double kB = 1.0;
int N = L * L;
double beta = 1.0 / (kB * T); //inverse temperature

//random number generator
random_device rd;
mt19937_64 rng(rd());
uniform_real_distribution<double> uniform(0.0, 1.0); //generates uniform real numbers in [0,1]
uniform_int_distribution<int> randint(0, L-1); //generates ints in [0, L-1]

//initialise lattice
vector<vector<int>> spins(L, vector<int>(L));
    if (ordered) { //if ordered: all spins set to +1
        for (int i=0; i<L; ++i)
            for (int j=0; j<L; ++j)
                spins[i][j] = 1;
    } else { //else: random spins ±1 with equal probability using the uniform rng
        for (int i=0; i<L; ++i)
            for (int j=0; j<L; ++j)
                spins[i][j] = (uniform(rng) < 0.5) ? 1 : -1;
    }

//precompute boltzmann factors (ΔE = 4J, 8J)
vector<double> w(17, 0.0); //index by ΔE + 8
//precompute boltzmann acceptance factors for positive ΔE values to avoid exp() in inner loop
//size 17 because it goes from 0,...,16
//w entries for other indices remain 0.0
    w[12] = exp(-beta * 4 * J); 
    w[16] = exp(-beta * 8 * J);

//initial total energy and magnetisation
double E = total_energy(spins, L, J); //for starting configuration
int M = total_magnetisation(spins, L);

//averages
double E_sum = 0, E2_sum = 0; //accumulates E and E*E each cycle
double M_sum = 0, Mabs_sum = 0, M2_sum = 0; //accumulates M and |M| and M*M

    //monte carlo loop
    for (int cycle = 0; cycle < n_cycles; ++cycle) {
        metropolis_step(spins, L, J, T, rng, w, E, M); //attempts N=L*L flips and updates spins, E, M
        E_sum    += E; //add current E, E^2, M, |M|, M^2 to accumulators
        E2_sum   += E*E;
        M_sum    += M;
        Mabs_sum += fabs(M);
        M2_sum   += M*M;
    }

double norm = 1.0 / n_cycles; 
double E_avg = E_sum * norm / N; //average total energy/energy per spin
double Mabs_avg = Mabs_sum * norm / N; //average absolute magnetisation per spin
double E2_avg = E2_sum * norm / (N*N); //energy per spin squared
double M2_avg = M2_sum * norm / (N*N); //magnetisation per spin squared

double Cv = (E2_avg - E_avg*E_avg) / (kB * T * T); //heat capacity per spin
double Chi = (M2_avg - Mabs_avg*Mabs_avg) / (kB * T); //susceptibility per spin
//these are based on the equations given in the assignment

    cout << fixed << setprecision(6); //results with 6 decimal places
    cout << "L=" << L << "  T=" << T << "  Cycles=" << n_cycles << endl;
    cout << "<E>/N       = " << E_avg << endl;
    cout << "<|M|>/N     = " << Mabs_avg << endl;
    cout << "Cv/N        = " << Cv << endl;
    cout << "Chi/N       = " << Chi << endl;

return 0;
}

//function definitions

//total energy with periodic boundary conditions
//computes hamiltonian
double total_energy(const vector<vector<int>>& spins, int L, double J)
{
    double E = 0.0;
    for (int i=0; i<L; ++i) {
        for (int j=0; j<L; ++j) {
            int S = spins[i][j];
            int right = spins[i][(j+1)%L]; //avoid double counting, periodic boundary conditions
            int down  = spins[(i+1)%L][j];
            E -= J * S * (right + down);
        }
    }
    return E; //returns total energy (not per spin)
}

//total magnetisation
//sums all spins to get total magnetisation
int total_magnetisation(const vector<vector<int>>& spins, int L)
{
    int M = 0;
    for (int i=0; i<L; ++i)
        for (int j=0; j<L; ++j)
            M += spins[i][j];
    return M;
}

//one monte carlo cycle (N attempted flips)
void metropolis_step(vector<vector<int>>& spins, int L, double J,
                     double T, mt19937_64& rng, const vector<double>& w,
                     double& E, int& M)
{
    uniform_real_distribution<double> uniform(0.0, 1.0);
    uniform_int_distribution<int> randint(0, L-1);

    for (int n=0; n<L*L; ++n) {
        int i = randint(rng); //choose a random lattice site (i,j) uniformly
        int j = randint(rng);
        int S = spins[i][j]; //current spin
        int nb = spins[i][(j+1)%L] + spins[i][(j-1+L)%L] + //sum of the four nearest neighbours using PBC
                 spins[(i+1)%L][j] + spins[(i-1+L)%L][j];
        int dE = 2 * J * S * nb; //compute energy change

        if (dE <= 0 || uniform(rng) <= w[dE + 8]) { //if dE </= 0: flipping reduces or leaves energy unchanged -> accept unconditionally, else -> accept with probability
            spins[i][j] = -S; //if accepted flip spin
            E += dE; //update
            M += -2 * S; //update
        }
    }
}
//outputs: <E>/N, <|M|>/N, Cv/N, Chi/N
//for 4b set L=2 and T=1.0
//for 4c vary n_cycles