//compiling: g++ -O3 -std=c++17 ising_problem6.cpp -o ising_problem6
//run: ./ising_problem6 20 1.0 200000 50000 10 samples_T1.csv

#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <cmath>
#include <string>
using namespace std;

//energy function
double total_energy(const vector<int>& s, int L) {
    //computes total energy for the lattice
    //using 1D representation: index = i*L + j
    double E = 0.0;

    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {

            int idx = i*L + j; //current site
            int right = i*L + (j+1)%L; //right neighbour
            int down  = ((i+1)%L)*L + j; //down neighbour

            //only count right + down to avoid double counting
            E -= s[idx] * (s[right] + s[down]);
        }
    }
    return E;
}

//magnetisation
int total_M(const vector<int>& s) {
    //just sums all spins
    int M = 0;
    for (int x : s) M += x;
    return M;
}

//onene MC cycle
void metropolis_cycle(vector<int>& s, int L, double beta,
                      mt19937_64& rng, const vector<double>& w,
                      double& E, int& M)
{
    //function does one MC cycle = N attempted flips
    uniform_real_distribution<double> unif(0.0, 1.0);
    uniform_int_distribution<int> rint(0, L-1);
    int N = L * L;

    for (int n = 0; n < N; n++) {

        //pick a random spin
        int i = rint(rng);
        int j = rint(rng);
        int idx = i*L + j;

        int S = s[idx]; //current spin value

        //find nearest neighbours (periodic BC)
        int left  = i*L + (j-1+L)%L;
        int right = i*L + (j+1)%L;
        int up    = ((i-1+L)%L)*L + j;
        int down  = ((i+1)%L)*L + j;

        int nb = s[left] + s[right] + s[up] + s[down];

        //energy change if flipped (J=1)
        int dE = 2 * S * nb;

        //metropolis acceptance rule
        if (dE <= 0 || unif(rng) <= w[dE + 8]) {
            s[idx] = -S; //flip spin
            E += dE; //update total energy
            M += -2*S; //update magnetisation
        }
    }
}

//main
int main(int argc, char** argv) {

    //check for correct number of command line arguments
    if (argc < 7) {
        cerr << "Usage: ./ising_problem6 L T n_cycles n_burn sample_interval outfile.csv\n";
        return 1;
    }

    //read arguments
    int L = stoi(argv[1]); //lattice size
    double T = stod(argv[2]); //temperature
    long long n_cycles = stoll(argv[3]); //total MC cycles
    long long n_burn = stoll(argv[4]); //how many cycles to discard
    int sample_int = stoi(argv[5]); //sample every X cycles
    string outname = argv[6]; //output CSV name

    double beta = 1.0 / T;
    int N = L * L;

    //RNG
    random_device rd;
    mt19937_64 rng(rd());
    uniform_real_distribution<double> u01(0.0, 1.0);

    //create lattice, initialise randomly with +/-1
    vector<int> s(N);
    for (int i = 0; i < N; i++)
        s[i] = (u01(rng) < 0.5 ? 1 : -1);

    //precompute boltzmann factors for ΔE = 4 and 8
    vector<double> w(17, 0.0);
    w[12] = exp(-beta * 4);
    w[16] = exp(-beta * 8);

    //compute initial E, M
    double E = total_energy(s, L);
    int M = total_M(s);
