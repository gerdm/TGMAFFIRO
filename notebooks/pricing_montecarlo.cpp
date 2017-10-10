#include <random>
#include <iostream>
#include <algorithm>
#include <math.h>

//<<Return a random sample from a standard normal deviation>>
// Taken from https://stackoverflow.com/questions/38244877/how-to-use-stdnormal-distribution
// --------------------------------------------------------------------------------------
// random device to initialize the seed
std::random_device rd;
// Mersenne Twister pseudorandom number generator initialized with
// the previous given seed https://en.wikipedia.org/wiki/Mersenne_Twister
std::mt19937 gen(rd());

double normal_sample()
{
    double sample;
    std::normal_distribution<double> d(0, 1);
    return d(gen);
}

double simul_payoff(double S0, double K, double T, double r, double sigma)
{
    //Simulate one instance of the payoff of a European call option
    // assuming the underlying behaves as the following Ito process
    // under a risk-neutral probability measure Q:
    //             dSt = St(r dt + sigma dWt)
    double WT, ST, payoff;
    WT = normal_sample() * sqrt(T);
    ST = S0 * exp((r - sigma * sigma / 2) * T + sigma * WT);
    payoff = std::max(ST - K, 0.0);
    
    return payoff;
}

double get_mean_price(double S0, double K, double T, double r, double sigma, int nsimul)
{
    // Compute the average price of an option by means of simulating it nsimul times
    // and computing its average. Finally, discount it to get the avg_price
    double total_price = 0.0;
    for (int i=0; i < nsimul; ++i){
        total_price += simul_payoff(S0, K, T, r, sigma);
    }
    return total_price / nsimul;
}

int main()
{
    double price;
    double S0, K, T, r, sigma;
    int nsimul;

    S0 = 100;
    K = 92;
    T = 150.0 / 365;
    r = 0.06;
    sigma = 0.23;
    
    nsimul = 1500;
    for (int i=0; i < nsimul; ++i){
        price += get_mean_price(S0, K, T, r, sigma, nsimul);
    }
    
    price = exp(-r * T) * price / nsimul;
    std::cout << price << std::endl;
    
    return 0;
}
