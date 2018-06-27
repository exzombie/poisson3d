#include <iostream>
#include <cmath>
#include <ctime>
#include <vector>
#include <random>
#include "../erf.h"

using namespace std;

int main()
{
  const uint N = 10000000;
  vector<double> Xs(N);
  mt19937_64 rng;
  uniform_real_distribution<double> dist(-10, -10);
  for (double& x: Xs) {
    x = dist(rng);
  }

  vector<double> Ys(N), Zs(N);
  clock_t start, stop;
  float tY, tZ;

  start = clock();
  for (uint i = 0; i < N; ++i) {
    volatile double tmp = erf(Xs[i]);
    Ys[i] = tmp;
  }
  stop = clock();
  tY = (stop - start) / (float)CLOCKS_PER_SEC;

  start = clock();
  for (uint i = 0; i < N; ++i) {
    volatile double tmp = vdt::fast_erf(Xs[i]);
    Zs[i] = tmp;
  }
  stop = clock();
  tZ = (stop - start) / (float)CLOCKS_PER_SEC;

  double sumdiff = 0;
  for (uint i = 0; i < N; ++i) {
    sumdiff += fabs(Ys[i] - Zs[i]);
  }

  cout << " libm: " << tY * (1.0e9 / N) << " ns" << endl
       << "  vdt: " << tZ * (1.0e9 / N) << " ns" << endl
       << "delta: " << sumdiff / (double)N << endl;

  return 0;
}
