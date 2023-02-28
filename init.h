#ifndef _INIT_H_
#define _INIT_H_

#include <iostream>
#include <random>

using namespace std;

template <typename T>
T random_normal(float mean, float stddev) 
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<double> dist(mean, stddev);
  return dist(gen);
}

#endif