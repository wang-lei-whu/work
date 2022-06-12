#include <iostream>
#include "Eigen/Dense"
using Eigen::MatrixXd;

class testFunc{
   public:
    static double rosenBrock(double X);
};

class PSO
{
public:
    PSO(int n_dim);
    int n_dim;
    
    MatrixXd P; // doubleMatrixXd P(3,6);/double
};




