#include <iostream>
#include "Eigen/Dense"
using Eigen::MatrixXd;

double rosenBrock(double X){
    printf("rosenbrock\n");
    return X;
}

class PSO
{
public:
    PSO(int n_dim);
    int n_dim;
    
    MatrixXd P; // doubleMatrixXd P(3,6);/double
};

PSO::PSO(int n_dim){
    this -> n_dim=n_dim;
}


