#include "Eigen/Dense"
#include "random"

using Eigen::MatrixXd;

class toolFunc
{
public:
    static MatrixXd rosenBrock(MatrixXd X);
    static MatrixXd random(int pop, int n_dim, MatrixXd lb, MatrixXd ub);
};
