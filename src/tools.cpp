#include "tools.h"
#include <Eigen/Dense>

using namespace std;
MatrixXd toolFunc::rosenBrock(MatrixXd X)
{
    // cout << "rosenbrock\n"
    //      << endl;
    // std::cout << X << std::endl;
    int row =X.rows();
    MatrixXd result=MatrixXd::Zero(row,1);
    MatrixXd term1=MatrixXd::Zero(row,1);
    MatrixXd term2=MatrixXd::Zero(row,1);
    for(int i=0;i<X.cols()-1;i+=2){
    term1 = X.col(i+1) - X.col(i);
    term2 = X.col(i) - MatrixXd::Constant(row, 1, 1);
    result += 10 * term1.cwiseProduct(term1) + term2.cwiseProduct(term2);
    };
    return result;
}

MatrixXd toolFunc::random(int pop, int n_dim, MatrixXd lb, MatrixXd ub)
{
    //"随机初始化一个MatrixXd对象"
    MatrixXd m = MatrixXd::Zero(pop, n_dim);
    default_random_engine e(time(0));
    for (int i = 0; i < n_dim; i++)
    {
        uniform_real_distribution<double> n(lb(0, i), ub(0, i));
        m.col(i) = m.col(i).unaryExpr([&n, &e](double dummy)
                                      { return n(e); });
    };
    return m;
}
