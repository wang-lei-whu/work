#include <iostream>
#include "random"
#include "Eigen/Dense"
using Eigen::MatrixXd;

class toolFunc
{
public:
    static MatrixXd rosenBrock(MatrixXd X);
    static MatrixXd random(int pop, int n_dim, MatrixXd lb, MatrixXd ub);
};

class PSO
{
public:
    PSO(MatrixXd(*func)(MatrixXd X), int n_dim, int pop, int max_iter, MatrixXd lb, MatrixXd ub, float w0, float w_min, float c1, float c2, bool verbose);
    MatrixXd(*func)(MatrixXd X);
    int n_dim;
    int pop;
    int max_iter;
    MatrixXd lb;
    MatrixXd ub;
    float w;
    float w0;
    float w_min;
    float cp;
    float cg;
    bool verbose;

    MatrixXd pbest_x;
    MatrixXd pbest_y;
    MatrixXd gbest_x;
    double gbest_y;    

    

    MatrixXd X;
    MatrixXd Y;
    MatrixXd V;
    void update_w(int _iter);
    int update_V();
    int update_X();
    MatrixXd cal_Y();
    int update_pbest();
    int update_gbest();
    int run();
};
