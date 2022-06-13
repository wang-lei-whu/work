#include <iostream>
#include "Eigen/Dense"
using Eigen::MatrixXd;

class testFunc
{
public:
    static MatrixXd rosenBrock(MatrixXd X);
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
    MatrixXd gbest_y;    

    

    MatrixXd X;
    MatrixXd Y;
    MatrixXd V;
    void update_w();
    int update_V();
    int update_X();
    MatrixXd cal_Y();
    MatrixXd cal_Y(MatrixXd X);
    int update_pbest();
    int update_gbest();
    int run();
};
