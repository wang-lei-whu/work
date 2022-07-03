#include <iostream>
#include "Eigen/Dense"
using Eigen::MatrixXd;
using namespace Eigen;

class PSO
{
public:
    PSO(MatrixXd (*func)(MatrixXd X), int n_dim, int pop, int max_iter, MatrixXd lb, MatrixXd ub, float w0, float w_min, float c1 = 0.6, float c2 = 0.5, bool verbose=true);
    MatrixXd (*func)(MatrixXd X);

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
    void update_V();
    void update_X();
    MatrixXd cal_Y();
    void update_pbest();
    void update_gbest();
    int run();
};
