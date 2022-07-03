#include "pso.h"
#include "float.h"
#include "iostream"
#include "cassert"
#include <ctime>
#define inf DBL_MAX
using namespace std;

// PSO类的构造函数
PSO::PSO(MatrixXd (*func)(MatrixXd X), int n_dim, int pop, int max_iter, MatrixXd lb, MatrixXd ub, float w0, float w_min, float c1, float c2, bool verbose)
{
    this->func = func;
    this->n_dim = n_dim; // this指针解决参数和变量同名问题
    this->pop = pop;
    this->max_iter = max_iter;
    this->lb = lb;
    this->ub = ub;
    this->w0 = w0;
    this->w = w0;
    this->w_min = w_min;
    cp = c1;
    cg = c2;
    this->verbose = verbose;
    X = toolFunc::random(pop, n_dim, lb, ub); //初始化X
    MatrixXd v_highest = ub - lb;
    // assert(v_highest.unaryExpr);  //应当检查输入正确性；
    V = MatrixXd::Zero(pop, n_dim);
    V = toolFunc::random(pop, n_dim, -v_highest, v_highest);

    pbest_x = X;
    Y = cal_Y();
    pbest_y = MatrixXd::Constant(pop, 1, inf);
    gbest_x = pbest_x.colwise().mean();
    gbest_y = inf;
}

MatrixXd PSO::cal_Y()
{
    Y = func(X);
    return Y;
}

void PSO::update_w(int iter)
{
    w = w0 - (w0 - w_min) * iter / (max_iter - 1);
}
void PSO::update_V()
{
    // (0,1)之间的随机系数
    MatrixXd r1 = MatrixXd::Random(pop, n_dim).array().abs();
    MatrixXd r2 = MatrixXd::Random(pop, n_dim).array().abs();
    //速度转移公式
    V = w * V + cp * (r1.cwiseProduct(pbest_x - X)) + cg * (r2.cwiseProduct(gbest_x.replicate(pop, 1) - X));
}
void PSO::update_X()
{
    X = X + V;
    X = X.cwiseMin(ub.replicate(pop, 1)).cwiseMax(lb.replicate(pop, 1));
}

void PSO::update_pbest()
{

    ArrayXXd Need_update = (pbest_y.array() > Y.array()).cast<double>();
    //未考虑约束问题
    for (int i = 0; i < n_dim; i++)
    {
        pbest_x.col(i) = (pbest_y.array() > Y.array()).select(X.col(i), pbest_x.col(i));
    };

    pbest_y = (pbest_y.array() > Y.array()).select(Y, pbest_y);
}
void PSO::update_gbest()
{
    MatrixXd::Index maxRow, maxCol;
    MatrixXd::Index minRow, minCol;

    double present_best_y = pbest_y.minCoeff(&minRow, &minCol);
    if (gbest_y > present_best_y)
    {
        gbest_x = X.row(minRow);
        gbest_y = present_best_y;
    };

    // cout<<X<<endl;
    // cout<<Y<<endl;
    // cout<<"\n gbest_x "<<gbest_x<<endl;
    // cout<<"\n gbest_y "<<gbest_y<<endl;
}
int PSO::run()
{
    for (int iter_num = 0; iter_num < max_iter; iter_num++)
    {
        update_V();
        update_w(iter_num);
        update_X();
        cal_Y();
        update_pbest();
        update_gbest();
        // cout << "X in step" << iter_num << ": \n"
        //      << X << "\n"
        //      << endl;
    }
    return 0;
}

MatrixXd toolFunc::rosenBrock(MatrixXd X)
{
    cout << "rosenbrock\n"
         << endl;
    // std::cout << X << std::endl;
    MatrixXd term1 = X.col(1) - X.col(0);
    // std::cout << term1 << std::endl;
    MatrixXd term2 = X.col(0) - MatrixXd::Constant(X.rows(), 1, 1);
    return 10 * term1.cwiseProduct(term1) + term2.cwiseProduct(term2);
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
