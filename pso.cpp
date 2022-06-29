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
    X = toolFunc::random(pop,n_dim, lb, ub); //初始化X
    MatrixXd v_highest=ub-lb;
    // assert(v_highest.unaryExpr);  //应当检查输入正确性；
    V = MatrixXd::Zero(pop, n_dim);
    V = toolFunc::random(pop,n_dim, -v_highest, v_highest);
    // Y = MatrixXd::Constant(pop, n_dim, inf);
    pbest_x = X;
    Y = cal_Y();
    pbest_y = MatrixXd::Constant(pop, 1, inf);
    gbest_x = pbest_x.colwise().mean();
    gbest_y = inf;
}

MatrixXd PSO::cal_Y()
{
    return func(X);
}

void PSO::update_w(int iter)
{
    w = w0 - (w0 - w_min) * iter / max_iter;
}
int PSO::update_V()
{
    cout<<"V:\n"<<V<<endl;
    MatrixXd r1 = MatrixXd::Random(pop,n_dim).array().abs();
    cout<<"r1:\n"<<r1<<endl;
    MatrixXd r2 = MatrixXd::Random(pop,n_dim).array().abs();
    cout<<"r2:\n"<<r2<<endl;
    cout<<"r1.cwiseProduct(gbest_x-X)\n"<<MatrixXd::Constant(pop,n_dim,cg)-X<<endl;
    V = w * V+cp*(r1.cwiseProduct(pbest_x-X))+cg*(r2.cwiseProduct(MatrixXd::Constant(pop,n_dim,cg)-X)); //速度转移公式
    cout<<"updated V:\n"<<V<<endl;
    return 0;
}
int PSO::update_X()
{
    return 0;
}

int PSO::update_pbest()
{
    return 0;
}
int PSO::update_gbest()
{
    return 0;
}
int PSO::run()
{
    update_V();
    return 0;
}

MatrixXd toolFunc::rosenBrock(MatrixXd X)
{
    cout << "rosenbrock\n"<< endl;
    // std::cout << X << std::endl;
    MatrixXd term1 = X.col(1) - X.col(0);
    // std::cout << term1 << std::endl;
    MatrixXd term2 = X.col(0) - MatrixXd::Constant(X.rows(), 1, 1);
    return 10 * term1.cwiseProduct(term1) + term2.cwiseProduct(term2);
}

MatrixXd toolFunc::random(int pop, int n_dim, MatrixXd lb, MatrixXd ub)
{
    MatrixXd m = MatrixXd::Zero(pop, n_dim);
    default_random_engine e(time(0));
    for (int i = 0; i < n_dim; i++)
    {
        uniform_real_distribution<double> n(lb(i, 0), ub(i, 0));
        m.col(i) = m.col(i).unaryExpr([&n, &e](double dummy)
                                      { return n(e); });
    };
    return m;
}
