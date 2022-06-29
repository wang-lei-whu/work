#include "pso.h"
#include "float.h"
# include "iostream"
#include <ctime>
#define inf DBL_MAX


// PSO类的构造函数
PSO::PSO(MatrixXd(*func)(MatrixXd X), int n_dim, int pop, int max_iter, MatrixXd lb, MatrixXd ub, float w0, float w_min, float c1, float c2, bool verbose)
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
    srand((unsigned)time(NULL)); //时间做随机种子
    X = (MatrixXd::Random(pop,n_dim)).array() *2; //正态分布随机矩阵，需要修改
    V = (MatrixXd::Random(pop,n_dim)).array() *1;
    Y = MatrixXd::Constant(pop,n_dim,inf);
    pbest_x = X;
    Y = cal_Y();
    pbest_y = Y;
    gbest_x = pbest_x.colwise().mean();
    gbest_y = cal_Y(gbest_x);
}

MatrixXd PSO::cal_Y(){
    return func(X);

}
MatrixXd PSO::cal_Y(MatrixXd X1){
    return func(X1);

}

void PSO::update_w(){
    ;
}
int PSO::update_V(){
    ;
}
int PSO::update_X(){
    ;
}

int PSO::update_pbest(){
    ;
}
int PSO::update_gbest(){
    ;
}
int PSO::run(){
    ;
}

MatrixXd testFunc::rosenBrock(MatrixXd X)
{
    printf("rosenbrock\n");
    // std::cout << X << std::endl;
    MatrixXd term1 = X.col(1)-X.col(0);
    // std::cout << term1 << std::endl;
    MatrixXd term2 = X.col(0)-MatrixXd::Constant(X.rows(),1,1);
    return 10*term1.cwiseProduct(term1) + term2.cwiseProduct(term2);
}
