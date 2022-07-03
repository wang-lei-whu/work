#include "pso.h"
#include <ctime>

using namespace std;


int main()
{
     // srand((unsigned)time(NULL)); //时间做随机种子
     // MatrixXd m = (MatrixXd::Random(2, 1)).array().abs() * 2;

     //
     int dim = 2;
     MatrixXd lb = MatrixXd::Constant(1, dim, -2);
     // lb(1, 0) = -4;
     MatrixXd ub = MatrixXd::Constant(1, dim, 2);
     // ub(1, 0) = 4;     
     
     MatrixXd m = 1.5 * toolFunc::random(5, dim,lb, ub);
     MatrixXd m1= MatrixXd::Constant(5, dim, 1);
     MatrixXd::Index maxRow, maxCol;
	MatrixXd::Index minRow, minCol;
     double min = m.minCoeff(&minRow,&minCol);
     cout <<"m\n"<< m <<"\n"<< endl;
     cout <<"min\n"<< min <<"\n"<< endl;
     cout <<"min x1\n"<< minRow <<"\n min x2\n"<< minCol <<"\n"<< endl;
     m.row(0)=MatrixXd::Constant(1, dim, 0);
     cout <<"m1\n"<< m <<"\n"<< endl;
     ArrayXXd m2 = (m.array() < m1.array()).cast<double>();
     cout <<"m2\n"<< m2 <<"\n"<< endl;
     cout<<m2.select(m,m1)<<endl;
     // MatrixXd m2 = m.cwiseMin(ub.replicate(5,1)).cwiseMax(lb.replicate(5,1));
     // cout << m <<"\n"<< endl;
     // cout << m2 << endl;
     // MatrixXd m1 = MatrixXd::Zero(5,dim);
     //cout << 1*m.cwiseProduct(m1) << endl; //点乘
     // MatrixXd m = MatrixXd::Zero(10, dim);
     // m = toolFunc::random(10, dim,lb, ub);
     // cout << m << endl;

     // m = i;
     // cout << m << endl;
     //     cout << m << endl;
     // for (int i = 0; i < lb.cols() ; i++)
     // {
     //     normal_distribution<double> n(lb(i), ub(i));
     // 	m.row(i).unaryExpr([n,e](double dummy){return n(e);});
     //     cout<< n(e)<<endl;
     // }
     // cout << "Gaussian random matrix row mean:\n"
     //  << m.rowwise().mean() << endl;
     // cout << "Mean: " << lb.mean() << endl;
     // MatrixXd m2 = (m.array() - m.mean()) * (m.array() - m.mean());
     // cout << "std: " << sqrt(m2.sum() / (m2.size() - 1)) << endl;


     PSO pso(toolFunc::rosenBrock, dim, 500, 100, lb, ub, 0.8, 0.1, 0.6, 0.5, true);
     pso.run();
     cout << "pso.gbest_y:\n"
          << pso.gbest_y << endl;
     cout << "pso.gbest_x:\n"
          << pso.gbest_x << endl;
     return 0;
}
