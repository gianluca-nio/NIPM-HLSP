#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>

#include <nipm_hlsp/nipm_hlsp.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;
typedef Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1> >::Index Index;

int main()
{
    int nVar = 2;
    int p = 3;
    VectorXi me = VectorXi::Zero(p);
    VectorXi mi = VectorXi::Zero(p);
    mi << 2, 2, 0;
    me << 0, 0, 2;

    int m_all = 0;
    VectorXi ml = VectorXi::Zero(p + 1);
    for (int l = 0; l < p; l++)
    {
        ml[l + 1] = ml[l] + me[l] + mi[l];
        m_all += me[l] + mi[l];
    }

    MatrixXd A = MatrixXd::Zero(m_all, nVar);
    VectorXd b = VectorXd::Zero(m_all);
    A << -0.1, 1, -1, 1, 1, 0, 1, 1, 1, 0, 0, 1;
    b << 0.55, -1.5, 2.5, 2, 0, 0;

    nipmhlsp::NIpmHLSP solver(p, nVar);
    for (Index l = 0; l < p; l++)
        solver.setData(l, A.middleRows(ml[l], me[l]), b.segment(ml[l], me[l]), A.middleRows(ml[l] + me[l], mi[l]), b.segment(ml[l] + me[l], mi[l]));
    solver.solve();

    std::cout << "=============== Test with " << nVar << " variables and " << p << " levels finished with KKT " << solver.KKT << " in " << solver.iter << " iterations and " << solver.time << " [s] with primal x: " << solver.get_x().transpose() << std::endl;
    if (solver.KKT > 1e-3)
    {
        cout << "ERROR: KKT norm too high, something's wrong" << endl;
        throw;
    }

    return 0;
}
