#include <stdio.h>
#include "blit_solvers.h"

int main() {
    const int nsize = 5, nz = 12, nrhs = 2;

    int Ap[nsize + 1] = {1, 3, 6, 10, 11, 13};
    int Ai[nz] = {1, 2, 1, 3, 5, 2, 3, 4, 5, 3, 2, 5};
    double Ax[nz] = {2., 3., 3., -1., 4., 4., -3., 1., 2., 2., 6., 1.};
    double b[nrhs][nsize] = {{8.0,  45.000,  -3.000,   3.000,  19.000},
        {18.0,  45.000,  -3.000,   3.000,  19.000}
    };
    double x[nrhs][nsize] = {0};
    int nzz = nz;

    BlitBLQMR<double> solver(nsize, nrhs, 100, 1e-7);
    solver.Prepare(Ap, Ai, Ax, nzz);
    solver.Solve((double*) & (x[0][0]), (double*) & (b[0][0]), 2);
    solver.Print();
    return 1;
}
