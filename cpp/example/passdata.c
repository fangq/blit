#include <stdio.h>
#include "blit_solvers.h"

int main() {
    BLQMRSolver qmr;
    int size = 100;

    ZBLQMRCreate(&qmr, &size);

    ZBLQMRPrint(&qmr);

    qmr.nrhs = 8;
    qmr.maxit = 1000;
    qmr.state = -1;
    qmr.pcond_type = 1;  // Changed from dopcond to pcond_type
    qmr.isquasires = 0;
    qmr.debug = 1;

    ZBLQMRPrint(&qmr);
    return 0;  // Also changed from 1 to 0 (conventional success return)
}