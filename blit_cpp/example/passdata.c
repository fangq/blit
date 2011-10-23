#include <stdio.h>
#include "blit_solvers.h"

int main(){
	BLQMRSolver qmr;
        int size=100;

        ZBLQMRCreate(&qmr,&size);

        ZBLQMRPrint(&qmr);

	qmr.nrhs=8;
	qmr.maxit=1000;
	qmr.state=-1;
	qmr.dopcond=1;
	qmr.isquasires=0;
	qmr.debug=1;

	ZBLQMRPrint(&qmr);
	return 1;
}
