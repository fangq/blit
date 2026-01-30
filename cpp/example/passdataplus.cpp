#include <stdio.h>
#include "blit_solvers.h"

int main() {
    BlitBLQMR<double> solver(9527, 8);
    solver.Print();
    return 1;
}
