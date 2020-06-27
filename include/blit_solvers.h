#ifndef _BLIT_SOLVERS_C_H
#define _BLIT_SOLVERS_C_H

#include <stdio.h>
#include <string.h>

#ifdef  __cplusplus
extern "C" {
#endif

#define BLIT_NULL       0
#define MTYPEID_REAL    1
#define MTYPEID_COMPLEX 2

enum BlitError {beTypeMismatch=-999,beNoLHS=-998,beNoRHS=-997};

typedef struct Blit_ILUPcond{
    int n, nz, status, iscomplex;
    int numeric, symbolic;
    double control[20];
} ILUPcond;

typedef struct Blit_BLQMRSolver{
    int n, nrhs, maxit, state, dopcond, flag, iter, isquasires, debug;
    double qtol, droptol, res, relres;
    ILUPcond *ilu;
} BLQMRSolver;

typedef struct Blit_F90Complex{
    double x,y;
} F90Complex;

#define DBLQMRCreate    __blit_blqmr_real_MOD_blqmroncreate
#define ZBLQMRCreate    __blit_blqmr_complex_MOD_blqmroncreate
#define DBLQMRPrep      __blit_blqmr_real_MOD_blqmrprep
#define ZBLQMRPrep      __blit_blqmr_complex_MOD_blqmrprep
#define DBLQMRSolve     __blit_blqmr_real_MOD_blqmrsolve
#define ZBLQMRSolve     __blit_blqmr_complex_MOD_blqmrsolve
#define DBLQMRDestroy   __blit_blqmr_real_MOD_blqmrondestroy
#define ZBLQMRDestroy   __blit_blqmr_complex_MOD_blqmrondestroy
#define DBLQMRPrint     __blit_blqmr_real_MOD_blqmrprint
#define ZBLQMRPrint     __blit_blqmr_complex_MOD_blqmrprint

#define ILUPcondCreate  __blit_ilupcond_MOD_ilupcondcreate
#define ILUPcondPrep    __blit_ilupcond_MOD_ilupcondprep
#define ILUPcondSolve   __blit_ilupcond_MOD_ilupcondsolve
#define ILUPcondDestroy __blit_ilupcond_MOD_ilupconddestroy

extern void DBLQMRCreate(BLQMRSolver *qmr,int *n);
extern void ZBLQMRCreate(BLQMRSolver *qmr,int *n);
extern void DBLQMRPrep(BLQMRSolver *qmr, int *Ap, int *Ai, double *Ax, int *nz);
extern void ZBLQMRPrep(BLQMRSolver *qmr, int *Ap, int *Ai, F90Complex *Ax, int *nz);
extern void DBLQMRSolve(BLQMRSolver *qmr,int *Ap, int *Ai, double *Ax, int *nz, double *x, double *b, int *nrhs);
extern void ZBLQMRSolve(BLQMRSolver *qmr,int *Ap, int *Ai, F90Complex *Ax, int *nz, F90Complex *x, F90Complex *b, int *nrhs);
extern void DBLQMRDestroy(BLQMRSolver *qmr);
extern void ZBLQMRDestroy(BLQMRSolver *qmr);
extern void DBLQMRPrint(BLQMRSolver *qmr);
extern void ZBLQMRPrint(BLQMRSolver *qmr);

extern void ILUPcondCreate(ILUPcond *ilu,int *n,int *nz);
extern void ILUPcondPrep(ILUPcond *ilu,int *Ap, int *Ai, double *Ax, double *droptol, double *Az);
extern void ILUPcondSolve(ILUPcond *ilu,int *Ap,int *Ai, double *Ax,int *rows,int *cols,
                      double *x,double *b,double *Az,double *xz,double *bz);
extern void ILUPcondDestroy(ILUPcond *ilu);


#endif

#ifdef  __cplusplus
}
#endif

#ifdef  __cplusplus

template <class T>
class BlitILU{

  private:
    ILUPcond ilu;
    int    *Ap, *Ai;
    double *Ax, *Az;
    int nz;

  public:
    BlitILU(int n, int nz){
        Ap=BLIT_NULL;Ai=BLIT_NULL;Ax=BLIT_NULL;Az=BLIT_NULL;nz=0;
    	ILUPcondCreate(&ilu,&n,&nz);
    }
   ~BlitILU(){
   	ILUPcondDestroy(&ilu);
    }
    void Run(int **App, int **Aii, double **Axx, double droptol,double **Azz=BLIT_NULL){
        Ap=*App;Ai=*Aii;Ax=*Axx;
	if(Azz!=BLIT_NULL) Az=*Azz;
	ILUPcondPrep(&ilu,Ap,Ai,Ax,&droptol,Az);
    }
    void Solve(int nrow, int ncol, double *x, double *b, double *xz=BLIT_NULL, double *bz=BLIT_NULL){
    	if(nz==0 || Ap==0) throw(beNoLHS);
    	if(x==0  || b==0)  throw(beNoRHS);
	ILUPcondSolve(&ilu,Ap,Ai,Ax,&nrow,&ncol,x,b,Az,xz,bz);
    }
};

template <class T>
class BlitBLQMR{

  private:
    BLQMRSolver qmr;
    int *Ap, *Ai;
    T   *Ax;
    int nz;

  public:
    BlitBLQMR(int n){
        Ap=BLIT_NULL;Ai=BLIT_NULL;Ax=BLIT_NULL;nz=0;
	if(sizeof(T)==sizeof(double)) 
		DBLQMRCreate(&qmr,&n);
	else if(sizeof(T)==sizeof(F90Complex))
		ZBLQMRCreate(&qmr,&n);
	else
		throw beTypeMismatch;
    }
    BlitBLQMR(int n,int nrhs,int maxit=100,double droptol=1e-3,int isquasires=0,int debug=1){
        Ap=BLIT_NULL;Ai=BLIT_NULL;Ax=BLIT_NULL;nz=0;
        if(sizeof(T)==sizeof(double)) 
	        DBLQMRCreate(&qmr,&qmr.n);
        else if(sizeof(T)==sizeof(F90Complex))
		ZBLQMRCreate(&qmr,&qmr.n);
	else
                throw beTypeMismatch;
        qmr.n=n;qmr.nrhs=nrhs;qmr.maxit=maxit;
        qmr.isquasires=isquasires;qmr.debug=debug;
        qmr.droptol=droptol;qmr.dopcond=(droptol>=0.0);
    }
   ~BlitBLQMR(){
        if(sizeof(T)==sizeof(double)) 
	        DBLQMRDestroy(&qmr);
        else if(sizeof(T)==sizeof(F90Complex))
                ZBLQMRDestroy(&qmr);
        else
                throw beTypeMismatch;
        if(Ap) delete [] Ap;
        if(Ai) delete [] Ai;
        if(Ax) delete [] Ax;
    }
    void Prepare(int *App, int *Aii, T *Axx, int nzz){
        if(Ap) delete [] Ap;
        if(Ai) delete [] Ai;
        if(Ax) delete [] Ax;

        nz=nzz;
        Ap=new int[qmr.n+1];
        Ai=new int[nz];
        Ax=new T [nz];
        memcpy(Ap,App,sizeof(int)*(qmr.n+1));
        memcpy(Ai,Aii,sizeof(int)*nz);
        memcpy(Ax,Axx,sizeof(T)*nz);
printf("Ap address before: %Xl\n",(unsigned long)Ap);
        if(sizeof(T)==sizeof(double)) 
	        DBLQMRPrep(&qmr,Ap,Ai,Ax,&nz);
        else if(sizeof(T)==sizeof(F90Complex))
                ZBLQMRPrep(&qmr,Ap,Ai,(F90Complex*)Ax,&nz);
        else
                throw beTypeMismatch;
printf("Ap address after: %Xl\n",(unsigned long)Ap);
    }
    void Solve(T *x, T *b, int nrhs){
    	if(nz==0 || Ap==BLIT_NULL) throw(beNoLHS);
    	if(x==BLIT_NULL || b==BLIT_NULL)  throw(beNoRHS);

        if(sizeof(T)==sizeof(double)) 
	        DBLQMRSolve(&qmr,Ap,Ai,Ax,&nz,x,b,&nrhs);
        else if(sizeof(T)==sizeof(F90Complex))
                ZBLQMRSolve(&qmr,Ap,Ai,(F90Complex*)Ax,&nz,(F90Complex*)x,(F90Complex*)b,&nrhs);
        else
                throw beTypeMismatch;
    }
    void Print(){
        if(sizeof(T)==sizeof(double)) 
	        DBLQMRPrint(&qmr);
        else if(sizeof(T)==sizeof(F90Complex))
		ZBLQMRPrint(&qmr);
        else
                throw beTypeMismatch;
    }
};
#endif

