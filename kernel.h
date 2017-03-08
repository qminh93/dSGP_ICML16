/*
 * Kernel.h
 *
 *  Created on: 6 Aug 2015
 *      Author: nghiaht
 */

#ifndef KERNEL_H_
#define KERNEL_H_

#include "libhead.h"
#include "term.h"
#include "hyper.h"

class Kernel
{
    public:

	HyperParams* hyp;

    Kernel(HyperParams* hyp);
   ~Kernel();

    void   k    		(mat *A, mat *B, int &i, int &j, double &res);
    void   kmat 		(mat *A, mat &KAA);
    void   kmat_par	    (mat *A, mat &KAA, int &nThread);
    void   kmat 		(mat *A, mat *B, mat &KAB);
    void   dls  		(mat *A, mat *B, mat &K, mat &dls, int &x);
    void   dls 	 	    (mat *A, mat &K, mat &dls, int &x);
    void   dls_par	    (mat *A, mat &K, mat &dls, int &x, int &nThread);
    void   dsig 		(mat &K, mat &dsig);
    void   kTerm		(mat *A, Term *res);
    void   kTerm_par	(mat *A, Term *res, int &nThread);
    void   kTerm		(mat *A, mat *B, Term *res);

    private:
    protected:
};


#endif /* KERNEL_H_ */
