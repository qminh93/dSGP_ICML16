/*
 * Kernel.cpp
 *
 *  Created on: 6 Aug 2015
 *      Author: nghiaht
 */

#include "kernel.h"

Kernel::Kernel(HyperParams* hyp)
{
    this->hyp = hyp;
}

Kernel::~Kernel()
{

}

// Compute K(A,A)
void Kernel::kmat(mat *A, mat &KAA)
{
	NFOR(i, j, A->n_rows, A->n_rows)
	{
		if (i <= j) k(A, A, i, j, KAA(i, j));
		else KAA(i, j) = KAA(j, i);
	}

	KAA.diag() += 0.001;
}

// Parallel compute K(A,A)
void Kernel::kmat_par(mat *A, mat &KAA, int &nThread)
{
	int chunk = max ((int) A->n_rows / nThread, 1);
	#pragma omp parallel for schedule(dynamic, chunk)
	SFOR(i, A->n_rows)
	for (int j = i ; j < (int) A->n_rows; j++) {
		k(A, A, i, j, KAA(i, j));
		KAA(j, i) = KAA(i, j);
	}

	KAA.diag() += 0.001;
}

// Compute K(A,B)
void Kernel::kmat(mat *A, mat *B, mat &KAB)
{
    NFOR(i, j, A->n_rows, B->n_rows)
        k(A, B, i, j, KAB(i, j));
}

// Compute res <- K(A,B)[i,j]
void Kernel::k(mat* A, mat* B, int &i, int &j, double &res)
{
    res = 0.0;
    SFOR(t, hyp->nDim)
        res += pow((A->at(i,t) - B->at(j,t)) / hyp->ls(t), 2.0);
    res = pow(hyp->signal(), 2.0) * exp(-0.5 * res);
}

// Compute dsig <- dK/dsignal
void Kernel::dsig(mat &K, mat &dsig)
{
	dsig = (2.0 / hyp->signal()) * K;
}

// Compute dls <- dK(A,B)/dls
void Kernel::dls(mat *A, mat *B, mat &K, mat &dls, int &x)
{
    NFOR(i, j, A->n_rows, B->n_rows)
        dls(i, j) = K(i, j) * pow(A->at(i, x) - B->at(j, x), 2.0) * pow(hyp->ls(x), -3.0);
}

// Compute dls <- dK(A,A)/dls
void Kernel::dls(mat *A, mat &K, mat &dls, int &x)
{
    NFOR(i, j, A->n_rows, A->n_rows)
	{
        if (i <= j) dls(i, j) = K(i, j) * pow(A->at(i, x) - A->at(j, x), 2.0) * pow(hyp->ls(x), -3.0);
        else dls(i, j) = dls(j, i);
	}
}

// Parallel compute dK(A,A)/dls
void Kernel::dls_par(mat *A, mat &K, mat &dls, int &x, int &nThread)
{
	int chunk = max((int) A->n_rows, 1);
	#pragma omp parallel for schedule(dynamic, chunk)
	SFOR(i, A->n_rows)
	for (int j = i; j < (int) A->n_rows; j++)
	{
		dls(i, j) = K(i, j) * pow(A->at(i, x) - A->at(j, x), 2.0) * pow(hyp->ls(x), -3.0);
		dls(j, i) = dls(i, j);
	}
}

// Parallel compute res <- dK(A,A)/dZ
void Kernel::kTerm_par(mat *A, Term *res, int &nThread)
{
	res->me            = mat(A->n_rows, A->n_rows);
	res->dsignal       = mat(A->n_rows, A->n_rows);
	res->dnoise		   = zeros <mat> (A->n_rows, A->n_rows);

	kmat_par(A, res->me, nThread);
	dsig(res->me, res->dsignal);

	SFOR(t, hyp->nDim)
	{
		res->dls[t] = mat(A->n_rows, A->n_rows);
		dls_par(A, res->me, res->dls[t], t, nThread);
	}
}

// Compute res <- dK(A,B)/dZ
void Kernel::kTerm(mat *A, mat *B, Term *res)
{
    res->me            = mat(A->n_rows, B->n_rows);
	res->dsignal       = mat(A->n_rows, B->n_rows);
	res->dnoise		   = zeros <mat> (A->n_rows, B->n_rows);

    kmat(A, B, res->me);
    dsig(res->me, res->dsignal);

	SFOR(t, hyp->nDim)
	{
		res->dls[t] = mat(A->n_rows, B->n_rows);
		dls(A, B, res->me, res->dls[t], t);
	}
}

// Compute res <- dK(A,A)/dZ
void Kernel::kTerm(mat *A, Term *res)
{
	res->me            = mat(A->n_rows, A->n_rows);
	res->dsignal       = mat(A->n_rows, A->n_rows);
	res->dnoise		   = zeros <mat> (A->n_rows, A->n_rows);

	kmat(A, res->me);
	dsig(res->me, res->dsignal);

	SFOR(t, hyp->nDim)
	{
		res->dls[t] = mat(A->n_rows, A->n_rows);
		dls(A, res->me, res->dls[t], t);
	}
}

