#include "term.h"

Term::Term(int nRows, int nCols, int nDim)
{
    this->nRows = nRows;
    this->nCols = nCols;
    this->nDim  = nDim;
    this->dls   = vm(nDim);
}

Term::~Term()
{
    clearall();
}

// Initialize this term with empty matrices
void Term::init()
{
    me      = mat(nRows, nCols);
    dsignal = mat(nRows, nCols);
    dnoise  = mat(nRows, nCols);
    SFOR(v, nDim)
    dls[v]  = mat(nRows, nCols);
}

// Initialize this_term with filler
void Term::init(double filler)
{
	me      = mat(nRows, nCols); me.fill(filler);
	dsignal = mat(nRows, nCols); dsignal.fill(filler);
	dnoise  = mat(nRows, nCols); dnoise.fill(filler);
	SFOR(v, nDim) {
	dls[v]  = mat(nRows, nCols); dls[v].fill(filler);
	}
}

// Compute res <- this_term + other (gradients only)
void Term::g_add(mat& other, Term* res)
{
    res->dsignal = dsignal + other;
    SFOR(v, nDim)
    res->dls[v]  = dls[v]  + other;
    res->dnoise  = dnoise  + other;
}

// Compute res <- this_term + other
void Term::add(mat& other, Term* res)
{
    res->me = me + other;
    g_add(other, res);
}

// Compute res <- left - this_term (gradients only)
void Term::g_subleft(mat& left, Term* res)
{
    res->dsignal = left - dsignal;
    SFOR(v, nDim)
    res->dls[v]  = left - dls[v];
    res->dnoise  = left - dnoise;
}

// Compute res < this_term - right (gradients only)
void Term::g_subright(mat& right, Term* res)
{
    res->dsignal = dsignal - right;
    SFOR(v, nDim)
    res->dls[v]  = dls[v]  - right;
    res->dnoise  = dnoise - right;
}

// Compute res <- left - this_term
void Term::subleft(mat& left, Term* res)
{
    res->me = left - me;
    g_subleft(left, res);
}

// Compute res <- this_term - right
void Term::subright(mat& right, Term* res)
{
    res->me = me - right;
    g_subright(right, res);
}

// Compute res <- const * this_term (gradients only)
void Term::g_mulconst(double con, Term* res)
{
    res->dsignal = con * dsignal;
    SFOR(v, nDim)
    res->dls[v]  = con * dls[v];
    res->dnoise  = con * dnoise;
}

// Compute res <- left * this_term (gradients only)
void Term::g_mulleft(mat& left, Term* res)
{
    res->dsignal = left * dsignal;
    SFOR(v, nDim)
    res->dls[v]  = left * dls[v];
    res->dnoise  = left * dnoise;
}

// Compute res <- this_term * right (gradients only)
void Term::g_mulright(mat& right, Term* res)
{
    res->dsignal = dsignal * right;
    SFOR(v, nDim)
    res->dls[v]  = dls[v]  * right;
    res->dnoise  = dnoise * right;
}

// Compute left * this->term * right (gradients only)
void Term::g_mulsides(mat& left, mat& right, Term* res)
{
    res->dsignal = left * dsignal * right;
    SFOR(v, nDim)
    res->dls[v]  = left * dls[v]  * right;
    res->dnoise  = left * dnoise  * right;
}

// Compute res <- const * this_term
void Term::mulconst(double con, Term* res)
{
    res->me = con * me;
    g_mulconst(con, res);
}

// Compute res <- left * this_term
void Term::mulleft(mat& left, Term* res)
{
    res->me = left * me;
    g_mulleft(left, res);
}

// Compute res -> this_term * right
void Term::mulright(mat& right, Term* res)
{
    res->me = me * right;
    g_mulright(right, res);
}

// Compute res <- left * this_term * right
void Term::mulsides(mat& left, mat& right, Term* res)
{
    res->me = left * me * right;
    g_mulsides(left, right, res);
}

// Inverse this_term
void Term::invert(bool sympd, Term* res)
{
    if (sympd) inv_sympd(res->me, me);
    else inv(res->me, me);

    g_mulsides(res->me, res->me, res);
    res->g_mulconst(-1.0, res);
}

// Transpose this_term
void Term::transpose()
{
	int temp = nRows; nRows = nCols; nCols = temp;

	inplace_trans(me);
    inplace_trans(dsignal);
    SFOR(v, nDim)
    inplace_trans(dls[v]);
    inplace_trans(dnoise);
}

// Compute res <- this_term^T
void Term::transpose(Term* res)
{
	res->nRows   = nCols;
	res->nCols   = nRows;
    res->me      = me.t();
    res->dsignal = dsignal.t();
    SFOR(v, nDim)
    res->dls[v]  = dls[v].t();
    res->dnoise  = dnoise.t();
}

// Compute trace(this_term)
void Term::tr(eterm *res)
{
	res->me 	 = trace(me);
	res->dsignal = trace(dsignal);
	res->dnoise  = trace(dnoise);
	SFOR(v, nDim)
	res->dls[v]  = trace(dls[v]);
}

// clone res <- this_term
void Term::clone(Term* res)
{
    res->me      = me;
    res->dsignal = dsignal;
    SFOR(v, nDim)
    res->dls[v]  = dls[v];
    res->dnoise  = dnoise;
}

// set this_term.submat(fr,fc,lr,lc) <- src
void Term::set_submat(int &fr, int &fc, int &lr, int &lc, Term *src)
{
	me.submat(fr, fc, lr, lc) 		= src->me;
	dsignal.submat(fr, fc, lr, lc)  = src->dsignal;
	dnoise.submat(fr, fc, lr, lc)   = src->dnoise;
	SFOR(v, nDim)
	dls[v].submat(fr, fc, lr, lc)   = src->dls[v];
}

// set this_term.submat(fr,fc,lr,lc) <- src^T
void Term::set_submat_t(int &fr, int &fc, int &lr, int &lc, Term *src)
{
	me.submat(fr, fc, lr, lc) = src->me.t();
	dsignal.submat(fr, fc, lr, lc) = src->dsignal.t();
	dnoise.submat(fr, fc, lr, lc) = src->dnoise.t();
	SFOR(v, nDim)
	dls[v].submat(fr, fc, lr, lc) = src->dls[v].t();
}

// Compute term[t1] + term[t2] (only gradients)
void Term::g_add(Term* t1, Term* t2, Term* res)
{
    res->dsignal = t1->dsignal + t2->dsignal;

    SFOR(v, t1->nDim)
    res->dls[v]  = t1->dls[v]  + t2->dls[v];

    res->dnoise  = t1->dnoise  + t2->dnoise;
}

// compute term[t1] + term[t2]
void Term::add(Term* t1, Term* t2, Term* res)
{
    res->me      = t1->me      + t2->me;
    g_add(t1, t2, res);
}

// compute term[t1] - term[t2] (only gradients)
void Term::g_sub(Term* t1, Term* t2, Term* res)
{
    res->dsignal = t1->dsignal - t2->dsignal;

    SFOR(v, t1->nDim)
    	res->dls[v]  = t1->dls[v]  - t2->dls[v];

    res->dnoise  = t1->dnoise  - t2->dnoise;
}

// compute term[t1] - term[t2]
void Term::sub(Term* t1, Term* t2, Term* res)
{
    res->me      = t1->me      - t2->me;
    g_sub(t1, t2, res);
}

// Compute const * term[t](i,j)
void Term::e_mulconst(Term *t, double con, int &i, int &j, eterm *res)
{
    res->me         = t->me(i, j) 		* con;
    res->dsignal    = t->dsignal(i, j)  * con;
    SFOR(v, t->nDim)
    res->dls[v]     = t->dls[v](i, j) 	* con;
    res->dnoise 	= t->dnoise(i, j) 	* con;
}

// Query term[t](i,j)
void Term::e_at(Term *t, int &i, int &j, eterm *res)
{
	res->me      = t->me(i, j);
	res->dsignal = t->dsignal(i, j);
	SFOR(v, t->nDim)
	res->dls[v]  = t->dls[v](i, j);
	res->dnoise  = t->dnoise(i, j);
}

// Compute (term[t1] * term[t2])(i,j)
void Term::e_add(Term *t, eterm *e, int &i, int &j, Term* res)
{
    res->me(i, j)         = t->me(i, j)      + e->me;
    res->dsignal(i, j)    = t->dsignal(i, j) + e->dsignal;
    SFOR(v, t->nDim)
    res->dls[v](i, j)     = t->dls[v](i, j)  + e->dls[v];
    res->dnoise(i, j) 	  = t->dnoise(i, j)  + e->dnoise;
}

// compute (term[t1] * term[t2])(i,j)
void Term::e_mul2(Term* t1, Term* t2, int &i, int &j, eterm* res)
{
	eterm* sres = new eterm(t1->nDim);

	sres->me = dot(t1->me.row(i), t2->me.col(j));
	sres->dsignal = dot(t1->dsignal.row(i), t2->me.col(j)) + dot(t1->me.row(i), t2->dsignal.col(j));
	sres->dnoise = dot(t1->dnoise.row(i), t2->me.col(j)) + dot(t1->me.row(i), t2->dnoise.col(j));
	SFOR(v, t1->nDim)
		sres->dls[v] = dot(t1->dls[v].row(i), t2->me.col(j)) + dot(t1->me.row(i), t2->dls[v].col(j));

	res->add(sres, res);
	sres->clear();
	delete sres;
}

// compute (term[t1] * term[t2] * term[t3])(i,j)
void Term::e_mul3(Term* t1, Term* t2, Term* t3, int &i, int &j, eterm* res)
{
	eterm* sres = new eterm(t1->nDim);
	sres->me = trace(t1->me.row(i) * t2->me * t3->me.col(j));
	sres->dsignal = trace(t1->dsignal.row(i) * t2->me * t3->me.col(j) + t1->me.row(i) * t2->dsignal * t3->me.col(j)
			                                                          + t1->me.row(i) * t2->me * t3->dsignal.col(j));
	sres->dnoise  = trace(t1->dnoise.row(i) * t2->me * t3->me.col(j)  + t1->me.row(i) * t2->dnoise * t3->me.col(j)
                                                                      + t1->me.row(i) * t2->me * t3->dnoise.col(j));
	SFOR(v, t1->nDim)
		sres->dls[v] = trace(t1->dls[v].row(i) * t2->me * t3->me.col(j) + t1->me.row(i) * t2->dls[v] * t3->me.col(j)
                                                                        + t1->me.row(i) * t2->me * t3->dls[v].col(j));
	res->add(sres, res);
	sres->clear();
    delete sres;
}
// Compute term[t1](i,u) * term[t2](x,z)
void Term::e_mul2(Term* t1, Term* t2, int &i, int &u, int &v, int &j, eterm* res)
{
    res->dsignal = t1->me(i, u) * t2->dsignal(v, j) + t1->dsignal(i, u) * t2->me(v, j);
    SFOR(t, t1->nDim)
    res->dls[t]  = t1->me(i, u) * t2->dls[t](v, j)  + t1->dls[t](i, u)  * t2->me(v, j);
    res->dnoise  = t1->me(i, u) * t2->dnoise(v, j)  + t1->dnoise(i, u)  * t2->me(v, j);
    res->me 	 = t1->me(i, u) * t2->me(u, j);
}

// Compute term[t1](i,u) * term[t2](x,z) * term[t3](v,j)
void Term::e_mul3(Term* t1, Term* t2, Term* t3, int &i, int &u, int &x, int &z, int &v, int &j, eterm* res)
{
    e_mul2(t1, t2, i, u, x, z, res);

    res->dsignal = res->me * t3->dsignal(v, j) + res->dsignal * t3->me(v, j);
    SFOR(t, t1->nDim)
    res->dls[t]  = res->me * t3->dls[t](v, j)  + res->dls[t]  * t3->me(v, j);
    res->dnoise  = res->me * t3->dnoise(v, j)  + res->dnoise  * t3->me(v, j);
    res->me     *=  t3->me(v, j);
}

// Compute term[t1] * term[t2]
void Term::mul2(Term* t1, Term* t2, Term* res)
{
    res->dsignal = t1->me * t2->dsignal + t1->dsignal * t2->me;
    SFOR(v, t1->nDim)
    res->dls[v]  = t1->me * t2->dls[v]  + t1->dls[v]  * t2->me;
    res->dnoise  = t1->me * t2->dnoise  + t1->dnoise  * t2->me;
    res->me      = t1->me *  t2->me;
}

// Compute term[t1] * term[t2] * term[t3]
void Term::mul3(Term* t1, Term* t2, Term* t3, Term* res)
{
	int p1 = t1->nRows * t2->nCols * (t2->nRows + t3->nCols),
		p2 = t2->nRows * t3->nCols * (t3->nRows + t1->nRows);

	if (p1 < p2)
	{
		mul2(t1,  t2, res);
		mul2(res, t3, res);
	}
	else
	{
		mul2(t2, t3, res);
		mul2(t1, res, res);
	}
}

// Compute term[trace(term[t1] * term[t2])]
void Term::trace2(Term* t1, Term* t2, eterm* res)
{
    // Assuming t1->nRows == t2->nCols
	eterm* sres = new eterm(t1->nDim);
    SFOR(i, t1->nRows)
    {
        e_mul2(t1, t2, i, i, sres);
        res->add(sres, res);
        sres->reset();
    }
    sres->clear();
    delete sres;

}

// Compute term[trace(term[t1] * term[t2] * term[t3])]
void Term::trace3(Term* t1, Term* t2, Term* t3, eterm* res)
{
	// Assuming t1->nRows == t3->nCols
	eterm* sres = new eterm(t1->nDim);
    SFOR(i, t1->nRows)
    {
        e_mul3(t1, t2, t3, i, i, sres);
        res->add(sres, res);
        sres->reset();
    }
    sres->clear();
    delete sres;
}

// Compute term[trace(t1 * term[t2])]
void Term::tracemat(mat &t1, Term *t2, eterm *res)
{
    res->me      = trace(t1 * t2->me);
    res->dsignal = trace(t1 * t2->dsignal);
    res->dnoise  = trace(t1 * t2->dnoise);
    SFOR(v, t2->nDim)
    res->dls[v]  = trace(t1 * t2->dls[v]);
}

// Compute term[trace(term[t1] * t2)]
void Term::tracemat(Term *t1, mat &t2, eterm *res)
{
    res->me      = trace(t1->me 	 * t2);
    res->dsignal = trace(t1->dsignal * t2);
    res->dnoise  = trace(t1->dnoise  * t2);
    SFOR(v, t1->nDim)
    res->dls[v]  = trace(t1->dls[v]  * t2);
}

// Compute term[trace(sidel * term[t] * sider)]
void Term::tracet3_mat(Term *t, mat &sidel, mat &sider, eterm *res)
{
    res->me      = trace(sidel * t->me 		* sider);
    res->dsignal = trace(sidel * t->dsignal * sider);
    res->dnoise  = trace(sidel * t->dnoise	* sider);
    SFOR(v, t->nDim)
    res->dls[v]  = trace(sidel * t->dls[v]	* sider);
}

// Compute term[trace(term[t] * term[mid] * term[t^T])]
void Term::tracet3(Term *t, Term *mid, eterm* res)
{
	mat Q	= t->me.t();
	eterm *sres = new eterm(mid->nDim);
	tracet3_mat(mid, t->me, Q, sres);
	Q = mid->me * Q;

	res->me      = sres->me;
	res->dsignal = 2 * trace(t->dsignal * Q)  + sres->dsignal;
	res->dnoise  = 2 * trace(t->dnoise  * Q)  + sres->dnoise;
	SFOR(v, mid->nDim)
	res->dls[v]  = 2 * trace(t->dls[v]  * Q)  + sres->dls[v];

	Q.clear();
	sres->clear();
	delete sres;
}

void Term::clearall()
{
    me.clear();
    dsignal.clear();
    dnoise.clear();
    SFOR(v, nDim) dls[v].clear();
    dls.clear();
}
