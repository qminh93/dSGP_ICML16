#ifndef TERM_H
#define TERM_H

#include "libhead.h"

struct eterm
{
    int     nDim;
    double  me, dsignal, dnoise;
    vd      dls;

    eterm(int nDim) { this->nDim = nDim; dls = vd(nDim, 0.0); me = dsignal = dnoise = 0.0; }
   ~eterm()         { clear(); }

    void clear()
    {
    	dls.clear();
    }
    void reset()
    {
    	me = dsignal = dnoise = 0.0;
    	SFOR(v, nDim) dls[v] = 0.0;
    }
    void add(eterm *other, eterm *res)
    {
        res->me      = me       + other->me;
        res->dsignal = dsignal  + other->dsignal;
        res->dnoise  = dnoise   + other->dnoise;

        SFOR(v, nDim)
        res->dls[v]  = dls[v]   + other->dls[v];
    }

    void subright(eterm *right, eterm *res)
    {
        res->me      = me       - right->me;
        res->dsignal = dsignal  - right->dsignal;
        res->dnoise  = dnoise   - right->dnoise;

        SFOR(v, nDim)
        res->dls[v]  = dls[v]   - right->dls[v];
    }

    void subleft(eterm *left, eterm *res)
    {
        res->me      = left->me      - me;
        res->dsignal = left->dsignal - dsignal;
        res->dnoise  = left->dnoise  - dnoise;

        SFOR(v, nDim)
        res->dls[v]  = left->dls[v]  - dls[v];
    }

    void mul(double con, eterm *res)
    {
        res->me      = con * me;
        res->dsignal = con * dsignal;
        res->dnoise  = con * dnoise;
        SFOR(v, nDim)
        res->dls[v]  = con * dls[v];
    }

    void add_const(double con, eterm *res)
    {
    	res->me      = con + me;
    	res->dsignal = con + dsignal;
    	res->dnoise  = con + dnoise;
    	SFOR(v, nDim)
    	res->dls[v]  = con + dls[v];
    }

    double dot(eterm* other)
    {
    	double res = dsignal * other->dsignal + dnoise * other->dnoise;
    	SFOR(i, nDim) res += dls[i] * other->dls[i];
    	return res;
    }

    void print()
    {
    	cout << dnoise << " " << dsignal;
    	SFOR(i, nDim) cout << " " << dls[i];
    	cout << endl;
    }
};

class Term
{
    public:
        int nRows, nCols, nDim;
        mat me, dsignal, dnoise;
        vm  dls;

        Term(int nRows, int nCols, int nDim);
       ~Term();

        // Term operations with single mat, last argument is the resultant term.
        // for self addition/ multiplication pass itself in as argument
        // ----------
        // Addition
        // ----------
        void g_add     (mat &other, Term *res);
        void add       (mat &other, Term *res);

        // Subtraction
        // ----------
        void g_subleft (mat &left , Term *res);
        void g_subright(mat &right, Term *res);
        void subleft   (mat &left , Term *res);
        void subright  (mat &right, Term *res);

        // Multiplication (with mat)
        // ----------
        void g_mulconst(double con   , Term *res);
        void g_mulleft (mat &left , Term *res);
        void g_mulright(mat &right, Term *res);
        void g_mulsides(mat &left , mat  &right, Term *res);
        void mulconst  (double con   , Term *res);
        void mulleft   (mat &left , Term *res);
        void mulright  (mat &right, Term *res);
        void mulsides  (mat &left , mat  &right, Term *res);

        // Other matrix operations, res of size (nRows * nCols)
        // ----------
        void init     	  ();
        void init    	  (double filler);
        void invert   	  (bool sympd, Term *res);
        void transpose	  ();
        void transpose	  (Term  *res);
        void clone    	  (Term  *res);
        void set_submat   (int &fr, int &fc, int &lr, int &lc, Term *src);
        void set_submat_t (int &fr, int &fc, int &lr, int &lc, Term *src);
        void tr		  	  (eterm *res);
        void cholesky     (Term  *res);

        // Element operations - Note: term (element of Term) != Term
        // ----------
        static void e_at	    (Term* t, int &i, int &j, eterm *res);
        static void e_add       (Term *t, eterm *e, int &i, int &j, Term *res);
        static void e_mulconst  (Term *t, double con, int &i, int &j, eterm *res);
        // Return (t1 * t2)[i, j] or (t1 * t2 * t3)[i, j]
        static void e_mul2      (Term *t1, Term *t2, int &i, int &j, eterm *res);
        static void e_mul3      (Term *t1, Term *t2, Term *t3, int &i, int &j, eterm *res);
        // Return t1(i, u) * t2(u, j)
        static void e_mul2      (Term *t1, Term *t2, int &i, int &u, int &v, int &j, eterm *res);
        // Return t1(i, u) * t2(u, v) * t3(v, j)
        static void e_mul3      (Term *t1, Term *t2, Term *t3, int &i, int &u, int &x, int &z, int &v, int &j, eterm *res);

        // Static functions
        // ----------
        // Term - Term addition
        static void g_add       (Term *t1, Term *t2, Term  *res);
        static void add         (Term *t1, Term *t2, Term  *res);
        static void g_sub       (Term *t1, Term *t2, Term  *res);
        static void sub         (Term *t1, Term *t2, Term  *res);
        // Term - Term multiplication, res of size (t{1}->nRows * t{2/3}->nCols)
        static void mul2        (Term *t1, Term *t2, Term  *res);
        static void mul3        (Term *t1, Term *t2, Term  *t3, Term *res);
        // Trace of product of Terms, res of size (1 x 1)
        static void trace2      (Term *t1, Term *t2, eterm *res);
        static void trace3      (Term *t1, Term *t2, Term  *t3, eterm *res);
        static void tracemat    (mat  &t1, Term *t2, eterm *res);
        static void tracemat    (Term *t1, mat  &t2, eterm *res);
        static void tracet3_mat	(Term *t,  mat &sidel, mat &sider, eterm *res);
        static void tracet3		(Term *t,  Term *mid, eterm *res);

        // Memory management
        // ----------
        void clearall();

    protected:
    private:
};

#endif // TERM_H
