#ifndef CLUSTER_H
#define CLUSTER_H

#include "libhead.h"
#include "term.h"
#include "kernel.h"
#include "organized.h"
#include "hyper.h"

class Cluster
{
    public:
        OrganizedData   *od;
        Kernel          *ker;
        int             first_blk, last_blk, cluster_size;
        bool            dtc;
        double          Phi, logPDD;
        Term            *KSS, *KSS_inv, *Alpha, *Beta;
        Term			*Sbb, *Sbk, *Skb, *Gkk, *Tk, *KSDb, *KSDi_t;
        Term			*left, *right, *alpha, *beta;
        mat				Sbb_inv, Uk, yk, ykt, yb, ybt, cmean, cvar;
        eterm           *Delta, *delta, *e;
        field < Term* > SBB;
        deque < Term* > KSD;

        Cluster(int first_blk, int last_blk, int cluster_size, bool dtc, OrganizedData *od, Kernel *ker, Term *KSS, Term *KSS_inv);
       ~Cluster();

        void process();
        void process_dtc();
        void process_last_cluster();
        void prepare();
        void prepare_dtc();
        void compute_Tk();
        void compute_Gkk();
        void combine_KSD();
        void compute_y();
        void compute_alpha_beta();
        void compute_delta();
        void compute_S(int &i, int &j, Term *KSDi_t, Term *Sij);
        void extract();
        void extract_dtc();
        void update();
        void update_dtc();
        void clearall();
    protected:
    private:
};

#endif // CLUSTER_H
