#ifndef FACTORY_H
#define FACTORY_H

#include "organized.h"
#include "kernel.h"
#include "cluster.h"
#include "term.h"
#include "libhead.h"
#include "wave.h"
#include "regressor.h"

class Factory
{
    public:

        OrganizedData*          od;
        Kernel*                 ker;
        vector < Cluster* >     markov_cluster;

        vector < mat >          mu;
        vector < HyperParams* > hp;
        vector <double> score;

        int     nThread, cluster_size;
        Term    *KSS, *KSS_inv, *GammaSS, *GammaSS_inv, *VSD, *VSD_t;
        eterm	*eR;
        double  Phi, logPDD, R1, R2, sign1, sign2;
        bool    dtc;

        Factory(OrganizedData* od, Kernel *ker, int &nThread, int &nBand, bool dtc);
       ~Factory();

        void	reset();
        void	clearall();
        void    precompute();
        void    process_stream(int &stream_number);
        void 	compute_R(eterm *R);

        void    save_result(char* filename);
        void    load_result(char* filename);
        double  	PIC_precompute(HyperParams *hyp, mat *qfs);
        Waves*  prepare_waves(HyperParams* hyp);
        double  predict(mat *qfs, HyperParams* hyp, Waves *tsunami);
        double  bcm_predict(mat *qfs);
        double  DTC_predict(mat *qfs, HyperParams *hyp);
        double  PIC_predict(mat *qfs, vm &M, vm &b);

    protected:
    private:
};

#endif // FACTORY_H
