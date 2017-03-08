#ifndef REGRESSOR_H_
#define REGRESSOR_H_

#include "libhead.h"
#include "kernel.h"
#include "term.h"
#include "wave.h"

class Regressor {
public:
    OrganizedData   *od;
    Kernel          *ker;
    Waves           *tsunami;

    mat KSS_inv, *qfs;
    vm  M;
    vvm Mt;
    int band_size, nThread;

    vector < deque <mat> > KSD;


	Regressor(int nThread, int band_size, OrganizedData *od, HyperParams *hyp, Waves* tsunami, mat* qfs);
   ~Regressor();

    void prepare_S(int t, int k, int kB_pos, mat &SBB);
    void slide_S(int t, int k, int kB_pos, mat &SBB);
    void prepare_KSD(int t, int k, int kB_pos);
    void slide_KSD(int t, int k, int kB_pos);
    void extract_S(int t, int k, mat &SBB, mat &Sbb_inv, mat &Sbk, mat &Skk);
    void compute_mu(int t, int k, int kB_pos, mat &mu, mat &mu_k);
    void extract_KuD(int p, int k, int kB_pos, mat &KuDb);
    void predict();
};

#endif /* REGRESSOR_H_ */
