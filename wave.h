#ifndef WAVES_H
#define WAVES_H

#include "libhead.h"
#include "organized.h"
#include "kernel.h"
#include "term.h"

struct Wave
{
    int wave_id;
    vector <mat> RUD, RDU, RDD;

    Wave(int wave_id) {
        this->wave_id = wave_id;
    }

    ~Wave() {
        clear();
    }

    void clear() {
        SFOR(i, RUD.size()) RUD[i].clear(); RUD.clear();
        SFOR(i, RDU.size()) RDU[i].clear(); RDU.clear();
        SFOR(i, RDD.size()) RDD[i].clear(); RDD.clear();
    }
};

class Waves
{
    public:
        OrganizedData   *od;
        Kernel          *ker;
        mat             KSS_inv;

        int band_size;

        deque < Wave* > wave_band;
        field < mat > SigmaUD;

        Waves(int band_size, OrganizedData *od, HyperParams *hyp);
       ~Waves();

       void ud_exact_eval(int p, int q, mat &Rpq);
       void du_exact_eval(int p, int q, mat &Rpq);
       void dd_exact_eval(int p, int q, mat &Rpq);
       void compute_R(int p, mat &R1, 	mat &R2);
       void combine_R(int p, mat &Rdu, 	mat &Rdd);
       void process(int nThread);


    protected:
    private:
};

#endif // WAVE_H
