#ifndef HYPERPARAMS_H
#define HYPERPARAMS_H

#include "libhead.h"

class HyperParams
{
    public:

        HyperParams();
        ~HyperParams();

        field <mat> params;
        int nDim;

        void    save(const char* datafile);
        void    load(const char* datafile);
        void    clone(HyperParams* other);
        void	print();

        void    setndim(int nDim);
        void    setmean(double _mean);
        void    setnoise(double _noise);
        void    setsignal(double _signal);
        void    setls(int i, double _lsi);
        void    setkparams(vd &k_params);

        double  mean();
        double  noise();
        double  signal();
        double  ls(int i);
        mat     kparams();

    protected:
    private:
};

#endif // HYPERPARAMS_H
