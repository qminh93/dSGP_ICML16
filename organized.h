#ifndef ORGANIZEDDATA_H
#define ORGANIZEDDATA_H

#include "hyper.h"
#include "raw.h"
#include "kmean.h"
#include "libhead.h"

class OrganizedData
{
    public:
        int            nTrain, nTest, nSupport, nDim, nBlock, bSize, tSize;
        field < mat* > train, test, support, ytrain_t;
        HyperParams*   hyper;
        Configuration* setting;

        OrganizedData(Configuration* setting);
       ~OrganizedData();

        void    save(vs dataset, string mode);
        void    save(vs dataset);
        void    load(vs dataset);
        void    loadHyp(string hypfile);
        void    process(RawData* raw, int nBlock, double pTest, int support_per_block, int max_number_support);

        mat     *getxb (int &i);   // i-th training input data block
        mat     *getyb (int &i);   // i-th training output vector
        mat		*getybt(int &i);	  // i-th training output vector transpose
        mat     *getxt (int &i);   // i-th testing input data block
        mat     *getyt (int &i);   // i-th testing output vector
        mat     *getxm ();        // inducing input
        mat     *getym ();        // inducing output

        double  noise();
        double  mean();
        double  signal();
        double  ls(int &i);

    protected:
    private:
};

#endif // ORGANIZEDDATA_H
