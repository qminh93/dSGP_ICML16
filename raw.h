#ifndef RAWDATA_H
#define RAWDATA_H

#define LOAD_LIMIT_DEFAULT 2000000

#include "libhead.h"

class RawData
{
    public:
        mat X;
        int nDim, nData, nLim;

        RawData();
        RawData(mat& X);
        RawData(int nLim);
        RawData(int nLim, mat &X);
       ~RawData();

        void save(const char* datafile);
        void load(const char* datafile);

        void normalize_x();
        void normalize_y(pdd &meandev);

    protected:
    private:
};

#endif // RAWDATA_H
