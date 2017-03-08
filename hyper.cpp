#include "hyper.h"

HyperParams::HyperParams()
{
    params  = field <mat> (1,3);
    nDim    = 0;
}
HyperParams::~HyperParams()
{
    params.clear();
}

void HyperParams::save(const char* datafile)
{
    FILE* fout = fopen(datafile,"w");

    fprintf(fout,"%f,%f,%f",mean(),log(noise()),log(signal()));
    SFOR(i,nDim)
        fprintf(fout,",%f",log(ls(i)));

    fclose(fout);
}

void HyperParams::clone(HyperParams* h)
{
    h->params = params;
    h->nDim   = nDim;
}

void HyperParams::load(const char* datafile)
{
    ifstream fin(datafile);
    string   token;
    vd       params;

    getline(fin,token,',');
    setmean(log(atof(token.c_str())));

    getline(fin,token,',');
    setnoise(atof(token.c_str()));

    nDim = -1;
    while (getline(fin,token,','))
    {
        nDim++;
        params.push_back(atof(token.c_str()));
    }

    setkparams(params);
    fin.close();
}

void HyperParams::print()
{
	cout << noise() << " " << signal();
	SFOR(i, nDim) cout << " " << ls(i);
	cout << endl;
}

void HyperParams::setmean(double _mean)
{
	params(0, 0) = mat(1, 1).fill(_mean);
}

void HyperParams::setnoise(double _noise)
{
    params(0, 1) = mat(1, 1).fill(_noise);
}

void HyperParams::setsignal(double _signal)
{
    params(0, 2)(0, 0) = _signal;
}

void HyperParams::setls(int i, double _lsi)
{
    params(0, 2)(0, i + 1) = _lsi;
}

void HyperParams::setkparams(vd &_kparams)
{
    params(0, 2) = mat(1,(int)_kparams.size());
    for (int i = 0; i < (int)_kparams.size(); i++)
        params(0, 2)(0, i) = _kparams[i];
}

void HyperParams::setndim(int nDim)
{
    this->nDim = nDim;
}

double HyperParams::mean()
{
    return params(0, 0)(0, 0);
}

double HyperParams::noise()
{
    return exp(params(0, 1)(0, 0));
}

double HyperParams::signal()
{
    return exp(kparams()(0, 0));
}

double HyperParams::ls(int i)
{
    return exp(kparams()(0, i + 1));
}

mat HyperParams::kparams()
{
    return params(0, 2);
}
