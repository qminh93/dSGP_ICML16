#include "raw.h"

RawData::RawData()
{
    nDim = nData = 0;
    nLim = LOAD_LIMIT_DEFAULT;
    X.clear();
}

RawData::RawData(mat& X)
{
    nDim  = X.n_cols;
    nData = X.n_rows;
    nLim  = LOAD_LIMIT_DEFAULT;
    this->X = X;
}

RawData::RawData(int nLim)
{
    this->X.clear();
    this->nLim = nLim; nDim = nData = 0;
}

RawData::RawData(int nLim, mat &X)
{
    nDim = X.n_cols; nData = X.n_rows;
    this->nLim = nLim;
    this->X = X;
}

RawData::~RawData()
{
    X.clear();
}

void RawData::save(const char* datafile)
{
    FILE* fout = fopen(datafile, "w");

    SFOR(i, X.n_rows) {
        fprintf(fout, "%f", X(i, 0));
        SFOR(j, X.n_cols)
            fprintf(fout, ",%f", X(i, j));
        fprintf(fout, "\n");
    }

    fclose(fout);
}

void RawData::load(const char* datafile)
{
    X.clear(); nDim = nData = 0;
    int lapse = 1000;

    vvd buffer;
    ifstream fin(datafile);
    string line, token;

    cout << "Loading raw data ..." << endl;

    while (getline(fin, line))
    {
        stringstream parser(line);
        vd temp;

        while (getline(parser, token, ','))
            temp.push_back(atof(token.c_str()));
        buffer.push_back(temp);

        nData++;
        if (nData % lapse == 0) cout << nData << " data points have been loaded ..." << endl;
    }

    cout << "Down sampling original data to " << nLim << " data points" << endl;

    vi sample = randsample(nData, nLim);
    X = mat(nLim, buffer[0].size());

    NFOR(i, j, X.n_rows, X.n_cols) X(i, j) = buffer[sample[i]][j];
    SFOR(i, X.n_rows) buffer[i].clear(); buffer.clear(); sample.clear();

    nDim = X.n_cols; nData = X.n_rows;
    cout << "Total number of data points : " << nData << endl;
}

void RawData::normalize_x()
{
	cout << "Normalizing ..." << endl;
	SFOR(i, nDim - 1)
	{
		double xMean = 0.0, dev = 0.0;

		SFOR(j, nData) xMean += X(j, i);
		xMean /= nData;

		SFOR(j, nData) dev += pow(X(j, i) - xMean, 2.0);
		dev = sqrt(dev / nData);

		SFOR(j, nData) X(j, i) = (X(j, i) - xMean) / (dev == 0 ? 1 : dev);
	}
	cout << "Done." << endl;
}

void RawData::normalize_y(pdd &meandev)
{
    double xMean = 0.0, dev = 0.0;

    SFOR(j, nData) xMean += X(j, nDim - 1);
    xMean /= nData;

    SFOR(j, nData) dev += pow(X(j, nDim - 1) - xMean, 2.0);
	dev = sqrt(dev / nData);

	SFOR(j, nData)
	{
		X(j, nDim - 1) = (X(j, nDim - 1) - xMean) / (dev == 0 ? 1 : dev);
	}

	meandev = pdd(xMean, dev);
}


