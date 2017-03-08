/*
 * kmean.cpp
 *
 *  Created on: 1 Dec, 2014
 *      Author: nghiaht
 */

#include "kmean.h"

KMean::KMean(RawData *data)
{
	this->nIter = KMEAN_ITERATION_DEFAULT;
	this->nThread = omp_get_num_procs();
	this->data = data;
	this->nBlock = data->nData;
	this->nQuota = 0;
	this->t1 = this->t2 = time(NULL);
}

KMean::KMean(int nIter, RawData *data)
{
	this->nIter = nIter;
	this->nThread = omp_get_num_procs();
	this->data = data;
	this->nQuota = 0;
	this->nBlock = this->data->nData;
	this->t1 = this->t2 = time(NULL);
}

KMean::~KMean()
{
	SFOR(i, C.size()) C[i].clear();
	SFOR(i, member.size()) member[i].clear();
	C.clear(); member.clear(); nAssign.clear(); nCount.clear();
}

void KMean::allocate()
{
	SFOR(t, nBlock)
    {
		this->member[t].clear(); // free memory
		this->nCount[t] = 0; // reset the counter
    }

	int chunk = this->data->nData / this->nThread;
	if (chunk == 0) chunk++;

	#pragma omp parallel for schedule(dynamic, chunk)
	SFOR(i, this->data->nData)
	{
		this->nAssign[i] = -1;
		double closest = INFTY;
		SFOR(j, nBlock)
        {
            double dist = Dist(j, i);
            if (dist < closest)
            {
                closest = dist;
                this->nAssign[i] = j;
            }
        }
	}

	SFOR(i, this->data->nData)
		this->member[this->nAssign[i]].push_back(i);

}

void KMean::reestimate()
{
	int chunk = this->nBlock / this->nThread;
	if (chunk == 0) chunk++;

	#pragma omp parallel for schedule(dynamic, chunk)
	SFOR(t, nBlock)
    {
		if ((int) this->member[t].size() == 0)
		{
			int pos = IRAND(0, this->data->nData - 1);
			C[t].clear(); // free memory
			rowvec R = data->X.row(pos);
			C[t] = r2v(R);
		}
		else
		{
			C[t].clear(); // free memory
			C[t] = vector <double> (data->nDim, 0.0);
			NFOR(i, j, (int) member[t].size(), data->nDim)
                C[t][j] += data->X(member[t][i], j);
			SFOR(i, data->nDim)
                C[t][i] /= (double) member[t].size();
		}
    }
}

void KMean::initialize()
{
	//SEED(SEED_DEFAULT);
	vector <int> mark(data->nData, 0);

	nAssign.clear(); nCount.clear(); // free memory
	nAssign = vector <int> (data->nData, 0);
	nCount = vector <int> (this->nBlock, 0);

	SFOR(i, (int)C.size()) C[i].clear(); C.clear(); // free memory
	SFOR(i, (int)member.size()) member[i].clear(); member.clear(); // free memory

	member = vector < vector <int> > (nBlock);
	int pos;

	SFOR(i, nBlock)
	{
		pos = IRAND(0, data->nData - 1);
		while (mark[pos] > 0)
			pos = IRAND(0, data->nData - 1);
		mark[pos] = 1;
		rowvec R = data->X.row(pos);
		C.push_back(r2v(R));
	}

	mark.clear(); // free memory
}

Partition* KMean::cluster(int nBlock)
{
	this->nBlock = nBlock; this->t1 = time(NULL);
	this->nQuota = this->data->nData / this->nBlock; // we assume nData % nBlock = 0 -- have to preprocess this externally

	cout << "Initializing Clusters" << endl;
	initialize();

	for (int t = 0; t < this->nIter; t++)
	{
	    cout << "Clustering Iteration " << t + 1 << endl;
		allocate();
		reestimate();
	}

	cout << "Almost done! Time to make clusters even!" << endl;

    vi temp; temp.clear();

    for (int thisBlk = 0; thisBlk < this->nBlock; thisBlk++)
    {
        int bSize = this->member[thisBlk].size();
        if (bSize <= nQuota) continue;

        vector < pair<double, int> > distvec; distvec.clear();

        SFOR(i, bSize)
            distvec.push_back(pair<double,int>(Dist(thisBlk, this->member[thisBlk][i]), this->member[thisBlk][i]));
        sort(distvec.begin(), distvec.end());

        // Remove extra points to the temporary vector and shrink the partition to the pre-defined quota size.
        SFOR(i, bSize) {
            if (i < nQuota) this->member[thisBlk][i] = distvec[i].second;
            else { temp.push_back(distvec[i].second); this->member[thisBlk].pop_back(); }
        }
    }

    SFOR(i, temp.size())
    {
        int assignTo = -1;
        double best  = INFTY;

        SFOR(j, this->nBlock) if ((int) this->member[j].size() < nQuota)
            if (assignTo == -1 || Dist(j, temp[i]) < best) {
                assignTo = j;
                best = Dist(j, temp[i]);
            }
        this->member[assignTo].push_back(temp[i]);
    }

	this->t2 = time(NULL);

	cout << "Done! Clustering Time = " << (double) (t2 -t1) << endl;

	Partition *result = new Partition(this->nBlock, this->C, this->member, this->nAssign);

	return result;
}

double KMean::Dist(int i, int j)
{
	double dist = 0.0;
	SFOR(t, data->nDim)
		dist += pow(C[i][t] - data->X(j, t), 2.0);
	return sqrt(dist);
}

