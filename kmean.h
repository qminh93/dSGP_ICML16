/*
 * kmean.h
 *
 *  Created on: 1 Dec, 2014
 *      Author: nghiaht
 */

#ifndef KMEAN_H_
#define KMEAN_H_

#include "libhead.h"
#include "raw.h"

#define KMEAN_ITERATION_DEFAULT 300

class KMean
{
	private :

	vvd C; // store the estimated centroids
	vvi member; // lists of data indices belonging to each cluster
	vi  nAssign; // nAssign[i] -- the cluster which the ith data point belongs to
	vi  nCount; // nCount[i] -- the number of data points already assigned to block i

	int nIter, nThread, nBlock, nQuota;
	RawData *data;

	time_t t1, t2;

	void initialize();
	void allocate();
	void reestimate();
	double Dist(int i, int j);

	public:

	KMean(RawData *data);
	KMean(int nIter, RawData *data);
   ~KMean();

	Partition* cluster(int nBlock);
};

#endif /* KMEAN_H_ */
