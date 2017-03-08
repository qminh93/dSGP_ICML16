#ifndef LIBHEAD_H
#define LIBHEAD_H

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <armadillo>
#include <algorithm>
#include <fstream>
#include <vector>
#include <queue>
#include <deque>
#include <set>
#include <map>
#include <cstring>
#include <ctime>
#include <cmath>
#include <iomanip>
#include <stack>
#include <bitset>
#include <functional>
#include <numeric>

/*#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
#include <unistd.h>
#include <sys/resource.h>

#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
#include <fcntl.h>
#include <procfs.h>

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
#include <stdio.h>

#endif

#else
#error "Cannot define getPeakRSS( ) or getCurrentRSS( ) for an unknown OS."
#endif*/

using namespace std;
using namespace arma;

//#define SEED_DEFAULT 4111987
#define SEED_DEFAULT 20051987
//#define SEED_DEFAULT 26031993
//#define SEED_DEFAULT 8071950
//#define SEED_DEFAULT 10081950

#define SEED(x) srand(x)
#define IRAND(a, b) (rand() % ((b) - (a) + 1) + (a)) // randomly generate an integer from [a, b]
#define DRAND(a, b) (((b) - (a)) * ((double) rand() / (RAND_MAX)) + (a)) // randomly generate a real number from [a, b]
#define EPS 0.00000000000000001
#define PI 3.141592653589793238462
#define EN 2.71828182845904509080
#define INFTY 1000000000000000.0
#define N(m, mean, var) ((1 / sqrt(2 * PI * (var))) * exp(-0.5 * (pow((m) - (mean), 2.0) / (var)))) // the pdf function for N(mean, var = sigma^2)
#define ENT(var) (0.5 * log(2.0 * PI * E * (var))) // the entropy of N(., var)
#define LN(m, mean, var) (-0.5 * log(2 * PI * (var)) - 0.5 * (pow((m) - (mean), 2.0)) / (var) ) // the log pdf function for N(mean, var = sigma^2)
#define SFOR(i,n) for (int i = 0; i < (int)n; i++)
#define NFOR(i,j,n,m) SFOR(i,n) SFOR(j,m)
#define ECHO(z) printf(z)
#define CHUNK(a,b) (a > b ? a / b : 1)
#define SQR(a) ((a) * (a))
#define pdd	 pair 	<double, double>
#define vi   vector < int >
#define vd   vector < double >
#define vm   vector < mat >
#define vs   vector < string >
#define vvi  vector < vi >
#define vvd  vector < vd >
#define vvm  vector < vm >
#define bmat field < mat* >
#define ves vector < ExpSetting* >


struct ExpSetting { int seed, nBlock, nBand, max_sp, sp_per_blk, predict_mode; double pTest; string res_file; };

struct Configuration
{
	#define TD "/* TRAINING DATA */"
	#define RS "/* RUNNING SETTINGS */"
	#define IN "/* INITIALISATION */"
	#define LR "/* LEARNING RATE */"
	#define AS "/* ANYTIME SETTINGS */"
	#define RL "/* RUN LIST */"
	#define CM "//"

	/* FILE DIRECTORIES */
	string  data_file, hyp_file;


	ves exp_list;
	bool bad_cfg;

	/* TRAINING SETTINGS */
	int		nThread, nExp,
			nPoint, // the maximum number of data points to be loaded in memory
			training_mode,			// Training mode: 0->train only length-scales (ls), 1->train ls and noise, 2->train everything
			training_num_ite,		// Number of training iterations
			bcm, seed;              // support = test set or not
	double  rls, rn, rs,
            tau, lambda, decay;  // Learning rate settings

	/* PREDICTION SETTINGS */
	int		anytime_num_ite,
			anytime_num_sample,
			anytime_interval,
			pred_mode;				// predict mode for anytime
	double	alpha, beta, gamma;		// Theta initialisation parameters

	/* HYPER-PARAMETERS INITIALISATION */
	double	signal, noise;			// Original
	vd		ls;						// Original

	void load(char *config)
	{
		bad_cfg = false;
		ifstream cfg(config);
		string header, comment;

		while (getline(cfg, header))
		{
			cout << header << endl;
			comment = CM;
			while (comment.substr(0, 2) == CM) { getline(cfg, comment); if (comment.substr(0, 2) == CM) cout << comment << endl; }
			stringstream input(comment);

			if (header == TD)
			{
				input >> data_file >> nPoint;
				cout  << data_file << " " << nPoint << endl;
			}
			else
			if (header == RS)
			{
				input >> nThread >> training_num_ite >> training_mode >> bcm;
				cout  << nThread << " " << training_num_ite << " " << training_mode << " " << bcm << endl;
			}
			else
			if (header == IN)
			{
				int nDim;
				input >> nDim;
				cout  << nDim << " ";

				if (nDim)
				{
					input >> noise >> signal;
					cout  << noise << " " << signal << endl;
					ls = vd(nDim);
					SFOR(i, nDim) { cfg >> ls[i]; cout << ls[i] << " "; }
					cout << endl;
					getline(cfg, comment);
				}
				else { getline(cfg, hyp_file); cout << hyp_file << endl; }
			}
			else
			if (header == LR)
			{
				input >> rls >> tau >> lambda >> decay >> rn >> rs;
				cout  << rls << " " << tau << " " << lambda << " " << decay << " " << rn << " " << rs << endl;
			}
			else
			if (header == AS)
			{
				// TO BE ADDED LATER
			}
			else
			if (header == RL)
			{
				input >> nExp;
				cout << nExp << endl;
				exp_list = ves(nExp);
				SFOR(i, nExp)
	    		{
	    			exp_list[i] = new ExpSetting();
	    			getline(cfg, comment);

					stringstream exp_input(comment);

					exp_input >> exp_list[i]->nBand  >> exp_list[i]->predict_mode;
					cout  << exp_list[i]->nBand  << " " << exp_list[i]->predict_mode << " ";
					exp_input >> exp_list[i]->nBlock >> exp_list[i]->max_sp >> exp_list[i]->sp_per_blk;
					cout  << exp_list[i]->nBlock << " " << exp_list[i]->max_sp << " " << exp_list[i]->sp_per_blk << " ";
					exp_input >> exp_list[i]->pTest  >> exp_list[i]->seed >> exp_list[i]->res_file;
					cout  << exp_list[i]->pTest  << " " << exp_list[i]->seed << " " << exp_list[i]->res_file << endl;
	    		}

			}
			else
			{
				cout << "SOMETHING WRONG WITH THE CONFIG FILE" << endl;
				bad_cfg = true;
			}
		}
	}
};

struct Partition
{
	int nBlock; // the number of partitions
	vvd C; // store the estimated centroids
	vvi member; // lists of data indices belonging to each cluster
	vi  nAssign; // nAssign[i] -- the cluster which the ith data point belongs to

	Partition(int nBlock, vvd &C, vvi &member, vi &nAssign)
	{
		this->nBlock  = nBlock;
		this->C       = C;
		this->member  = member;
		this->nAssign = nAssign;
	}

	~Partition()
	{
		for (int i = 0; i < (int) C.size(); i++)
			C[i].clear();
		for (int i = 0; i < (int) member.size(); i++)
			member[i].clear();
		C.clear(); member.clear(); nAssign.clear();
	}
};

string  num2str(double x);
string  num2str(int x);
double  lapse (clock_t tStart, clock_t tEnd); // return the CPU time in miliseconds
vi      randsample(int popSize, int sampleSize);
vd      r2v (rowvec &R);
vd      c2v (colvec &C);
mat     v2m (vvd &A);
template <class ForwardIterator, typename T>
void    iota(ForwardIterator first, ForwardIterator last, T value);
void    csv2bin_blkdata(string src,string dest);
void    csv2bin_support(string src,string dest);

/*size_t getPeakRSS();
size_t getCurrentRSS();*/

void shuffle(char* original, char* sample, int nTotal, int nSample);
Configuration* load_configuration(char* config);

#endif /* LIBHEAD_H_ */
