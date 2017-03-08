/*
 * libhead.cpp
 *
 *  Created on: 30 Nov 2015
 *      Author: nghiaht
 */

#include "libhead.h"

string num2str(double x)
{
	return static_cast < ostringstream* > (&(ostringstream() << x))->str();
}

string num2str(int x)
{
	return static_cast < ostringstream* > (&(ostringstream() << x))->str();
}

template<class ForwardIterator, typename T>
void iota(ForwardIterator first, ForwardIterator last, T value)
{
    while(first != last)
    {
        *first++ = value;
        ++value;
    }
}

vi randsample(int popSize, int sampleSize)
{
    vi temp(popSize);
    iota(temp.begin(),temp.end(),0);
    random_shuffle(temp.begin(),temp.end());

    vi result(temp.begin(),temp.begin() + sampleSize);
    return result;
}

double lapse (clock_t tStart, clock_t tEnd) // return the number of CPU time in miliseconds
{
    return (1000.0 * (tEnd - tStart) / CLOCKS_PER_SEC);
}

vd r2v (rowvec &R)
{
    vd r;
    SFOR(i, R.n_elem) r.push_back(R(i));
    return r;
}

vd c2v (colvec &C)
{
    vd c;
    SFOR(i, C.n_elem) c.push_back(C(i));
    return c;
}

mat v2m (vvd &A)
{
    if (A.size() == 0) return mat(0,0);
    mat M((int)A.size(),(int)A[0].size());
    for (int i = 0; i < (int)A.size(); i++)
        for (int j = 0; j < (int)A[0].size(); j++)
            M(i,j) = A[i][j];
    return M;
}

void csv2bin_support(string src,string dest)
{
    ifstream fin(src.c_str());
    string   line, token;
    vvd      Sx, Sy;
    while (getline(fin, line))
    {
        stringstream ss(line);
        vd point, pointy;

        while (getline(ss,token,','))
    		point.push_back(atof(token.c_str()));

        pointy.push_back(point.back());
        point.pop_back();

    	Sx.push_back(point);
    	Sy.push_back(pointy);
    }

    field <mat> M (1,2);
    M(0,0) = v2m(Sx);
    M(0,1) = v2m(Sy);
    M.save(dest.c_str(),arma_binary);
    fin.close();
}

void csv2bin_blkdata(string src,string dest)
{
	ifstream     fin(src.c_str());
	string       line, token;
	vector <vvd> Sx,Sy;
	int          curblk = 1;
	vvd          blkx, blky;
	vd           pointx, pointy;

	while (getline(fin, line))
	{
		stringstream ss(line);
		pointx.clear();
		pointy.clear();

		getline(ss,token,',');
		int blkno = atoi(token.c_str());
		while (curblk < blkno)
		{
			Sx.push_back(blkx); blkx.clear();
			Sy.push_back(blky); blky.clear();
			curblk++;
		}

		while (getline(ss,token,','))
			pointx.push_back(atof(token.c_str()));

        pointy.clear(); pointy.push_back(pointx.back()); pointx.pop_back();
		blkx.push_back(pointx);
		blky.push_back(pointy);
	}

	Sx.push_back(blkx);
	Sy.push_back(blky);

	field <mat> M((int)Sx.size(),2);

	for (int i = 0; i < (int)Sx.size(); i++)
	{
		M(i,0) = v2m(Sx[i]);
		M(i,1) = v2m(Sy[i]);
	}

	M.save(dest.c_str(), arma_binary);
    Sx.clear(); blkx.clear(); pointx.clear();
    Sy.clear(); blky.clear(); pointy.clear();
	fin.close();
}

/**
 * Returns the peak (maximum so far) resident set size (physical
 * memory use) measured in bytes, or zero if the value cannot be
 * determined on this OS.
 */
/*
size_t getPeakRSS()
{
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo( GetCurrentProcess( ), &info, sizeof(info) );
    return (size_t)info.PeakWorkingSetSize;

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
    struct psinfo psinfo;
    int fd = -1;
    if ( (fd = open( "/proc/self/psinfo", O_RDONLY )) == -1 )
        return (size_t)0L;
    if ( read( fd, &psinfo, sizeof(psinfo) ) != sizeof(psinfo) )
    {
        close( fd );
        return (size_t)0L;
    }
    close( fd );
    return (size_t)(psinfo.pr_rssize * 1024L);

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
    struct rusage rusage;
    getrusage( RUSAGE_SELF, &rusage );
#if defined(__APPLE__) && defined(__MACH__)
    return (size_t)rusage.ru_maxrss;
#else
    return (size_t)(rusage.ru_maxrss * 1024L);
#endif

#else
    return (size_t)0L;
#endif
}
*/

/**
 * Returns the current resident set size (physical memory use) measured
 * in bytes, or zero if the value cannot be determined on this OS.
 */
/*
size_t getCurrentRSS( )
{
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo( GetCurrentProcess( ), &info, sizeof(info) );
    return (size_t)info.WorkingSetSize;

#elif defined(__APPLE__) && defined(__MACH__)
    struct mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if ( task_info( mach_task_self( ), MACH_TASK_BASIC_INFO,
        (task_info_t)&info, &infoCount ) != KERN_SUCCESS )
        return (size_t)0L;
    return (size_t)info.resident_size;

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
    long rss = 0L;
    FILE* fp = NULL;
    if ( (fp = fopen( "/proc/self/statm", "r" )) == NULL )
        return (size_t)0L;
    if ( fscanf( fp, "%*s%ld", &rss ) != 1 )
    {
        fclose( fp );
        return (size_t)0L;
    }
    fclose( fp );
    return (size_t)rss * (size_t)sysconf( _SC_PAGESIZE);

#else
    return (size_t)0L;
#endif
}
*/

void shuffle(char* original, char* sample, int nTotal, int nSample)
{
	vi pool(nTotal, 0);
	vi mark(nTotal, 0);
	SFOR(i, nTotal) pool[i] = i;

	int nMax = nTotal - 1;
	SFOR(i, nSample)
	{
		int pos = IRAND(0, nMax), s = pool[pos];
		pool[pos] = pool[nMax]; nMax--;
		mark[s] = 1;
	}

	ifstream cin(original);
	ofstream cout(sample);

	string line;

	SFOR(i, nTotal)
	{
		cin >> line;
		if (mark[i] == 1) cout << line << endl;
	}

	cin.close(); cout.close();
}


