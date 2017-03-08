#include "regressor.h"

Regressor::Regressor(int nThread, int band_size, OrganizedData *od, HyperParams* hyp, Waves *tsunami, mat *qfs)
{
	this->band_size = band_size;
    this->od        = od;
    this->ker       = new Kernel(hyp);
    this->tsunami   = tsunami;
    this->qfs       = qfs;
    this->nThread   = nThread;

    KSS_inv = mat(od->nSupport, od->nSupport);
    ker->kmat(od->getxm(), KSS_inv);
    inv_sympd(KSS_inv, KSS_inv);

    M 	= vm(od->nBlock);
    Mt 	= vvm(od->nBlock);

    SFOR(i, od->nBlock)
    {
    	M[i]  = zeros <mat> (od->tSize, 1);
    	Mt[i] = vm(nThread);
    }

    NFOR(i, t, od->nBlock, nThread) Mt[i][t] = zeros <mat> (od->tSize, 1);
}

Regressor::~Regressor()
{
	NFOR(i, t, od->nBlock, nThread) Mt[i][t].clear();
    SFOR(i, od->nBlock) { M[i].clear(); Mt[i].clear();}
    SFOR(i, KSD.size()) KSD[i].clear();
    M.clear(); Mt.clear(); KSD.clear();
}

void Regressor::prepare_S(int t, int k, int kB_pos, mat &SBB)
{
    int fr, fc, band = kB_pos - k;
    mat Kij(od->bSize, od->bSize);
    SBB.clear();
    SBB = mat((band + 1) * od->bSize, (band + 1) * od->bSize);

    SFOR(i, band + 1)
    for (int j = i; j < band + 1; j++)
    {
        fr  = i * od->bSize;
        fc  = j * od->bSize;

        int ik = i + k, jk = j + k;
        ker->kmat(od->getxb(ik), od->getxb(jk), Kij);
        SBB.submat(fr, fc, fr + od->bSize - 1, fc + od->bSize - 1) = Kij - KSD[t][i].t() * KSS_inv * KSD[t][j];
        SBB.submat(fc, fr, fc + od->bSize - 1, fr + od->bSize - 1) = SBB.submat(fr, fc, fr + od->bSize - 1, fc + od->bSize - 1).t();
    }

    SBB.diag() += pow(ker->hyp->noise(), 2.0);
}

void Regressor::slide_S(int t, int k, int kB_pos, mat &SBB)
{
    int fc, fr, lr = (kB_pos - k) * od->bSize - 1;
    mat temp;

    if ((kB_pos - k) < band_size)
    {
        temp = SBB.submat(od->bSize, od->bSize, SBB.n_rows - 1, SBB.n_cols - 1);
        SBB.clear(); SBB  = temp;
        temp.clear();
        return;
    }

    SBB.submat(0, 0, lr, lr) = SBB.submat(od->bSize, od->bSize, SBB.n_rows - 1, SBB.n_cols - 1);
    temp = mat(od->bSize, od->bSize);
    fc   = band_size * od->bSize;
    SFOR(i, band_size + 1)
    {
        fr = i * od->bSize;
        int ik = i + k;

        ker->kmat(od->getxb(ik), od->getxb(kB_pos), temp);
        temp -= KSD[t][i].t() * KSS_inv * KSD[t].back();

        if (i == band_size) temp.diag() += pow(ker->hyp->noise(), 2.0);
        else SBB.submat(fc, fr, fc + od->bSize - 1, fr + od->bSize - 1) = temp.t();

        SBB.submat(fr, fc, fr + od->bSize - 1, fc + od->bSize - 1) = temp;
    }
    temp.clear();
}

void Regressor::prepare_KSD(int t, int k, int kB_pos)
{
    mat KSDi(od->nSupport, od->bSize);
    for (int i = k; i <= kB_pos; i++)
    {
        ker->kmat(od->getxm(), od->getxb(i), KSDi);
        KSD[t].push_back(KSDi);
    }
    KSDi.clear();
}

void Regressor::slide_KSD(int t, int k, int kB_pos)
{
    if (!KSD[t].empty()) {
        KSD[t].front().clear();
        KSD[t].pop_front();
    }

    if ((kB_pos - k) < band_size) return;

    mat KSDnew(od->nSupport, od->bSize);
    ker->kmat(od->getxm(), od->getxb(kB_pos), KSDnew);
    KSD[t].push_back(KSDnew);
    KSDnew.clear();
}

void Regressor::extract_S(int t, int k, mat &SBB, mat &Sbb_inv, mat &Sbk, mat &Skk)
{
	Skk.clear(); Sbk.clear(); Sbb_inv.clear();

	Skk = SBB.submat(0, 0, od->bSize - 1, od->bSize - 1);
	mat temp = Skk.i() * ((*od->getyb(k) - od->mean()) - KSD[t][0].t() * KSS_inv * ((*qfs) - od->mean()));
	if (k == od->nBlock - 1 || band_size == 0)
	{
		SFOR(p, od->nBlock)
			Mt[p][t] += tsunami->SigmaUD(p, k) * temp;
		temp.clear(); return;
	}
    Sbk = SBB.submat(od->bSize, 0, SBB.n_rows - 1, od->bSize - 1);
    inv_sympd(Sbb_inv, SBB.submat(od->bSize, od->bSize, SBB.n_rows - 1, SBB.n_cols - 1));
    temp = inv(Skk - Sbk.t() * Sbb_inv * Sbk) * ((*od->getyb(k) - od->mean()) - KSD[t][0].t() * KSS_inv * ((*qfs) - od->mean()));
    SFOR(p, od->nBlock)
    	Mt[p][t] += tsunami->SigmaUD(p, k) * temp;

    temp.clear();
}

void Regressor::compute_mu(int t, int k, int kB_pos, mat &mu, mat &mu_k)
{
    mu.clear();
    mu   = mat((kB_pos - k) * od->bSize, 1);

    SFOR(i, kB_pos - k)
    {
        int ik = i + k + 1;
        mu.submat(i * od->bSize, 0, (i + 1) * od->bSize - 1, 0) = ((*od->getyb(ik)) - KSD[t][i + 1].t() * KSS_inv * ((*qfs) - od->mean()));
    }

    mu -= od->mean();
    mu_k 	= ((*od->getyb(k)) - od->mean()) - KSD[t][0].t() * KSS_inv * (*qfs - od->mean());
}

void Regressor::extract_KuD(int p, int k, int kB_pos, mat &KuDb)
{
    KuDb.clear();
    KuDb = mat(od->tSize, (kB_pos - k) * od->bSize);
    SFOR(i, kB_pos - k)
        KuDb.submat(0, i * od->bSize, od->tSize - 1, (i + 1) * od->bSize - 1) = tsunami->SigmaUD(p, k + i + 1);
}

void Regressor::predict()
{
    KSD = vector < deque < mat > > (nThread);
    SFOR(i, nThread) KSD[i] = deque < mat > (0);
    int chunk = (od->nBlock > nThread ? od->nBlock / nThread : 1);
    if (od->nBlock % nThread > 0) chunk++;

    #pragma omp parallel for schedule(static, 1)
    SFOR(t, nThread)
    {
        mat SBB, Sbb_inv, Sbk, Skk, Tk, buffer1, buffer2, mu, mu_k, KuDb;
        int first_blk = t * chunk,
            last_blk  = min(first_blk + chunk - 1, od->nBlock - 1);

        if (t == nThread - 1) last_blk = od->nBlock - 1;

        for (int k = first_blk; k <= last_blk; k++)
        {
            int kB_pos = min(od->nBlock - 1, k + band_size);

            if (band_size == 0)
            {
            	mat KSk(od->nSupport, od->bSize), Kkk(od->bSize, od->bSize);
            	ker->kmat(od->getxm( ), od->getxb(k), KSk);
            	ker->kmat(od->getxb(k), Kkk);
            	if (KSD[t].size() < 1) KSD[t].push_back(KSk);
            	else {
            		KSD[t][0].clear();
            		KSD[t][0] = KSk;
            	}
            	SBB = Kkk - KSk.t() * KSS_inv * KSk;
            	SBB.diag() += pow(od->noise(), 2.0);
            	KSk.clear(); Kkk.clear();
            }
            else if (k == first_blk) {
                prepare_KSD(t, k, kB_pos);
                prepare_S(t, k, kB_pos, SBB);
            }
            else {
                slide_KSD(t, k, kB_pos);
                slide_S(t, k, kB_pos, SBB);
            }

            extract_S(t, k, SBB, Sbb_inv, Sbk, Skk);

            if (k == od->nBlock - 1 || band_size == 0) continue;
            compute_mu(t, k, kB_pos, mu, mu_k);

            Tk  = Sbb_inv * Sbk;
            buffer1 = inv(Skk - Tk.t() * Sbk) * Tk.t();
            buffer2 = Tk * buffer1 * mu;

            SFOR(p, od->nBlock)
            {
                extract_KuD(p, k, kB_pos, KuDb);
                Mt[p][t] += KuDb * (buffer2 - buffer1.t() * mu_k) - tsunami->SigmaUD(p,k) * buffer1 * mu;
            }
        }

        SBB.clear(); 		Sbb_inv.clear();
        Sbk.clear(); 		Skk.clear();
        Tk.clear();  		buffer1.clear();
        buffer2.clear(); 	mu_k.clear();
        mu.clear();  		KuDb.clear();
    }

    NFOR(i, t, od->nBlock, nThread) M[i] += Mt[i][t];
}

