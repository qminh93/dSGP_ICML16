#include "wave.h"

Waves::Waves(int band_size, OrganizedData *od, HyperParams *hyp)
{
    this->od        = od;
    this->ker       = new Kernel(hyp);
    this->band_size = band_size;

    KSS_inv = mat(od->nSupport, od->nSupport);

    ker->kmat(od->getxm(), KSS_inv);
    inv_sympd(KSS_inv, KSS_inv);

    wave_band.clear();
    SigmaUD = field < mat > (od->nBlock, od->nBlock);
}

Waves::~Waves()
{
	KSS_inv.clear();
    SFOR(i, wave_band.size()) wave_band[i]->clear(); wave_band.clear();
    NFOR(i, j, od->nBlock, od->nBlock) SigmaUD(i, j).clear(); SigmaUD.clear();
}

// This fn evaluate R_Up_Dq for q \in [pB_neg, pB_pos]
void Waves::ud_exact_eval(int p, int q, mat& Rpq)
{
    mat K_Up_Dq(od->tSize, od->bSize),
        K_Up_S(od->tSize, od->nSupport),
        K_S_Dq(od->nSupport, od->bSize);

    ker->kmat(od->getxt(p), od->getxb(q), K_Up_Dq);
    ker->kmat(od->getxt(p), od->getxm( ), K_Up_S);
    ker->kmat(od->getxm( ), od->getxb(q), K_S_Dq);
    Rpq = K_Up_Dq - K_Up_S * KSS_inv * K_S_Dq;

    K_Up_Dq.clear(); K_Up_S.clear(); K_S_Dq.clear();
}

// This fn evaluate R_Dp_Uq for q \in [pB_neg, pB_pos]
void Waves::du_exact_eval(int p, int q, mat& Rpq)
{
    mat K_Dp_Uq(od->bSize, od->tSize),
        K_Dp_S(od->bSize, od->nSupport),
        K_S_Uq(od->nSupport, od->tSize);

    ker->kmat(od->getxb(p), od->getxt(q), K_Dp_Uq);
    ker->kmat(od->getxb(p), od->getxm( ), K_Dp_S);
    ker->kmat(od->getxm( ), od->getxt(q), K_S_Uq);
    Rpq = K_Dp_Uq - K_Dp_S * KSS_inv * K_S_Uq;

    K_Dp_Uq.clear(); K_Dp_S.clear(); K_S_Uq.clear();
}

// This fn evaluate R_Dp_Dq for q \in [pB_neg, pB_pos]
void Waves::dd_exact_eval(int p, int q, mat& Rpq)
{
    mat K_Dp_Dq(od->bSize, od->bSize),
        K_Dp_S(od->bSize, od->nSupport),
        K_S_Dq(od->nSupport, od->bSize);

    ker->kmat(od->getxb(p), od->getxb(q), K_Dp_Dq);
    ker->kmat(od->getxb(p), od->getxm( ), K_Dp_S);
    ker->kmat(od->getxm( ), od->getxb(q), K_S_Dq);
    Rpq = K_Dp_Dq - K_Dp_S * KSS_inv * K_S_Dq;
    if (q == p) Rpq.diag() += pow(ker->hyp->noise(), 2.0);

    K_Dp_Dq.clear(); K_Dp_S.clear(); K_S_Dq.clear();
}

void Waves::compute_R(int p, mat& R1, mat &R2)
{
    int fr, pB_pos = min(od->nBlock - 1, p + band_size);
    mat* xb = new mat((pB_pos - p) * od->bSize, od->nDim);

    for (int i = p + 1; i <= pB_pos; i++)
    {
        fr = (i - p  - 1) * od->bSize;
        xb->submat(fr, 0, fr + od->bSize - 1, od->nDim - 1) = *od->getxb(i);
    }

    // Compute R_Dpb_Dpb
    mat Kbb(xb->n_rows, xb->n_rows),
        KbS(xb->n_rows, od->nSupport),
        temp, Rbb;

    ker->kmat(xb, Kbb);
    ker->kmat(xb, od->getxm(), KbS);
    temp = KSS_inv * KbS.t();
    Rbb  = Kbb - KbS * temp;
    Rbb.diag() += pow(ker->hyp->noise(), 2.0);
    Rbb = inv_sympd(Rbb);

    Kbb.clear(); KbS.clear();

    // Compute R_Dp_Dpb
    mat Kdb(od->bSize, xb->n_rows),
        KdS(od->bSize, od->nSupport),
        Rdb;
    ker->kmat(od->getxb(p), xb, Kdb);
    ker->kmat(od->getxb(p), od->getxm(), KdS);
    Rdb = Kdb - KdS * temp;

    Kdb.clear(); KdS.clear();

    // Compute R_Up_Dpb
    mat Kub(od->tSize, xb->n_rows),
        KuS(od->tSize, od->nSupport),
        Rub;
    ker->kmat(od->getxt(p), xb, Kub);
    ker->kmat(od->getxt(p), od->getxm(), KuS);
    Rub = Kub - KuS * temp;

    Kub.clear(); KuS.clear(); temp.clear();

    // Compute R1, R2
    R1 = Rdb * Rbb;
    R2 = Rub * Rbb;

    Rdb.clear(); Rub.clear(); Rbb.clear(); delete xb;
}

void Waves::combine_R(int p, mat &Rdu, mat &Rdd)
{
    Rdu.clear(); Rdu = mat(band_size * od->bSize, od->tSize);
    Rdd.clear(); Rdd = mat(band_size * od->bSize, od->bSize);

    int fr;
    SFOR(i, band_size)
    {
        fr = i * od->bSize;
        Rdu.submat(fr, 0, fr + od->bSize - 1, od->tSize - 1) = wave_band[band_size - 1 - i]->RDU[p + i + 1];
        Rdd.submat(fr, 0, fr + od->bSize - 1, od->bSize - 1) = wave_band[band_size - 1 - i]->RDD[p + i + 1];
    }
}

void Waves::process(int nThread)
{
	int chunk = (od->nBlock > nThread ? od->nBlock / nThread : 1);
	if (band_size == 0)
	{
		#pragma omp parallel for schedule(static, chunk)
		SFOR(j, od->nBlock)
		{
			mat KiS(od->tSize, od->nSupport), KSj(od->nSupport, od->bSize);
			ker->kmat(od->getxm(), od->getxb(j), KSj);

			SFOR(i, od->nBlock)
			{
				ker->kmat(od->getxt(i), od->getxm(), KiS);
				if (i == j)
				{
					SigmaUD(i, j) = mat(od->tSize, od->bSize);
					ker->kmat(od->getxt(i), od->getxb(j), SigmaUD(i, j));
				}
				else SigmaUD(i, j) = KiS * KSS_inv * KSj;
			}
			KiS.clear(); KSj.clear();
		}
		return;
	}

	// Compute R1 = R_Dp_Dpb * R_Dpb_Dpb ^ {-1}, R2 = R_Up_Dpb * R_Dpb_Dpb^{-1}
    vm R1(od->nBlock), R2(od->nBlock);
    chunk = ((od->nBlock - 1) > nThread ? (od->nBlock - 1) / nThread : 1);
	#pragma omp parallel for schedule(static, chunk)
    SFOR(p, od->nBlock - 1)
    	compute_R(p, R1[p], R2[p]);

    SFOR(d, od->nBlock)
    {
    	printf("Wave %d\n", d);
        Wave* wd = new Wave(d);

        wd->RUD = vm(od->nBlock - d);
        wd->RDU = vm(od->nBlock - d);
        wd->RDD = vm(od->nBlock - d);

        chunk = ((od->nBlock - d) > nThread ? (od->nBlock - d) / nThread : 1);
        #pragma omp parallel for schedule(dynamic, (chunk + 1) / 2)
        for (int p = 0; p < od->nBlock - d; p++)
        {
        	mat Rud, Rdd, Rdu;
            if (d <= band_size)
            {
                ud_exact_eval(p, p + d, Rud); wd->RUD[p] = Rud;
                du_exact_eval(p, p + d, Rdu); wd->RDU[p] = Rdu;
                dd_exact_eval(p, p + d, Rdd); wd->RDD[p] = Rdd;
                Rud.clear(); Rdd.clear(); Rdu.clear();
                continue;
            }

            combine_R(p, Rdu, Rdd);

            // Compute RUD, RDD, RDU of wd
            Rud = R2[p] * Rdd; wd->RUD[p] = Rud;
            Rdd = R1[p] * Rdd; wd->RDD[p] = Rdd;
            Rdu = R1[p] * Rdu; wd->RDU[p] = Rdu;
            Rud.clear(); Rdd.clear(); Rdu.clear();
        }

		#pragma omp parallel for schedule(dynamic, (chunk + 1) / 2)
        for (int p = 0; p < od->nBlock - d; p++)
        {
            int q = p + d;
            mat KuS(od->tSize, od->nSupport), KSd(od->nSupport, od->bSize);

            ker->kmat(od->getxt(p),od->getxm( ), KuS);
            ker->kmat(od->getxm( ),od->getxb(q), KSd);
            SigmaUD(p, q) = wd->RUD[p] + KuS * KSS_inv * KSd;

            ker->kmat(od->getxt(q),od->getxm( ), KuS);
            ker->kmat(od->getxm( ),od->getxb(p), KSd);
            SigmaUD(q, p) = wd->RDU[p].t() + KuS * KSS_inv * KSd;

            KuS.clear(); KSd.clear();
        }

        if ((int) wave_band.size() >= band_size)
        {
            delete wave_band.front();
            wave_band.pop_front();
        }

        wave_band.push_back(wd);
    }

    SFOR(i, wave_band.size()) wave_band[i]->clear(); wave_band.clear();
    chunk = (od->nBlock > nThread ? od->nBlock / nThread : 1);
    #pragma omp parallel for schedule(static, chunk)
    SFOR(i, od->nBlock)
    {
    	R1[i].clear(); R2[i].clear();
    }
    R1.clear(); R2.clear();
}
