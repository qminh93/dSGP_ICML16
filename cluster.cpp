#include "cluster.h"

Cluster::Cluster(int first_blk, int last_blk, int cluster_size, bool dtc, OrganizedData *od, Kernel *ker, Term *KSS, Term *KSS_inv)
{
    this->first_blk     = first_blk;
    this->last_blk      = last_blk;
    this->cluster_size  = cluster_size;
    this->dtc           = dtc;
    this->od            = od;
    this->ker           = ker;
    this->KSS           = KSS;
    this->KSS_inv       = KSS_inv;

    Phi    = 0.0;
    logPDD = 0.0;
    cmean  = zeros <mat> (od->nSupport, 1);
    cvar   = zeros <mat> (od->nSupport, od->nSupport);
    Alpha  = new Term (od->nSupport, 1, od->nDim); Alpha->init(0.0);
    Beta   = new Term (od->nSupport, od->nSupport, od->nDim); Beta->init(0.0);
    Delta  = new eterm(od->nDim);
    delta  = new eterm(od->nDim);
    e	   = new eterm(od->nDim);
}

Cluster::~Cluster()
{
    clearall();
}

void Cluster::clearall()
{
    SBB.clear();
    KSD.clear();

    delete Alpha;
    delete Beta;
    delete Delta;
    delete delta;
    delete e;
}

void Cluster::compute_Tk()
{
	// **********
	// Compute Tk  = -	Sbb_inv * Sbk
	// 		   dTk = 	Sbb_inv * ( dSbb * Sbb_inv * Sbk - dSbk )
	// ----------
	// Step 0 : Prepare Term Tk
	Tk = new Term(od->bSize * cluster_size, od->bSize, od->nDim);

	// Step 1 : Tk = Sbb_inv * Sbk
	Tk->me = Sbb_inv * Sbk->me;

	// Step 2 : dTk = dSbb * Sbb_inv * Sbk
	Sbb->g_mulright(Tk->me, Tk);

	// Step 3 : dTk = dSbb * Sbb_inv * Sbk - dSbk
	Term::g_sub(Tk, Sbk, Tk);

	// Step 4 : dTk = Sbb_inv * (dSbb * Sbb_inv * Sbk - dSbk)
	Tk->g_mulleft(Sbb_inv, Tk);

	// Step 5 : Tk  = - Sbb_inv * Sbk
	Tk->me = - Tk->me;

	// Memory control
	delete Sbb;
	Sbb_inv.clear();
}

void Cluster::compute_Gkk()
{
	// **********
	// Step 0: Prepare
	Gkk = new Term(od->bSize, od->bSize, od->nDim);

	// Step 1: Compute Gkk = Skk - Skb * Sbb_inv * Sbk = Skk + Skb * Tk
	Term::mul2(Skb, Tk, Gkk);
	Term::add(SBB(0,0), Gkk, Gkk);
    Gkk->invert(true, Gkk);

	// Side product:
	chol(Uk, Gkk->me);
	SFOR(i, od->bSize)
	logPDD += 2.0 * log(Uk(i, i));

	// Memory control
	delete Sbk;
	Uk.clear();
}

void Cluster::compute_y()
{
	double mean = od->mean();
	int blk;

	yk 	= (*od->getyb (first_blk)) - mean;
	ykt = (*od->getybt(first_blk)) - mean;
	yb 	= mat(od->bSize * cluster_size, 1);
	ybt = mat(1, od->bSize * cluster_size);

	SFOR(i, cluster_size) {
		blk = first_blk + i + 1;
		yb.submat (i * od->bSize, 0, (i + 1) * od->bSize - 1, 0) = (*od->getyb (blk));
		ybt.submat(0, i * od->bSize, 0, (i + 1) * od->bSize - 1) = (*od->getybt(blk));
	}

	yb  -= mean;
	ybt -= mean;
}

void Cluster::combine_KSD()
{
	KSDb = new Term(od->nSupport, od->bSize * cluster_size, od->nDim);
	KSDb-> init();
	KSDb-> dnoise.fill(0.0);

	SFOR(i, cluster_size)
	{
		KSDb->me.submat		(0, i * od->bSize, od->nSupport - 1, (i + 1) * od->bSize - 1) = KSD[i + 1]->me;
	    KSDb->dsignal.submat(0, i * od->bSize, od->nSupport - 1, (i + 1) * od->bSize - 1) = KSD[i + 1]->dsignal;
	   	SFOR(v, od->nDim)
	    KSDb->dls[v].submat (0, i * od->bSize, od->nSupport - 1, (i + 1) * od->bSize - 1) = KSD[i + 1]->dls[v];
	}
}

void Cluster::compute_alpha_beta()
{
	// **********
	// Compute left  = KSDb * Tk,
	//         right = Tk_t * (yb - mean)
	// ----------
	// Step 0:	Prepare
	alpha = new Term(od->nSupport, 1, od->nDim);
	beta  = new Term(od->nSupport, od->nSupport, od->nDim);
	left  = new Term(od->nSupport, od->bSize, od->nDim);
	right = new Term(1, od->bSize, od->nDim);

	compute_y();
	combine_KSD();

	// Step 1: 	Initialize
	// left  = KSDb * Tk
	// right = Tk_t * (yb - mean)
	Term::mul2(KSDb, Tk, left);
	Tk->mulleft(ybt, right);
	right->transpose();

	// Step 2: 	Add
	// left  = left  + KSDk ,
	// right = right + (yk - mean)
	Term::add(left, KSD[0], left);
	right->add(yk, right);

    // Memory management
	delete KSDb;

	// Step 3:	Compute
	// alpha = left * Gkk * right
	// beta  = left * Gkk * left_t

	Term::mul2(left, Gkk, beta);
	//Term::mul3(left, Gkk, right, alpha);
	Term::mul2(beta, right, alpha); // maybe it will get faster here !
	left->transpose();
	Term::mul2(beta, left, beta);

	// Step 4: 	Add
	// Alpha = Alpha + alpha
	// Beta  = Beta  + beta
	Term::add(Alpha, alpha, Alpha);
	Term::add(Beta, beta, Beta);

	// Memory control
	delete left;
	delete right;
	delete alpha;
	delete beta;
	// **********
}

void Cluster::compute_delta()
{
	// *********
	// Compute delta = sum_{i,j} [dTr(Gik * Gkk_inv * Gkj * Wji) - Tr(Sij * dPji)]
	// 				 = d0 + d1 + d2 + d3 - Tr(dPij * Sji)
	// d0: i = 0, j = 0; d0 = dTr(Gkk * Wkk) = dTr(Gkk * Skk) - dTr(noise^2 * Gkk) + dTr(ykt * Gkk * yk)
	// d1: i = 0, j > 0; d1 = dTr(Gkb * Wbk) = dTr(I) - dTr(Gkk * Skk) + dTr(ybt * Tk * Gkk * yk)
	// d2: i > 0, j = 0; d2 = dTr(Gbk * Wkb) = dTr(I) - dTr(Gkk * Skk) + dTr(ybt * Tk * Gkk * yk)
	// d3: i > 0, j > 0; d3 = dTr(Gkb * Wbb * Gbk * Gkk_inv)  = dTr(Gkk * Skk) - dTr(I) - dTr(noise^2 * Tk_t * Gkk * Tk) + dTr(ybt * Tk * Gkk * Tk_t * yb)
	// ----------
	// Step 0: Prepare
	left    = new Term(1, od->bSize, od->nDim);
	right   = new Term(od->bSize, 1, od->nDim);

	// Step 1: Tr(I)
	delta->me += od->bSize;

	// Step 2: Tr(noise^2 * Gkk)
	Gkk->tr(e);
	e->mul(pow(od->noise(), 2.0), e);
	e->dnoise += e->me * (2.0 / od->noise());
	delta->subright(e, delta);

	// Step 3: Tr(yk_t * Gkk * yk)
	Gkk->mulright(yk, right);
	Term::tracemat(ykt, right, e);
	delta->add(e, delta);
	e->reset();

	// Step 4: 2 * Tr(yb_t * Tk * Gkk * yk)
	Tk->mulleft(ybt, left);
	Term::trace2(left, right, e);
	e->mul(2.0, e);
	delta->add(e, delta);

	// Step 5:  Tr(noise^2 * Tk_t * Gkk * Tk)
	Term::tracet3(Tk, Gkk, e);
	e->mul(pow(od->noise(), 2.0), e);
	e->dnoise += e->me * (2.0 / od->noise());
	delta->subright(e, delta);

	// Step 6: Tr(Gkk * Tk_t * yb * yb_t * Tk)
	Term::tracet3(left, Gkk, e);
	delta->add(e, delta);

	// Step 6.5: Memory control
	delete left;
	delete right;
	yk.clear();
	yb.clear();
	ybt.clear();
	ykt.clear();

	// Side product:
	Phi += delta->me;

	// Step 7: Tr(Sij * dPji) = - Tr(Pij * dSji) = - Tr(Gik * Gkk_inv * Gkj * dSji) = - Tr(Gkk * dSkk) - 2 * Tr((Tk * Gkk)_t * dSkb) - Tr(Tk(i) * Gkk * Tk(j)_t * dSji)
	// 7.1: - Tr(Gkk * dSkk)
	Term::tracemat(Gkk->me , SBB(0, 0), e);
	delta->add(e, delta);
	// 7.2: 2 * Tr((Tk * Gkk)_t * dSkb)
	mat temp_ij, temp = Tk->me * Gkk->me; //temp = temp.t();
	Term::tracemat(temp, Skb, e);
	e->mul(2.0, e);
	delta->add(e, delta);
	// 7.3: - Tr(Tk(i) * Gkk * Tk(j)_t * dSji)
	temp = temp * Tk->me.t();
	int fr, fc;

	SFOR(i, cluster_size)
	{
		fr = i * od->bSize;
		temp_ij = temp.submat(fr, fr, fr + od->bSize - 1, fr + od->bSize - 1);
		Term::tracemat(temp_ij, SBB(i + 1, i + 1), e);
		delta->add(e, delta);
		temp_ij.clear();
		for (int j = 0; j < i; j++)
		{
			fc = j * od->bSize;
			temp_ij = temp.submat(fr, fc, fr + od->bSize - 1, fc + od->bSize - 1);
			Term::tracemat(temp_ij, SBB(j + 1, i + 1), e);
			e->mul(2.0, e);
			delta->add(e, delta);
			temp_ij.clear();
		}
	}

	// Step 7.5: Memory control
	delete Skb;
	delete Gkk;
	delete Tk;
	temp.clear();

	// Step 8: Add deltas to Delta
	Delta->add(delta, Delta);

	// Reset bucket
	delta->reset();
	e	 ->reset();
}

void Cluster::process_dtc()
{
    Term *Kkk   =  new Term(od->bSize, od->bSize, od->nDim);
	KSDi_t  	=  new Term(od->nSupport, od->bSize, od->nDim);
	Term *temp  =  new Term(od->nSupport, od->bSize, od->nDim);
	alpha   	=  new Term(od->nSupport, 1, od->nDim);
	beta    	=  new Term(od->nSupport, od->nSupport, od->nDim);
	yk 			=  (*od->getyb (first_blk)) - od->mean();
	ykt			=  (*od->getybt(first_blk)) - od->mean();

    // Compute alpha = KSD[0] * Gkk * yk
	//		   beta  = KSD[0] * Gkk * KSD[0]_t
	KSD[0]->mulconst(1.0 / SQR(od->noise()), temp);
	temp->dnoise -= (2.0 / od->noise()) * temp->me;
	KSD[0]->transpose(KSDi_t);
	Term::mul2(temp, KSDi_t, beta);
	temp->mulright(yk, alpha);

	// Compute Alpha = Alpha + alpha
	//		   Beta  = Beta  + beta
	Term::add(Alpha,alpha,Alpha);
	Term::add(Beta,beta,Beta);

    // Compute delta = delta - dTr(Gkk * KkS * KSS_inv * KSk) = delta - dTr(beta * KSS_inv)
	Term::trace2(beta, KSS_inv, e);
	delta->subright(e, delta);

	// Memory control
	delete KSDi_t;
	delete alpha;
	delete beta;
    delete temp;

	// Compute delta = d0 - Tr(Skk * dPkk)
	// d0 = dTr(Gkk * Wkk) = dTr(Gkk * Kkk) - dTr(Gkk * Kks * Kss_inv * Ksk) + dTr(Gkk * yk * ykt)
	// d0 = pow(noise,-2) * [dTr(Kkk) - dTr(Kks * Kss_inv * Ksk) (computed) + dTr(yk * ykt)]

	ker->kTerm(od->getxb(first_blk), Kkk);
	Kkk->tr(e);
	e->me += dot(yk, yk);
	e->mul(1.0 / SQR(od->noise()), e);
	e->dnoise -= (2.0 / od->noise()) * e->me;
	delta->add(e, delta);

    delete Kkk;
	yk.clear();
	ykt.clear();

	// Side product
	Phi += delta->me;
	logPDD -= 2.0 * od->bSize * log(od->noise());

    // Compute Tr(Skk * dPkk) = -Tr(Pkk * dSkk) = -Tr(Gkk * dSkk)
	delta->dnoise += 2.0 * od->bSize / od->noise();

	Delta->add(delta,Delta);

	// Reset bucket
	delta -> reset();
	e	  -> reset();

	update_dtc();
}

void Cluster::process_last_cluster()
{
	Gkk 		=  new Term(od->bSize, od->bSize, od->nDim);
	KSDi_t  	=  new Term(od->nSupport, od->bSize, od->nDim);
	right		=  new Term(od->bSize, 1, od->nDim);
	alpha   	=  new Term(od->nSupport, 1, od->nDim);
	beta    	=  new Term(od->nSupport, od->nSupport, od->nDim);
	yk 			=  (*od->getyb (first_blk)) - od->mean();
	ykt			=  (*od->getybt(first_blk)) - od->mean();
	SBB(0, 0) 	-> invert(true, Gkk);

	// Compute alpha = KSD[0] * Gkk * yk
	//		   beta  = KSD[0] * Gkk * KSD[0]_t
	Term::mul2(KSD[0], Gkk, alpha);
	KSD[0]->transpose(KSDi_t);
	Term::mul2(alpha, KSDi_t, beta);
	alpha->mulright(yk, alpha);

	// Compute Alpha = Alpha + alpha
	//		   Beta  = Beta  + beta
	Term::add(Alpha,alpha,Alpha);
	Term::add(Beta,beta,Beta);

	// Memory control
	delete KSDi_t;
	delete alpha;
	delete beta;

	// Compute delta = d0 - Tr(Skk * dPkk)

	// d0 = dTr(Gkk * Wkk) = dTr(Gkk * Skk) - dTr(noise^2 * Gkk) + dTr(yk_t * Gkk * yk)
	// Step 1: Tr(d(Gkk * Skk)) = Tr(dI)
	delta->me += od->bSize;

	// Step 2: Tr(d(noise^2 * Gkk))
	Gkk->tr(e);
	e->mul(pow(od->noise(),2.0), e);
	e->dnoise += e->me * (2.0 / od->noise());
	delta->subright(e, delta);

	// Step 3: Tr(yk_t * Gkk * yk)
	Gkk->mulright(yk, right);
	Term::tracemat(ykt, right, e);
	delta->add(e, delta);

	// Side product
	Phi += delta->me;

	chol(Uk, Gkk->me);
	SFOR(i, od->bSize) logPDD += 2.0 * log(Uk(i, i));

	// Compute Tr(Skk * dPkk) = -Tr(Pkk * dSkk) = -Tr(Gkk * dSkk)
	Term::tracemat(Gkk->me, SBB(0, 0), e);
	delta->add(e, delta);

	// Memory control
	delete Gkk;
	delete right;
	yk.clear();
	ykt.clear();
	Uk.clear();

	// Add delta to Delta
	Delta -> add(delta,Delta);

	// Reset bucket
	delta -> reset();
	e	  -> reset();
	update();
}

void Cluster::process()
{
	ECHO("Preparing Cluster\n");
	if (dtc) prepare_dtc();
    else prepare();

    while (first_blk <= last_blk)
    {
    	cout << "Current Cluster: " << first_blk << endl;
    	if (cluster_size == 0)
    	{
    	    if (dtc) process_dtc();
    		else process_last_cluster();
    		continue;
    	}
        extract();
        compute_Tk();
        compute_Gkk();
        compute_alpha_beta();
        compute_delta();
        update();
    }
}

void Cluster::extract_dtc()
{
    Sbb = new Term(od->bSize * cluster_size, od->bSize * cluster_size, od->nDim);
	Skb = new Term(od->bSize, od->bSize * cluster_size, od->nDim);
	Sbk = new Term(od->bSize * cluster_size, od->bSize, od->nDim);

	Sbb ->init(0.0);
    Skb ->init(0.0);
    Sbk ->init(0.0);

    Sbb->me.diag() += SQR(od->noise());
    Sbb->dnoise.diag() += 2 * od->noise();
    Sbb_inv = zeros <mat> (Sbb->me.n_rows, Sbb->me.n_cols);
    Sbb_inv.diag() += pow(od->noise(), -2.0);
}

void Cluster::extract()
{
	int fr, fc, lr, lc;

	Sbb = new Term(od->bSize * cluster_size, od->bSize * cluster_size, od->nDim);
	Skb = new Term(od->bSize, od->bSize * cluster_size, od->nDim);
	Sbk = new Term(od->bSize * cluster_size, od->bSize, od->nDim);

    Sbb ->init();
    Skb ->init();

    SFOR(j, cluster_size)
    {
    	fr = 0;
    	fc = j * od->bSize;
    	lr = fr + od->bSize - 1;
    	lc = fc + od->bSize - 1;
    	Skb->set_submat(fr, fc, lr, lc, SBB(0, j + 1));
    }

    SFOR(i, cluster_size)
    for (int j = i; j <cluster_size; j++)
    {
    	fr = i * od->bSize;
    	fc = j * od->bSize;
    	lr = fr + od->bSize - 1;
    	lc = fc + od->bSize - 1;
    	Sbb->set_submat	  (fr, fc, lr, lc, SBB(i + 1, j + 1));
    	Sbb->set_submat_t (fc, fr, lc, lr, SBB(i + 1, j + 1));
    }

    Skb->transpose(Sbk);
    inv_sympd(Sbb_inv, Sbb->me);
}

void Cluster::prepare_dtc()
{
    int blk;
	Term *KSDi;

    // Prepare KSD
	while (first_blk + cluster_size >= od->nBlock) cluster_size--;

    SFOR(i, cluster_size + 1)
    {
    	blk = first_blk + i;
        KSDi = new Term(od->nSupport, od->bSize, od->nDim);
        ker -> kTerm(od->getxm(), od->getxb(blk), KSDi);
        KSD.push_back(KSDi);
    }

    // Prepare SBB
    SBB = field < Term* > (cluster_size + 1, cluster_size + 1);
    NFOR(i, j, cluster_size + 1, cluster_size + 1)
    {
        SBB(i, j) = new Term(od->bSize, od->bSize, od->nDim);
        SBB(i, j)->init(0.0);
        if (i == j)
        {
            SBB(i, j) ->me.diag() += SQR(od->noise());
            SBB(i, j) ->dnoise    += 2 * od->noise();
        }
    }
}

void Cluster::prepare()
{
	int blk;
	Term *KSDi;

    // Prepare KSD
	while (first_blk + cluster_size >= od->nBlock) cluster_size--;

    SFOR(i, cluster_size + 1)
    {
    	blk = first_blk + i;
        KSDi = new Term(od->nSupport, od->bSize, od->nDim);
        ker -> kTerm(od->getxm(), od->getxb(blk), KSDi);
        KSD.push_back(KSDi);
    }

    // Prepare SBB
    SBB = field < Term* > (cluster_size + 1, cluster_size + 1);
    SFOR(i, cluster_size + 1)
    {
        KSDi_t  = new Term(od->bSize, od->nSupport, od->nDim);
        KSD[i] -> transpose(KSDi_t);

        for (int j = i; j <= cluster_size; j++)
        {
            SBB(i, j) = new Term(od->bSize, od->bSize, od->nDim);
            compute_S(i, j, KSDi_t, SBB(i, j));
        }
        delete KSDi_t;
    }
}

void Cluster::compute_S(int &i, int &j, Term* KSDi_t, Term* Sij)
{
	int blk_i = first_blk + i,
		blk_j = first_blk + j;
	// **********
	// Compute Sij = Kij - Qij + noise^2 * I

	// Step 1: Compute Kij
    Term *Kij;
    Kij  = new Term(od->bSize, od->bSize, od->nDim);
    ker -> kTerm(od->getxb(blk_i), od->getxb(blk_j), Kij);

    // Step 2: Compute Qij
    Term::mul3(KSDi_t, KSS_inv, KSD[j], Sij);

    // Step 3: Compute Sij
    Term::sub(Kij, Sij, Sij);

    // Step 4: Add noise
    if (i == j)
    {
    	Sij->me.diag()      += pow(od->hyper->noise(), 2.0);
    	Sij->dnoise.diag()  += 2.0 * od->hyper->noise();
    }

    // Memory control
    delete Kij;
}

void Cluster::update_dtc()
{
    if (++first_blk > last_blk)
    {
        delete KSD[0];
        delete SBB(0, 0);
        return;
    }

    ker->kTerm(od->getxm(), od->getxb(first_blk), KSD[0]);
}

void Cluster::update()
{
    if (++first_blk > last_blk)
    {
    	SFOR(i, cluster_size + 1)
		{
    		// Delete KSD
    		delete KSD[i];
    		// Delete SBB
    		for (int j = i; j <= cluster_size; j++) delete SBB(i, j);
		}
    	return;
    }

    delete KSD[0];
    delete SBB(0, 0);

    // Slide KSD, delete previous SBB
    SFOR(i, cluster_size)
    {
    	KSD[i] = KSD[i + 1];
    	delete(SBB(0, i + 1));
    }

    // Slide SBB
    SFOR(i, cluster_size)
    for (int j = i; j < cluster_size; j++)
    {
    	SBB(i, j) = SBB(i + 1, j + 1);
    }

    int blk = first_blk + cluster_size;
    if (cluster_size <= od->nBlock - first_blk - 1)
    {
        // Update KSD
        KSD[cluster_size] = new Term(od->nSupport, od->bSize, od->nDim);
        ker				 -> kTerm(od->getxm(), od->getxb(blk), KSD[cluster_size]);

        // Update SBB
        SFOR(i, cluster_size + 1)
        {
        	KSDi_t  = new Term(od->bSize, od->nSupport, od->nDim);
        	KSD[i] -> transpose(KSDi_t);
        	SBB(i, cluster_size) = new Term(od->bSize, od->bSize, od->nDim);
            compute_S(i, cluster_size, KSDi_t, SBB(i, cluster_size));

            // Memory control
            delete KSDi_t;
        }
    }
    else
    {
    	// Decrease cluster size
    	cluster_size--;
    }
}
