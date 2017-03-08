#include "factory.h"
#include "raw.h"

Factory::Factory(OrganizedData* od, Kernel *ker, int &nThread, int &nBand, bool dtc)
{
    this->nThread        = nThread;
    this->cluster_size   = nBand;
    this->dtc            = dtc;
    this->od             = od;
    this->ker            = ker;
    this->markov_cluster = vector < Cluster* > (nThread);
    this->hp.clear();
    this->mu.clear();
    this->score.clear();
}

Factory::~Factory()
{
    clearall();
    markov_cluster.clear();
}

void Factory::reset()
{
	clearall();
	markov_cluster.clear();
	markov_cluster = vector < Cluster* > (nThread);
}

void Factory::clearall()
{
	KSS->clearall(); 		delete KSS;
	KSS_inv->clearall();	delete KSS_inv;
}

void Factory::precompute()
{
	KSS     =  new Term(od->nSupport, od->nSupport, od->nDim);
	KSS_inv =  new Term(od->nSupport, od->nSupport, od->nDim);
    ker	-> kTerm_par(od->getxm(), KSS, nThread);
	KSS	-> invert(true, KSS_inv);
}

void Factory::process_stream(int &stream_number)
{
    int chunk     = (od->nBlock > nThread ? od->nBlock / nThread : 1),
        first_blk = stream_number * chunk,
        last_blk  = min(first_blk + chunk - 1, od->nBlock - 1);

    if (stream_number == nThread - 1) last_blk = od->nBlock - 1;

    printf("Initializing Stream %d\n", stream_number);
    markov_cluster[stream_number] = new Cluster(first_blk, last_blk, cluster_size, dtc, od, ker, KSS, KSS_inv);
    printf("Processing Stream %d\n", stream_number);
    markov_cluster[stream_number]->process();
    printf("Done!\n");
}

double Factory::DTC_predict(mat *qfs, HyperParams *hyp)
{
    double 	rmse = 0.0;

    Kernel *ker = new Kernel(hyp);
    vd		err(nThread);
    mat 	KSS_inv(od->nSupport, od->nSupport);
    ker->kmat(od->getxm(), KSS_inv);
    inv_sympd(KSS_inv, KSS_inv);

    omp_set_num_threads(nThread);

	#pragma omp parallel for schedule(static, CHUNK(od->nBlock, nThread))
	SFOR(i, od->nBlock)
	{
		int t = omp_get_thread_num();
		mat KUiS(od->tSize, od->nSupport);
		ker->kmat(od->getxt(i), od->getxm(), KUiS);
		mat pred  = KUiS * KSS_inv * ((*qfs) - od->mean()) + od->mean() - (*od->getyt(i));
		err[t] += (1.0 / (double)od->nTest) * dot(pred, pred);
		KUiS.clear();
	}
	KSS_inv.clear();

	SFOR(i, nThread) rmse += err[i]; err.clear();
	return sqrt(rmse);
}

double Factory::PIC_precompute(HyperParams *hyp, mat *qfs)
{
	Kernel *ker = new Kernel(hyp);
	mat 	KSS(od->nSupport, od->nSupport), KSS_inv;

	ker->kmat(od->getxm(), KSS); inv_sympd(KSS_inv, KSS);

	double rmse = 0.0;
    SFOR(i, od->nBlock)
    {
    	mat KtS(od->tSize, od->nSupport),
    		KtB(od->tSize, od->bSize),
			KSB(od->nSupport, od->bSize),
			KBB(od->bSize, od->bSize),
			KBB_inv, R11, R12, R21, R22;

    	ker->kmat(od->getxb(i), KBB);
    	KBB.diag() += pow(od->noise(), 2.0);
    	inv_sympd(KBB_inv, KBB);

    	ker->kmat(od->getxt(i), od->getxm(),  KtS);
    	ker->kmat(od->getxt(i), od->getxb(i), KtB);
    	ker->kmat(od->getxm(),  od->getxb(i), KSB);

    	R11  = inv(KSS - KSB * KBB_inv * KSB.t());
    	R22  = inv(KBB - KSB.t() * KSS_inv * KSB);
    	R21  = - KBB_inv * KSB.t() * R11;
    	R12  = R21.t();

    	mat Mi = (KtS * R11 + KtB * R21);
    	mat bi = (KtS * R12 + KtB * R22) * ((*od->getyb(i)) - od->mean());

	mat pred = Mi * ((*qfs) - od->mean()) + bi + od->mean() - (*od->getyt(i));
        rmse += dot(pred, pred);

    	KtS.clear(); KtB.clear(); KSB.clear(); KBB.clear(); KBB_inv.clear();
    	R11.clear(); R12.clear(); R21.clear(); R22.clear(); Mi.clear(); bi.clear(); pred.clear();
    }

    KSS.clear(); KSS_inv.clear();
    return sqrt(rmse / od->nTest);
}

double Factory::PIC_predict(mat *qfs, vm &M, vm &b)
{
    double 	rmse = 0.0;
    vd		err(nThread);

	#pragma omp parallel for schedule(static, CHUNK(od->nBlock, nThread))
	SFOR(i, od->nBlock)
	{
		int t = omp_get_thread_num();
		mat pred  = M[i] * ((*qfs) - od->mean()) + b[i] + od->mean() - (*od->getyt(i));
		err[t] += dot(pred, pred);
	}

	SFOR(i, nThread) rmse += err[i];
	rmse /= (od->tSize * od->nBlock);

	err.clear();
	return sqrt(rmse);
}

Waves* Factory::prepare_waves(HyperParams *hyp)
{
	Waves* tsunami = new Waves(cluster_size, od, hyp);
	tsunami->process(nThread);

	return tsunami;
}

double Factory::bcm_predict(mat *qfs)
{
	double rmse = 0.0, diff = 0.0;

	SFOR(i, od->nTest)
	{
		diff = qfs->at(i, 0) - od->support(0, 1)->at(i, 0);
		rmse += (diff * diff) * 1.0 / od->nTest;
	}

	rmse = sqrt(rmse);

	return rmse;
}

double Factory::predict(mat *qfs, HyperParams *hyp, Waves *tsunami)
{
	cout << "Initializing Predictor ..." << endl;
    Regressor* reg = new Regressor(nThread, cluster_size, od, hyp, tsunami, qfs);

    cout << "Starting ..." << endl;
    reg->predict();

    cout << "Evaluating error ..." << endl;
    double rmse = 0.0;
    vd rmse_local(nThread, 0.0);

    #pragma omp parallel for schedule(static, CHUNK(od->nBlock, nThread))
    SFOR(p, od->nBlock)
    {
        vec temp = (reg->M[p] - (*od->getyt(p)) + od->mean());
        int id = (int) omp_get_thread_num();
        rmse_local[id] += dot(temp, temp);
        temp.clear();
    }

    SFOR(t, nThread) rmse += rmse_local[t] / (od->tSize * od->nBlock);

    // Memory control
    rmse_local.clear();
    delete reg;

    return sqrt(rmse);
}

void Factory::compute_R(eterm *R)
{
    GammaSS     = new Term (od->nSupport, od->nSupport, od->nDim);
    GammaSS_inv = new Term (od->nSupport, od->nSupport, od->nDim);
    VSD         = new Term (od->nSupport, 1, od->nDim);
    VSD_t       = new Term (1, od->nSupport, od->nDim);
    eR          = new eterm(od->nDim);

    Phi 	= 0.0;
    logPDD 	= 0.0;
    VSD		->init(0.0);
    GammaSS	->init(0.0);

    // Step 1: Combine Alpha -> VSD, Beta -> GammaSS, Delta -> R
    SFOR(i, nThread)
    {
        Term::add(GammaSS, markov_cluster[i]->Beta, GammaSS);
        Term::add(VSD, markov_cluster[i]->Alpha, VSD);
        Phi 	+= markov_cluster[i]->Phi;
        logPDD  += markov_cluster[i]->logPDD;
        R	-> add(markov_cluster[i]->Delta, R);
    }

    // Memory control
    SFOR(i, nThread) delete markov_cluster[i];

    // Step 2: Initialize GammaSS = KSS
    Term::add(GammaSS, KSS, GammaSS);

    // Step 3: GammaSS -> GammaSS_inv , VSD -> VSDt
    GammaSS->invert(false, GammaSS_inv);
    VSD->transpose(VSD_t);

    // Step 4: dR = dR + trace(GammaSS_inv * dGammaSS)
    Term::tracemat(GammaSS_inv->me, GammaSS, eR);
    R->add(eR, R);
    eR->reset();

    // Step 5: dR = dR - trace(KSS_inv * dKSS)
    Term::tracemat(KSS_inv->me, KSS, eR);
    R->subright(eR, R);
    eR->reset();

    // Step 6: dR = dR - d(VSDt * GammaSS_inv * VSD)
    Term::trace3(VSD_t, GammaSS_inv, VSD, eR);
    R->subright(eR, R);

    // Step 7: R was included in above computations (incorrect) -> recompute R.
    log_det(R1, sign1, GammaSS->me);
    log_det(R2, sign2, KSS->me);

    // Step 8: Recompute R
    R->me = Phi - logPDD - eR->me + R1 - R2 + od->nTrain * log (2 * PI);
    R->mul(-0.5, R);

    // Step 9: Compute qfs
    mat qfs      = od->mean() + KSS->me * GammaSS_inv->me * VSD->me;

    HyperParams *hyp = new HyperParams();
    od->hyper->clone(hyp); //hyp->setsignal(log(1.5));
    hp.push_back(hyp);
    mu.push_back(qfs);
    score.push_back(R->me);

    // Memory control
    delete eR;
    delete GammaSS;
    delete GammaSS_inv;
    delete VSD;
    delete VSD_t;
}

void Factory::load_result(char* filename)
{
	field < field <mat> > res;
	res.load(filename, arma_binary);

	int n = res.n_rows;
	this->hp.clear(); this->mu.clear(); this->score.clear();

	SFOR(i, n)
	{
		HyperParams* hyper = new HyperParams();
		hyper->nDim = res[i](0, 0).n_rows - 2;

		hyper->setnoise(log(res[i](0, 0)(1, 0)));
		hyper->setsignal(log(res[i](0, 0)(0, 0)));

		SFOR(j, hyper->nDim) hyper->setls(j, log(res[i](0, 0)(j + 2, 0)));
		this->hp.push_back(hyper);

		this->mu.push_back(res[i](1, 0));
		this->score.push_back(res[i](2, 0)(0, 0));
	}
}

void Factory::save_result(char* filename)
{
	int n = (int) this->hp.size();
	field < field <mat> > res(n, 1);

	SFOR(i, n)
	{
		res[i] = field <mat> (3, 1);

		res[i](0, 0) = mat(this->hp[i]->nDim + 2, 1);
		res[i](0, 0)(0, 0) = this->hp[i]->signal();
		res[i](0, 0)(1, 0) = this->hp[i]->noise();

		for (int j = 2; j < this->hp[i]->nDim + 2; j++)
			res[i](0, 0)(j, 0) = this->hp[i]->ls(j - 2);

		res[i](1, 0) = this->mu[i];
		res[i](2, 0) = mat(1, 1); res[i](2, 0)(0, 0) = this->score[i];
	}

	res.save(filename, arma_binary);
}

