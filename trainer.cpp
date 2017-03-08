    /*
 * trainer.cpp
 *
 *  Created on: 18 Aug 2015
 *      Author: nghiaht
 */

#include "trainer.h"
#define DTC  0
#define PIC  1
#define PITC 2

Trainer::Trainer(OrganizedData *od, int nThread, int nIte, ExpSetting* exs)
{
	this->od	 	  = od;
	this->nThread 	  = nThread;
	this->nBand	  	  = exs->nBand;
	this->nIte	  	  = nIte;
	this->pred_mode   = exs->predict_mode;
	this->dtc         = !(pred_mode);
	this->exs	  	  = exs;
	this->ker	  	  = new Kernel(od->hyper);
	this->fact 	      = new Factory(od, ker, this->nThread, this->nBand, this->dtc);
}

Trainer::~Trainer()
{

}

eterm* Trainer::compute_log_grad(eterm *R)
{
	eterm* lR = new eterm(od->nDim);

	lR->me = R->me;
	lR->dnoise  = od->noise() * R->dnoise;
	lR->dsignal = od->signal() * R->dsignal;
	SFOR(i, od->nDim)
	lR->dls[i]  = od->ls(i) * R->dls[i];

	return lR;
}

double Trainer::compute_norm_grad(eterm *R, int turn)
{
	double norm_dR  = 0;

	if (od->setting->training_mode == 3)
    {
        if (turn == 0) norm_dR = pow(R->dnoise, 2.0);
        if (turn == 1) norm_dR = pow(R->dsignal, 2.0);
        if (turn == 2) SFOR(v, od->nDim) norm_dR += pow(R->dls[v],  2.0);
        return sqrt(norm_dR);
    }

	// Normalizing gradient
	SFOR(v, od->nDim) 					norm_dR += pow(R->dls[v],  2.0);
	if (od->setting->training_mode > 0) norm_dR += pow(R->dnoise,  2.0);
	if (od->setting->training_mode > 1) norm_dR += pow(R->dsignal, 2.0);

	return sqrt(norm_dR);
}

void Trainer::update_params_sgd(double rate, eterm* R, int turn)
{
	double norm_dR  = compute_norm_grad(R, turn);

	if (od->setting->training_mode == 3)
    {
        if (turn == 0) od->hyper->setnoise(log(od->hyper->noise()) + rate * R->dnoise / norm_dR);
        if (turn == 1) od->hyper->setsignal(log(od->hyper->signal()) + rate * R->dsignal / norm_dR);
        if (turn == 2) SFOR(v, od->nDim) od->hyper->setls(v, log(od->hyper->ls(v)) + rate * R->dls[v] / norm_dR);
        return;
    }

	// Update parameters
	SFOR(v, od->nDim)                   od->hyper->setls(v, log(od->hyper->ls(v)) + rate * R->dls[v] / norm_dR);
	if (od->setting->training_mode > 0) od->hyper->setnoise(log(od->hyper->noise()) + rate * R->dnoise / norm_dR);
	if (od->setting->training_mode > 1) od->hyper->setsignal(log(od->hyper->signal()) + rate * R->dsignal / norm_dR);
}

double Trainer::compute_rate(double r, double &tau, double decay, double i)
{
	// Possible schedules to consider
	// ------
	// tau += 0.2 * i * i + 1.5 * i + 0.1;
	// tau += 0.05 * pow(i, 2.5 - sqrt(nBand)) + (0.25 - 0.15 * sqrt(nBand)) * i;
	tau += 0.4 * sqrt(i);
	// tau += 0.7 * sqrt(i);
	// tau += 0.05 * nBand * pow(i, 2.5 - sqrt(nBand)) + (0.25 - 0.15 * sqrt(nBand)) * i * nBand;
	// ------

	return r / pow(1 + r * tau * i, decay);
}

void Trainer::compute_RMSE(vd &RMSE, int i)
{
    double err = 0.0;
    if (nBand == 0 && pred_mode != PIC)
        err = fact->DTC_predict(&fact->mu[i], fact->hp[i]);
    else if (od->setting->bcm)
        err = fact->bcm_predict(&fact->mu[i]);
	else if (nBand == 0)
        err = fact->PIC_precompute(fact->hp[i], &fact->mu[i]);
    else
    {
		Waves *tsunami = fact->prepare_waves(fact->hp[i]);
        err = fact->predict(&fact->mu[i], fact->hp[i], tsunami);
        delete tsunami;
	}

	RMSE.push_back(err);
	cout << "RMSE of " << i << "-th set of hyper parameters = " << RMSE[i] << endl;
}

void Trainer::sgd_train(pdd trans)
{
	cout << "Factory starting ..." << endl;
	this->trans = trans;

	time_t t1 = time(NULL);
	cout << "Commencing training ..." << endl;
	cout << "Number of iterations = " << nIte << endl;
	cout << "Number of support = " << this->od->nSupport << endl;

	double  rate;
    double  tau     = od->setting->tau,
            decay   = od->setting->decay;

    vd r(3); r[2] = od->setting->rls; r[1] = od->setting->rs; r[0] = od->setting->rn;

	eterm* R = new eterm(od->nDim);
	vd RMSE, training_time;
	double total_pred_time = 0.0;

	//SEED(SEED_DEFAULT);

	SFOR(i, nIte)
	{
		time_t s1 = time(NULL), s2;

		cout << "Training iteration " << i << endl;
		cout << "Precomputing KSS and KSS_inv" << endl;
		fact->precompute();

		#pragma omp parallel for schedule(static, 1)
		SFOR(t, nThread) fact->process_stream(t);

		cout << "Computing objective function ..." << endl;
		fact->compute_R(R);
		cout << "Objective function value : " << R->me << endl;

		cout << "Updating kernel parameters ..." << endl;

		s2 = time(NULL);
		training_time.push_back((double)(s2 - s1));
		cout << "Iteration " << i << " finished after " << training_time[i] << " sec(s) ..." << endl;

		s1 = time(NULL);
		compute_RMSE(RMSE, i);
		s2 = time(NULL);
		total_pred_time += (double) (s2 - s1);

		cout << "Prediction time for iteration " << i << " takes " << (double) (s2 - s1) << " sec(s) ..." << endl;
		cout << "Updating kernel parameters ..." << endl;
		eterm* lR = compute_log_grad(R);

        int turn, ite;
		if (od->setting->training_mode < 3) { turn = 2; ite = i; }
		else { turn = i % 3; ite = i / 3; }

        rate = compute_rate(r[turn], tau, decay, ite);
        update_params_sgd(rate, lR, turn);
        R->reset(); fact->reset();
	}

	time_t t2 = time(NULL);
	cout << "Training & testing phases finished after " << (double)(t2 - t1) << " sec(s) ..." << endl;

	cout << "Average Prediction Time = " << total_pred_time / nIte << endl;

	cout << "Summary: " << endl;
	SFOR(i, RMSE.size()) cout << RMSE[i] << " " << fact->score[i] << endl;


	FILE* logfile = fopen(exs->res_file.c_str(), "w");
	SFOR(i, RMSE.size())
		fprintf(logfile, "%d %.5f %.5f %.5f\n", i, RMSE[i], fact->score[i], training_time[i]);
	fclose(logfile);

	training_time.clear();
}

