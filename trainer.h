#ifndef TRAINER_H_
#define TRAINER_H_

#include "libhead.h"
#include "factory.h"
#include "term.h"
#include "organized.h"
#include "kernel.h"

#define DECAY_PARAM 0.75

class Trainer {
	public:
		OrganizedData *od;
		Kernel *ker;
		Factory *fact;
		ExpSetting* exs;
		pdd trans;
		int nThread, nBand, nIte, pred_mode;
        bool dtc;

		Trainer(OrganizedData *od, int nThread, int nIte, ExpSetting* exs);
	   ~Trainer();
		void sgd_train(pdd trans);
		eterm* compute_log_grad(eterm* R);
		double compute_norm_grad(eterm* R, int turn);
		double compute_rate(double r, double &tau, double decay, double i);
		void   compute_RMSE(vd &RMSE, int i);
		void   update_params_sgd(double rate, eterm* R, int turn);
	private:
	protected:
};

#endif /* TRAINER_H_ */

