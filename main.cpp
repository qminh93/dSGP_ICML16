#include "libhead.h"
#include "organized.h"
#include "factory.h"
#include "trainer.h"
#include "hyper.h"

void deploy_training(char* config)
{
	Configuration* setting = new Configuration();
	setting->load(config);

	if (setting->bad_cfg)
	{
	   cout << "BAD COMMAND" << endl;
	   return;
	}

	SEED(SEED_DEFAULT);

	RawData *raw;

	if (setting->nPoint) raw = new RawData(setting->nPoint);
	else raw = new RawData();

	raw->load(setting->data_file.c_str());
	raw->normalize_x();

	SFOR (i, setting->nExp)
	{
		ExpSetting* exs = setting->exp_list[i];
		SEED(exs->seed);

		OrganizedData* od = new OrganizedData(setting);

		od->process(raw, exs->nBlock, exs->pTest, exs->sp_per_blk, exs->max_sp);

		pdd y_scale(0, 1);

		omp_set_num_threads(setting->nThread);

		Trainer *model = new Trainer(od, setting->nThread , setting->training_num_ite, exs);
		model->sgd_train(y_scale);

		delete od;
		delete model;
	}

	delete raw;
}

int main(int argc, char* argv[])
{
    deploy_training("./aimpeak_config.txt");
    return 0;
    if (argc < 1) {
    	cout << "Configuration file expected" << endl;
    	cout << "Usage: <Executable> <Path to config file>" << endl;
    }
    else {
    	deploy_training(argv[1]);
    }

    return 0;
}

