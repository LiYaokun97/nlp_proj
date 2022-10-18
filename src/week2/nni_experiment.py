from nni.experiment import Experiment
experiment_type = 'hpo'
experiment = Experiment('local')
experiment.config.trial_command = 'python3 BinaryClassification.py'
#experiment = Experiment('remote')
        #experiment.config.trial_command = 'python3 trial.py'
        #experiment.config.machines.append(RemoteMachineConfig(ip=10.113.160.1, user_name=wangqiongyan))
        #experiment.run(8080)
experiment.config.trial_code_directory = '.'
search_space = {'lr': {'_type': 'uniform', '_value': [0.0001, 0.01]},
                'batch_size': {'_type': 'choice', '_value': [32,64,128]}}
experiment.config.search_space = search_space
experiment.config.tuner.name = 'TPE'  # 试hyperband（多步之后保留Top k的结果 淘汰收敛较慢的结果）
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
experiment.config.max_trial_number = 15
experiment.config.trial_concurrency = 1
experiment.config.training_service.use_active_gpu = True
experiment.config.trial_gpu_number = 1
experiment.run(8008)