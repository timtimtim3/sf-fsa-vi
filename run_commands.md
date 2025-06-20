# All Experiment Run Commands

Each experiment configuration below was run with 5 random seeds (except 'FlatQ - Original Office 6 FSA Long', which was run using the first 3 seeds only):

- `+seed=1001 +batch_run_name="run1"`
- `+seed=1002 +batch_run_name="run2"`
- `+seed=1003 +batch_run_name="run3"`
- `+seed=1004 +batch_run_name="run4"`
- `+seed=1005 +batch_run_name="run5"`

The examples below show the command for the first seed. To reproduce other runs, simply substitute the appropriate seed and batch_run_name.

---

## SF-OLS

### Original Office
```bash
srun python -u run_sfols.py algorithm=sfols-dqn +fsa_name="['task1','task2','task3']" env=officeAreasFeatures +value_iter_type="SFFSAValueIterationAugmented" algorithm.gamma=0.95 +plot=True env.level_name="original_office_areas_rbf" ols.epsilon=0.2 env.add_empty_to_start=True +env.terminate_action=True +env.term_only_on_term_action=True env.env_name="OfficeAreasFeaturesContinuous-v0" +learn_all_extremum=True +fsa_symbols_from_env=True gpi=dqn gpi.learn.total_timesteps=150000 algorithm.epsilon_decay_steps=75000 +use_batch_dir=True +batch_dir_postfix="Original-TermOnlyOnTerm-150k" +seed=1001 +batch_run_name="run1"
```

### Original Office 6 FSA
```bash
srun python -u run_sfols.py algorithm=sfols-dqn +fsa_name="['task1','task2','task3','task4','task5','task6']" env=officeAreasFeatures +value_iter_type="SFFSAValueIterationAugmented" algorithm.gamma=0.95 +plot=True env.level_name="original_office_areas_rbf" ols.epsilon=0.2 env.add_empty_to_start=True +env.terminate_action=True +env.term_only_on_term_action=True env.env_name="OfficeAreasFeaturesContinuous-v0" +learn_all_extremum=True +fsa_symbols_from_env=True gpi=dqn gpi.learn.total_timesteps=150000 algorithm.epsilon_decay_steps=75000 +use_batch_dir=True +batch_dir_postfix="Original-6FSA-TermOnlyOnTerm-150k" +seed=1001 +batch_run_name="run1"
```

### Office Areas
```bash
srun python -u run_sfols.py algorithm=sfols-dqn +fsa_name="['task1','task2','task3']" env=officeAreasFeatures +value_iter_type="SFFSAValueIterationAugmented" algorithm.gamma=0.95 +plot=True env.level_name="office_areas_rbf_goals_apart" ols.epsilon=0.2 env.add_empty_to_start=True +env.terminate_action=True +env.term_only_on_term_action=True env.env_name="OfficeAreasFeaturesContinuous-v0" +learn_all_extremum=True +fsa_symbols_from_env=True gpi=dqn gpi.learn.total_timesteps=150000 algorithm.epsilon_decay_steps=75000 +use_batch_dir=True +batch_dir_postfix="Areas-TermOnlyOnTerm-150k" +seed=1001 +batch_run_name="run1"
```

### Teleport Office
```bash
srun python -u run_sfols.py algorithm=sfols-dqn +fsa_name="['teleport-task2','teleport-task3']" env=officeAreasFeatures +value_iter_type="SFFSAValueIterationAugmented" algorithm.gamma=0.95 +plot=True env.level_name="office_areas_rbf_teleport_big" ols.epsilon=0.02 env.add_empty_to_start=True +env.terminate_action=True +env.term_only_on_term_action=True env.env_name="OfficeAreasFeaturesContinuous-v0" +learn_all_extremum=True gpi=dqn gpi.learn.total_timesteps=200000 algorithm.epsilon_decay_steps=100000 +use_batch_dir=True +batch_dir_postfix="TeleportBig-TermOnlyOnTerm-200k" +seed=1001 +batch_run_name="run1"
```

## LOF

### Original Office
```bash
srun python -u run_lof.py algorithm=lof-dqn +fsa_name="['task1','task2','task3']" env=officeAreasFeatures env.level_name="original_office" env.add_empty_to_start=True +env.terminate_action=False env.env_name="OfficeAreasFeaturesContinuous-v0" algorithm.learning_steps=175000 algorithm.epsilon_decay_steps=87500 +fsa_symbols_from_env=True +use_batch_dir=True +batch_dir_postfix="Original-175k-87k" +seed=1001 +batch_run_name="run1"
```

### Original Office 6 FSA
```bash
srun python -u run_lof.py algorithm=lof-dqn +fsa_name="['task1','task2','task3','task4','task5','task6']" env=officeAreasFeatures env.level_name="original_office" env.add_empty_to_start=True +env.terminate_action=False env.env_name="OfficeAreasFeaturesContinuous-v0" algorithm.learning_steps=175000 algorithm.epsilon_decay_steps=87500 +fsa_symbols_from_env=True +use_batch_dir=True +batch_dir_postfix="Original-6FSA-175k-87k" +seed=1001 +batch_run_name="run1"
```

### Office Areas
```bash
srun python -u run_lof.py algorithm=lof-dqn +fsa_name="['task1','task2','task3']" env=officeAreasFeatures env.level_name="office_areas_rbf_goals_apart" env.add_empty_to_start=True +env.terminate_action=False env.env_name="OfficeAreasFeaturesContinuous-v0" algorithm.learning_steps=300000 algorithm.epsilon_decay_steps=150000 +fsa_symbols_from_env=True +use_batch_dir=True +batch_dir_postfix="Areas-300k-150k" +seed=1001 +batch_run_name="run1"
```

### Teleport Office
```bash
srun python -u run_lof.py algorithm=lof-dqn +fsa_name="['teleport-task2','teleport-task3']" env=officeAreasFeatures env.level_name="office_areas_rbf_teleport_big" env.add_empty_to_start=True +env.terminate_action=False env.env_name="OfficeAreasFeaturesContinuous-v0" algorithm.learning_steps=300000 algorithm.epsilon_decay_steps=150000 +use_batch_dir=True +batch_dir_postfix="TeleportBig-300k-150k" +seed=1001 +batch_run_name="run1"
```

## FlatQ

### Original Office
```bash
srun python -u run_flatq.py algorithm=flat-dqn +fsa_name="['task1','task2','task3']" env=officeAreasFeatures +plot=True env.level_name="original_office_areas_rbf" env.add_empty_to_start=True +env.terminate_action=False env.env_name="OfficeAreasFeaturesContinuous-v0" +fsa_symbols_from_env=True algorithm.total_timesteps=350000 algorithm.epsilon_decay_steps=175000 algorithm.net_arch='[256,256]' +use_batch_dir=True +batch_dir_postfix="Original-350k" +seed=1001 +batch_run_name="run1"
```

### Original Office 6 FSA
```bash
srun python -u run_flatq.py algorithm=flat-dqn +fsa_name="['task1','task2','task3','task4','task5','task6']" env=officeAreasFeatures +plot=True env.level_name="original_office_areas_rbf" env.add_empty_to_start=True +env.terminate_action=False env.env_name="OfficeAreasFeaturesContinuous-v0" +fsa_symbols_from_env=True algorithm.total_timesteps=175000 algorithm.epsilon_decay_steps=85000 algorithm.net_arch='[256,256]' +use_batch_dir=True +batch_dir_postfix="Original-6FSA-175k" +seed=1001 +batch_run_name="run1"
```

### Original Office 6 FSA Long
```bash
srun python -u run_flatq.py algorithm=flat-dqn +fsa_name="['task1','task2','task3','task4','task5','task6']" env=officeAreasFeatures +plot=True env.level_name="original_office_areas_rbf" env.add_empty_to_start=True +env.terminate_action=False env.env_name="OfficeAreasFeaturesContinuous-v0" +fsa_symbols_from_env=True algorithm.total_timesteps=350000 algorithm.epsilon_decay_steps=175000 algorithm.net_arch='[256,256]' +use_batch_dir=True +batch_dir_postfix="Original-6FSA-350k" +seed=1001 +batch_run_name="run1"
```

### Office Areas
```bash
srun python -u run_flatq.py algorithm=flat-dqn +fsa_name="['task1','task2','task3']" env=officeAreasFeatures +plot=True env.level_name="office_areas_rbf_goals_apart" env.add_empty_to_start=True +env.terminate_action=False env.env_name="OfficeAreasFeaturesContinuous-v0" +fsa_symbols_from_env=True algorithm.total_timesteps=300000 algorithm.epsilon_decay_steps=150000 algorithm.net_arch='[256,256]' +use_batch_dir=True +batch_dir_postfix="Areas-300k" +seed=1001 +batch_run_name="run1"
```

### Teleport Office
```bash
srun python -u run_flatq.py algorithm=flat-dqn +fsa_name="['teleport-task2','teleport-task3']" env=officeAreasFeatures +plot=True env.level_name="office_areas_rbf_teleport_big" env.add_empty_to_start=True +env.terminate_action=False env.env_name="OfficeAreasFeaturesContinuous-v0" algorithm.total_timesteps=300000 algorithm.epsilon_decay_steps=150000 algorithm.net_arch='[256,256]' +use_batch_dir=True +batch_dir_postfix="TeleportBig-300k" +seed=1001 +batch_run_name="run1"
```
