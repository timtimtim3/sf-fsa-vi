# Run commands go here...

## Fast runs
python run_sfols.py algorithm=sfols-qlearning +fsa_name=task1 env=officeAreasRBFOnly +seed=1001 ols=bigger-epsilon
python run_sfols.py algorithm=sfols-qlearning +fsa_name=task1 env=officeAreasRBFOnly +seed=1001 ols=bigger-epsilon algorithm.gamma=0.95

## Big regular runs
python run_sfols.py algorithm=sfols-qlearning +fsa_name=task1 env=officeAreasRBFOnly +seed=1001

## Longer runs

### Eps decay still fast
python run_sfols.py algorithm=sfols-qlearning +fsa_name=task1 env=officeAreasRBFOnly +seed=1001 ols=bigger-epsilon algorithm.gamma=0.95 gpi.learn.total_timesteps=40000

### Eps decays slowly as well
python run_sfols.py algorithm=sfols-qlearning +fsa_name=task1 env=officeAreasRBFOnly +seed=1001 ols=bigger-epsilon algorithm.gamma=0.95 gpi.learn.total_timesteps=40000  algorithm.epsilon_decay_steps=20000


# Play commands...
python play_sfols.py algorithm=sfols-qlearning +fsa_name=task1 env=officeAreasRBFOnly +seed=1001
python play_sfols.py algorithm=sfols-qlearning +fsa_name=task1 env=officeAreasRBFOnly +seed=1001 +dir_postfix="eps0.2-gamma0.95-dist4"

python play_sfols.py algorithm=sfols-qlearning +fsa_name=task1 env=officeAreasRBFOnly +seed=1001 +dir_postfix="eps0.02-gamma0.95-dist4" +value_iter_type="SFFSAValueIterationRBFCentersOnly" +plot=False
python play_sfols.py algorithm=sfols-qlearning +fsa_name=task1 env=officeAreasRBFOnly +seed=1001 +dir_postfix="eps0.02-gamma0.95-dist4" +value_iter_type="SFFSAValueIterationMean" +plot=False
python play_sfols.py algorithm=sfols-qlearning +fsa_name=task1 env=officeAreasRBFOnly +seed=1001 +dir_postfix="eps0.2-gamma0.95-dist4" +value_iter_type="SFFSAValueIterationAugmented" +plot=False

python play_sfols.py algorithm=sfols-qlearning +fsa_name=SemiCircle-task1 env=officeAreasRBFOnly +seed=1001 env.level_name="office_areas_rbf_semi_circle" +dir_postfix="SemiCircle-eps0.2-gamma0.95-dist4" +value_iter_type="SFFSAValueIterationRBFCentersOnly" +plot=False
python play_sfols.py algorithm=sfols-qlearning +fsa_name=SemiCircle-task1 env=officeAreasRBFOnly +seed=1001 env.level_name="office_areas_rbf_semi_circle" +dir_postfix="SemiCircle-eps0.2-gamma0.95-dist4" +value_iter_type="SFFSAValueIterationMean" +plot=False
python play_sfols.py algorithm=sfols-qlearning +fsa_name=SemiCircle-task1 env=officeAreasRBFOnly +seed=1001 env.level_name="office_areas_rbf_semi_circle" +dir_postfix="SemiCircle-eps0.2-gamma0.95-dist4" +value_iter_type="SFFSAValueIterationAugmented" +plot=False


### Semi-Circle RBFs
python run_sfols.py algorithm=sfols-qlearning +fsa_name=SemiCircle-task1 env=officeAreasRBFOnly +seed=1001 ols=bigger-epsilon algorithm.gamma=0.95 env.level_name="office_areas_rbf_semi_circle"
python play_sfols.py algorithm=sfols-qlearning +fsa_name=SemiCircle-task1 env=officeAreasRBFOnly +seed=1001 env.level_name="office_areas_rbf_semi_circle" +dir_postfix="SemiCircle-eps0.2-gamma0.95-dist4"