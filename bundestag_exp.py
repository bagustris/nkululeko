# main.py
# Demonstration code to use the ML-experiment framework

import sys
sys.path.append("/home/fburkhardt/ResearchProjects/nkululeko/src")

import experiment as exp
import dataset as ds
import configparser
from emodb import Emodb
import glob_conf
from util import Util

def main(config_file):
    # for two sexes
    sexes = ['female', 'male']
    util = Util()

    for s in sexes:
        # load one configuration per experiment
        config = configparser.ConfigParser()
        config.read(config_file)
        # set the sex
        config['DATA']['sex'] = s 
        # add the sex to the experiment name
        config['EXP']['name'] = config['EXP']['name']+'_'+s

        # create a new experiment
        expr = exp.Experiment(config)
        util.debug(f'running {expr.name}')

        # load the data
        expr.load_datasets()

        # split into train and test
        expr.fill_train_and_tests()
        util.debug(f'train shape : {expr.df_train.shape}, test shape:{expr.df_test.shape}')

        # extract features
        expr.extract_feats()
        util.debug(f'train feats shape : {expr.feats_train.df.shape}, test feats shape:{expr.feats_test.df.shape}')

        # initialize a run manager
        expr.init_runmanager()

        # run the experiment
        expr.run()


if __name__ == "__main__":
    main('/home/fburkhardt/ResearchProjects/nkululeko/exp_bundestag.ini')# sys.argv[1])