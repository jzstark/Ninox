import os
from exp import *
import database as db
import utils

#import cProfile


def main(db_dir):
    db_dir = utils.full_path(db_dir)
    db.init_db(db_dir)

    def run_exp(exp_fun, save_dir):
        exp_fun(os.path.join(db_dir, save_dir))

    # Execute experiments

    #run_exp(exp_step, "exp_step")
    #run_exp(exp_samplesize, "exp_samplesize")
    #run_exp(exp_straggle_perc, "exp_straggle_perc")
    #run_exp(exp_straggleness, "exp_straggleness")
    #run_exp(exp_straggle_accuracy, "exp_straggle_accuracy")
    #run_exp(exp_frontier, "exp_frontier")
    #run_exp(exp_ratio, "exp_ratio")
    #run_exp(exp_regression, "exp_regression")
    run_exp(exp_scalability, "exp_scalability")
    #run_exp(exp_scalability_step, "exp_scalability_step")

main(db.default_dir)
#cProfile.run('main(db.default_dir)')
