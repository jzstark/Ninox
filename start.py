import os
from exp import *
import database as db
import utils

def main(db_dir):
    db_dir = utils.full_path(db_dir)
    db.init_db(db_dir)

    def run_exp(exp_fun, save_dir):
        exp_fun(os.path.join(db_dir, "exp_straggle_perc"))

    # Execute experiments

    #run_exp(exp_step, "exp_step")
    #run_exp(exp_samplesize, "exp_samplesize")
    #run_exp(exp_straggle_perc, "exp_straggle_perc")
    run_exp(exp_accuracy, "exp_accuracy")

main(db.default_dir)
