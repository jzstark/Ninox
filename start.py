import os
import exp
import database as db

# Simulation entrypoint. Need to specify what to do with
def main(db_path):
    db_path = os.path.expanduser(db_path)
    db_path = os.path.abspath(db_path)
    db.init_db(db_path)

    # Execute experiments

    # exp.exp_step(os.path.join(db_path, "exp_step"))
    exp.exp_samplesize(os.path.join(db_path, "exp_samplesize"))
    # exp.exp_straggle_perc(os.path.join(db_path, "exp_straggle_perc"))


main(db.default_dir)
