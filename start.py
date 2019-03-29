import os
import exp
import database as db

# Simulation entrypoint. Need to specify what to do with
def main(db_path):
    db_path = os.path.expanduser(db_path)
    db_path = os.path.abspath(db_path)
    db.init_db(db_path)
    exp.exp_step(os.path.join(db_path, "exp_step"))

#if __name__ == "__main__":
    #main(sys.argv[0])
main(db.default_dir)
