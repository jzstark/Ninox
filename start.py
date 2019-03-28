import os
import exp
import database as db

default_dir = '~/Tmp/Ninox'

# Simulation entrypoint. Need to specify what to do with
def main(root_path):
    root_path = os.path.abspath(root_path)
    db.init_db(root_dir)
    exp.exp1(os.path.join(root_path, db.dir, "exp1"))

if __name__ == "__main__":
    #main(sys.argv[0])
    main(default_dir)
