# To start a simulation with given configuration

import simulator
import exp
import database

db_config = "mongodb://localhost:27017/"


"""
Definition of config;
"""
def config_to_string(config):
    return "tb_pssp_s10_p1"

configs = [
    {}, #config 1
    {}  #config 2
]

def main():
    database.init_db()

    config = [...]
    map(simulator.execute, config)

    exp.exp1(dbconfig, name=config_to_string(config))
    exp.exp2(dbconfig, name=config_to_string(config))

if __name__ == "__main__":
    main()
