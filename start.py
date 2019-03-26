# To start a simulation with given configuration

import simulator
import exp
import database as db


"""
Definition of config:
- whole net: size (#s), stop_time (s), straggler_perc (percentage), straggleness (scale)
- node: exec_time (s), trans_time (s), exec_randomness(), trans_randomness
(no need to specify the barrier parameters. The all need to be there at each configuration)

Basic config: e.g. exec_time = 5, trans_time = 1
We can only tune the straggler and randomness parameters, not these basic speed.
"""


configs = [
    {size=100 , straggler_perc=1, straggleness=0.5}
    {size=1000, straggler_perc=1, straggleness=0.5}
]

def config_to_string(config):
    return "tb_pssp_s10_p1"


def main():
    database.init_db()

    map(simulator.run, configs)

    exp.exp1(db.dbconfig)
    exp.exp2(db.dbconfig)

if __name__ == "__main__":
    main()
