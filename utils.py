import os

"""
Definition of config:
- whole net: size (#s), stop_time (s), straggler_perc (percentage), straggleness (scale)
- node: exec_time (s), trans_time (s), exec_randomness(), trans_randomness
(no need to specify the barrier parameters. The all need to be there at each configuration)

Basic config: e.g. exec_time = 5, trans_time = 1
We can only tune the straggler and randomness parameters, not these basic speed.
"""

def config_to_string(config, barrier_str):
    file = ('tbl_%d_st%d_stp%d_' + barrier_str) % \
        (config['size'], config['straggleness'], config['straggler_perc'])
    return os.path.join(config['path'], file)
