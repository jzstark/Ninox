import os

"""
Definition of config:
- whole net: size (#s), stop_time (s), straggler_perc (percentage), straggleness (scale)
- node: exec_time (s), trans_time (s), exec_randomness(), trans_randomness
(no need to specify the barrier parameters. The all need to be there at each configuration)

Basic config: e.g. exec_time = 5, trans_time = 1
We can only tune the straggler and randomness parameters, not these basic speed.
"""

def config_to_string(config):
    file_prefix = ('tbl_%d_st%.1f_stp%d_') % \
        (config['size'], config['straggleness'], config['straggler_perc'])
    return os.path.join(config['path'], file_prefix)


def dbfilename(config, barrier_name, ob_type):
    return config_to_string(config) + barrier_name + '_' + ob_type + '.csv'


def full_path(path):
    path = os.path.expanduser(path)
    path = os.path.abspath(path)
    return path

def to_upper(s):
    if s == 'pbsp' : return 'pBSP'
    if s == 'pssp' : return 'pSSP'
    return str.upper(s)

def barrier_to_label(name):
    name = name.split("_")
    assert(len(name) <= 3)
    if len(name) == 1:
        return to_upper(name[0])
    if len(name) == 2:
        label = '$\\beta$' if name[1][0] == 'p' else name[1][0]
        param = label + '=' + name[1][1:]
        return "%s(%s)" % (to_upper(name[0]), param)
    if len(name) == 3:
        label = '$\\beta$' if name[2][0] == 'p' else name[2][0]
        param1 = name[1][0] + '=' + name[1][1:]
        param2 = label + '=' + name[2][1:]
        return r'%s(%s,%s)' % (to_upper(name[0]), param1, param2)
