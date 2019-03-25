# Runs the whole simulation

# Requirement : I need to add more tables, without running the basic simulation again. The thing is that sampling size and network size might need to be adjusted.

# All the considered barriers: ASP, BSP, pBSP-1/2/3/4.., SSP-2/3/5/10, pSSP-[2/3/5/10]-[1/2/3/4/...]

import barrier
import database as db

class Node:
    def __init__(self, config):
        self.exec_time  = exec_time
        self.randomness = randomness
        self.iteration  = 0
        self.clock = 0.

    def single_iter(barrier):
        true_exec_time = f(exec_time, randomness)

    def get(self):
        return self.iteration


class Link:
    pass


class Network:

    def __init__(self, config):
        self.name = config.name

        self.
        network size
        node: speed,

        db.init()


    def execute(self, stop):
        while(!stop):
            for node in self.nodes:
                t_run   = node.single_iter()

                t_wait_asp =
                t_wait_bsp = barrier.bsp(nodes.get(), [n.get() for n in nodes])
                t_wait_ssp_10 =


                t_comm2 = edge.speed() # might be local or to two-pass
                node.count += 1

                db.write(t_run, db, table1)
                db.write(t_wait, db, table2)
        return


    def postprocess(): pass


def run(stop):
    if config not in config_set:
        network = Network(init_config)
        network.execute(stop)
