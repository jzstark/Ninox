# Ninox
Simulator of Actor PSP evaluation


## Scheme of result tables

For Each (barrier, network size) combination, there are these tables...

Initially:
- iteration vs. nodes. Running time
- iteration vs. nodes. Wait time
- iteration vs. nodes. Transmission time.

What can be further derived from the previous tables:

- Real Time vs. nodes. iteration number. (iteration -> real time translation)
  + How to get: "iter i" means the i-th update of a node is already produced and a new update is not produced yet.
- Update sequence `x`. One big array for all nodes altogether, per barrier. Also for each element in the sequence, the real time it is generated is calculated.

## Assumption

- No node drop out or added in.
- There is one, and only one, central server. It never dies.
- The only transmission in the network is sending update to central server.



## Naming Scheme

The data files: `root_path`/`exp_path`/(`config_prefix` + `barrier_name` + `observation`.csv)


## Other

Remember to disable/enable the `run(config)` when you want to start the simulation again or just want to use existing data.
May need to automate later if necessary.
