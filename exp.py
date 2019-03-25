# Send request to database, get data, and plot.

"""
Experiment 1: Distribution of final iteration progress.
"""

# - Table 1, across barriers, final status of all nodes. PDF/CDF vs process
# - Adjust straggler percentage. Redo evalulation. Process mean/std line vs percentage.
# - Adjust straggler scale. Redo evaluation. Process mean/std line vs percentage.

def exp_iteration(db):
    pass

"""
Experiment 2: "Accuracy"
"""

# - Sequence length vs (accuracy compared to BSP); which node *generates* a new update. I expect pBSP and pSSP are bounded, but not ASP. The definition of "difference" should follow that in math proof.
# (Note that sequence length itself is "number of updates")
# - Change straggler percentage of pBSP, pSSP. Redo Evaluation,
# - Change straggler scale. Redo.
# - Chnage x-axis to real time for all the above

"""
Experiment 3: Comparison of time used on running/waiting/transmission.
"""

# - Bar chart. Take only final status (or whole process if you want, but the point is not very clear). Compare barriers.

"""
Experiment 4: Scalability
"""

# Leave this to be decided at this stage. Before using notes as x-axis, I need to do some observation of the Accuracy vs. time/iteratio performance under different network size and **sampling size**. Then I can decide what is a good way to present the scability of sampling size.
