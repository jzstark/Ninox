# Send request to database

"""
Experiment 1: Distribution of final iteration progress.
"""

# - Table 1, across barriers, final line of all nodes. PDF/CDF vs process
# - Adjust straggler percentage. Redo evalulation. Process mean/std line vs percentage.
# - Adjust straggler scale. Redo evaluation. Process mean/std line vs percentage.

"""
Experiment 2: "Accuracy"
"""

# - Sequence length vs (accuracy compared to BSP); which node *generates* a new update. I expect pBSP and pSSP are bounded, but not ASP. The definition of "difference" should follow that in math proof.

# - Change straggler percentage of pBSP, pSSP. Redo Evaluation,

# - Change straggler scale. Redo.

"""
Experiment 3: Comparison of time used on running/waiting/transmission.
"""

# - Bar chart. Take only final status (or whole process if you want, but the point is not very clear). Compare barriers.


"""
Experiment 4: Quality of updates.
"""

# - Number of updates vs real time. Expect ASP to generate most.
# - Sequence length vs real time for each barrier -->  Number of updates vs. Accuracy.  Expect ASP to be lowest.

"""
Experiment 5: Scalability (???)
"""

# - Similar to Exp 1. Fix straggler percentage, compare iteretion progress.
# - Test the effect of \beta
