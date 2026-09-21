Empirical Investigation in Overestimation in DQN (Deep Q Networks)

For all results/training regime/ conceptual discussion please consult Results_Report.pdf

###  Abstract ###
DQN and related algorithms have a well known problem of overestimation. This paper leverages a theoretical
upper bound of the Cartpole environment as a proxy measure for overestimation. This paper begins by
probing at DQN/DDQN overestimation di erences ,but found that target network update frequency (K) has
a much large impact on these violations as well as more general impact on learned Q value scale. This relation
between update frequency, Q scale and learnt policy are explored in greater depth; nding that the later two
appear to be strongly decoupled. The paper to establishes whether this learned Q value scale has any e ect
on policy. It was found that networks that exceedingly violate this upper bound or having inappropriate
Q value scales can still achieve near-optimal policy whilst also e ectively mapping Q values to normalised
episodic discounted returns. We then o er a candidate explanation as to why, in these cases, training is still
possible and why this phenomena maybe speci c to survival like environments such as Cartpole; where the
absence of a next frame is often a stronger learning signal than the reward for reaching the next.
For Results/Findings/Theoretical basis please consult Results_Report.pdf 


### Code ###
For Use of Code - 
- `Train.py` — DQN training and agent generation
- `Policy_Evaluation.py` — Evaluation of trained policies
- `Q_value_analysis.py` — Q-value and overestimation analysis
- `Plotting_Basic.py` — Generation of (some)figures
- `Results/` — Experimental results, CSVs and figures
- `Results_Report.pdf` — Full methodology, Results and Discussion

## Trained Agents
The trained agents used in the analysis and state genertaion/ranking are available under **Releases**


Install the required Python packages with:
```bash
pip install -r requirements.txt

