# The 100-task result
| Metric | BASE | REASON |
|---|---|---|
| Accuracy | 17% | 74% |
| Mean output tokens | 45.2 | 281.7 |
| Median output tokens | 30.5 | 270 |
| Mean latency | 1.20 s | 7.41 s |
| Median latency | 0.81 s | 7.16 s |
| Mean GPU energy | 72.7 J | 458.4 J |
| Median GPU energy	| 48.6 J | 437.6 J |

`REASON` therefore costs roughly:

$$ \boxed{6.2\times\text{ latency}} $$

and

$$ \boxed{6.3\times\text{ GPU energy}} $$

compared with `BASE`. Therefore, we can say that there is a significant difference in computation cost of the two types, and hence, there is a need to manage, at run time, the choice of action.

## What happens to quality?

Across the 100 questions:

| BASE | REASON | Tasks | Interpretation |
|---|---|---|---|
| Wrong	| Correct	| 59	| REASON helps |
| Wrong |	Wrong	| 24	| REASON wastes compute |
| Correct	| Correct	| 15	| REASON unnecessary |
| Correct	| Wrong	| 2	| REASON actively hurts |

Therefore:

$$ \Delta Q_{\text{REASON}}>0:\quad \boxed{59\%} $$ $$ \Delta Q_{\text{REASON}}=0:\quad \boxed{39\%} $$ $$ \Delta Q_{\text{REASON}}<0:\quad \boxed{2\%} $$

with

$$ E[\Delta Q_{\text{REASON}}]=\boxed{+0.57}. $$

### Inference 
Although `REASON` is valuable on most questions, blindly applying it to every question wastes roughly 6× the resources on 39% of samples without improving quality, and actually makes 2% worse.

The controller's problem is therefore real:

$$ \boxed{ \text{Can we identify the 59\% where REASON is worth its cost?} } $$

rather than spending ~458 J of additional GPU inference indiscriminately.