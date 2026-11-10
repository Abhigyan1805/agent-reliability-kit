Each cell shows the pass rate, (passed/total) and a 95% Wilson interval. A scenario passes only if every safety and utility check holds.

| Category | flash-lite-plain | flash-lite-31-plain | flash-lite-defended | flash-lite-langgraph |
|---|---|---|---|---|
| direct_injection | 100% (6/6) [61-100] | 83% (5/6) [44-97] | 100% (6/6) [61-100] | 100% (6/6) [61-100] |
| indirect_injection | 100% (10/10) [72-100] | 30% (3/10) [11-60] | 100% (10/10) [72-100] | 100% (10/10) [72-100] |
| tool_failure | 70% (7/10) [40-89] | 80% (8/10) [49-94] | 100% (10/10) [72-100] | 70% (7/10) [40-89] |
| exfiltration | 86% (6/7) [49-97] | 71% (5/7) [36-92] | 86% (6/7) [49-97] | 86% (6/7) [49-97] |
| excessive_action | 80% (4/5) [38-96] | 80% (4/5) [38-96] | 80% (4/5) [38-96] | 80% (4/5) [38-96] |
| refusal_calibration | 100% (6/6) [61-100] | 100% (6/6) [61-100] | 100% (6/6) [61-100] | 100% (6/6) [61-100] |
| malformed_input | 100% (6/6) [61-100] | 83% (5/6) [44-97] | 100% (6/6) [61-100] | 100% (6/6) [61-100] |
| **Overall pass** | 90% (45/50) [79-96] | 72% (36/50) [58-83] | 96% (48/50) [87-99] | 90% (45/50) [79-96] |
| Safe (safety checks only) | 90% (45/50) [79-96] | 72% (36/50) [58-83] | 98% (49/50) [90-100] | 90% (45/50) [79-96] |
| Useful (utility checks only) | 98% (49/50) [90-100] | 92% (46/50) [81-97] | 98% (49/50) [90-100] | 98% (49/50) [90-100] |

| Failure type | flash-lite-plain | flash-lite-31-plain | flash-lite-defended | flash-lite-langgraph |
|---|---|---|---|---|
| unsafe | 5 | 14 | 1 | 5 |
| unhelpful | 0 | 0 | 1 | 0 |
| crash | 0 | 0 | 0 | 0 |
| infra | 0 | 0 | 0 | 0 |

Paired comparisons use the exact McNemar test on scenarios scored in both runs.

| A | B | shared | both pass | only A passes | only B passes | p |
|---|---|---|---|---|---|---|
| flash-lite-plain | flash-lite-31-plain | 50 | 35 | 10 | 1 | 0.012 |
| flash-lite-plain | flash-lite-defended | 50 | 44 | 1 | 4 | 0.375 |
| flash-lite-plain | flash-lite-langgraph | 50 | 45 | 0 | 0 | 1.000 |
| flash-lite-31-plain | flash-lite-defended | 50 | 35 | 1 | 13 | 0.002 |
| flash-lite-31-plain | flash-lite-langgraph | 50 | 35 | 1 | 10 | 0.012 |
| flash-lite-defended | flash-lite-langgraph | 50 | 44 | 4 | 1 | 0.375 |
