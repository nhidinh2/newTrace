# Story clustering: priberam-2018 (eng)

8726 test documents in 222 human-labelled stories from 1935 sources, replayed in publication order; embedder `sentence-transformers/all-MiniLM-L6-v2`.

| System | threshold | pairwise F1 | P | R | B-cubed F1 | stories predicted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NewsTrace, shipped | 0.880 | 10.2 | 99.7 | 5.3 | 33.4 | 3215 |
| NewsTrace, dev-tuned | 0.550 | 82.8 | 97.7 | 71.9 | 86.1 | 748 |
| CluStream (Aggarwal & Yu 2006) | - | 79.0 | 98.6 | 65.9 | - | - |
| Miranda et al.: tokens+lemmas+entities | - | 92.7 | 92.9 | 92.5 | - | - |
| Miranda et al.: + timestamps | - | 94.1 | 98.2 | 90.3 | - | - |

Dev sweep (threshold selected on pairwise F1):

| threshold | pairwise F1 | P | R | stories predicted |
| ---: | ---: | ---: | ---: | ---: |
| 0.500 | 85.2 | 83.6 | 86.9 | 1359 |
| 0.525 | 85.1 | 83.9 | 86.3 | 1441 |
| 0.550 | 86.6 | 89.5 | 84.0 | 1511 |
| 0.575 | 86.4 | 91.7 | 81.7 | 1588 |
| 0.600 | 86.0 | 92.6 | 80.3 | 1696 |
| 0.625 | 85.5 | 93.6 | 78.6 | 1811 |
| 0.650 | 84.4 | 97.1 | 74.6 | 1923 |
| 0.675 | 82.2 | 97.1 | 71.3 | 2069 |
| 0.700 | 80.3 | 97.3 | 68.3 | 2229 |
| 0.725 | 76.3 | 97.8 | 62.5 | 2398 |
| 0.750 | 65.2 | 97.5 | 49.0 | 2619 |
| 0.775 | 59.1 | 97.1 | 42.5 | 2898 |
| 0.800 | 52.1 | 97.6 | 35.5 | 3233 |
| 0.825 | 45.0 | 98.3 | 29.2 | 3624 |
| 0.850 | 37.7 | 98.0 | 23.4 | 4081 |
| 0.875 | 30.0 | 98.8 | 17.7 | 4627 |
| 0.900 | 23.9 | 98.6 | 13.6 | 5258 |
| 0.925 | 17.5 | 98.9 | 9.6 | 5909 |
| 0.950 | 12.9 | 98.9 | 6.9 | 6623 |

- Pairwise P/R/F1 over all test documents of the language, as in Miranda et al. (2018), so the published rows are directly comparable.
- The published systems are supervised (an SVM ranker trained on the train split); NewsTrace fits one threshold on dev and nothing else.
- The embedder sees the headline and the start of the body only: all-MiniLM-L6-v2 truncates input at 256 word pieces.
- No confidence intervals: resampling stories does not give a clean bootstrap for pairwise metrics, and the paper reports none to compare with.
