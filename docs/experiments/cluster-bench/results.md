# Story clustering: priberam-2018 (eng)

8726 test documents in 222 human-labelled stories from 1935 sources, replayed in publication order; embedder `sentence-transformers/all-MiniLM-L6-v2`.

| System | title w | entity w | threshold | pairwise F1 | P | R | B-cubed F1 | stories predicted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NewsTrace, shipped | 0.20 | 0.10 | 0.880 | 10.2 | 99.7 | 5.3 | 33.4 | 3215 |
| NewsTrace, dev-tuned | 0.20 | 0.10 | 0.425 | 92.2 | 97.4 | 87.4 | 91.0 | 502 |
| NewsTrace, dev-learned | 0.43 | 0.52 | 0.425 | 91.3 | 97.0 | 86.2 | 90.7 | 478 |
| CluStream (Aggarwal & Yu 2006) | - | - | - | 79.0 | 98.6 | 65.9 | - | - |
| Miranda et al.: tokens+lemmas+entities | - | - | - | 92.7 | 92.9 | 92.5 | - | - |
| Miranda et al.: + timestamps | - | - | - | 94.1 | 98.2 | 90.3 | - | - |

Dev sweep, shipped weights (threshold selected on pairwise F1):

| threshold | pairwise F1 | P | R | stories predicted |
| ---: | ---: | ---: | ---: | ---: |
| 0.300 | 76.7 | 64.5 | 94.4 | 689 |
| 0.325 | 80.6 | 70.7 | 93.6 | 777 |
| 0.350 | 84.5 | 77.5 | 92.9 | 879 |
| 0.375 | 85.5 | 79.1 | 93.1 | 967 |
| 0.400 | 86.3 | 80.8 | 92.7 | 1060 |
| 0.425 | 86.8 | 82.2 | 92.1 | 1133 |
| 0.450 | 86.5 | 83.5 | 89.8 | 1213 |
| 0.475 | 86.4 | 83.7 | 89.3 | 1280 |
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

Dev sweep, learned weights (threshold selected on pairwise F1):

| threshold | pairwise F1 | P | R | stories predicted |
| ---: | ---: | ---: | ---: | ---: |
| 0.300 | 76.5 | 64.3 | 94.4 | 669 |
| 0.325 | 79.4 | 68.6 | 94.3 | 753 |
| 0.350 | 83.5 | 75.8 | 92.8 | 839 |
| 0.375 | 85.4 | 78.8 | 93.1 | 930 |
| 0.400 | 86.5 | 80.8 | 93.0 | 1010 |
| 0.425 | 87.0 | 82.2 | 92.3 | 1094 |
| 0.450 | 85.5 | 82.1 | 89.0 | 1163 |
| 0.475 | 85.9 | 83.5 | 88.4 | 1227 |
| 0.500 | 83.8 | 83.6 | 83.9 | 1298 |
| 0.525 | 84.4 | 84.2 | 84.7 | 1363 |
| 0.550 | 86.3 | 90.4 | 82.5 | 1430 |
| 0.575 | 82.4 | 91.1 | 75.2 | 1496 |
| 0.600 | 81.3 | 91.2 | 73.3 | 1572 |
| 0.625 | 80.8 | 92.3 | 71.8 | 1665 |
| 0.650 | 79.9 | 96.5 | 68.2 | 1756 |
| 0.675 | 76.4 | 97.0 | 62.9 | 1874 |
| 0.700 | 70.8 | 97.2 | 55.6 | 1996 |
| 0.725 | 68.5 | 97.3 | 52.9 | 2121 |
| 0.750 | 63.5 | 97.4 | 47.1 | 2243 |
| 0.775 | 60.9 | 96.8 | 44.4 | 2409 |
| 0.800 | 55.6 | 97.4 | 38.9 | 2593 |
| 0.825 | 47.7 | 97.9 | 31.5 | 2811 |
| 0.850 | 40.4 | 98.3 | 25.4 | 3086 |
| 0.875 | 35.3 | 97.5 | 21.6 | 3347 |
| 0.900 | 30.8 | 98.4 | 18.3 | 3653 |
| 0.925 | 27.2 | 98.5 | 15.8 | 3983 |
| 0.950 | 23.4 | 98.5 | 13.3 | 4370 |

- Pairwise P/R/F1 over all test documents of the language, as in Miranda et al. (2018), so the published rows are directly comparable.
- The published systems are supervised (an SVM ranker trained on the train split). NewsTrace fits a threshold on dev (dev-tuned), or two blend weights and then a threshold (dev-learned); nothing is fitted on test.
- The embedder sees the headline and the start of the body only: all-MiniLM-L6-v2 truncates input at 256 word pieces.
- No confidence intervals: resampling stories does not give a clean bootstrap for pairwise metrics, and the paper reports none to compare with.
