janpf@p085info010013 ~/p/SuperGLEBer (germeval25)> cd predictions/analysis && python subset_analysis.py
                                                   cd predictions/analysis && python subset_analysis.py

SuperGLEBer Task Subset Analysis
==================================================

Loaded data for 43 models
Existing tasks: 29
New tasks: 8
Total tasks: 37

Top 10 models by overall average:
LSX-UniWue/ModernGBERT_1B               : 0.8096 (rank 1)
LSX-UniWue/LLaMmlein_7B                 : 0.7922 (rank 2)
LeoLM/leo-hessianai-7b                  : 0.7828 (rank 3)
utter-project/EuroLLM-9B                : 0.7808 (rank 4)
maxidl/DOSMo-7B-v0.2                    : 0.7793 (rank 5)
DiscoResearch/Llama3-German-8B          : 0.7735 (rank 6)
malteos/bloom-6b4-clp-german            : 0.7698 (rank 7)
meta-llama/Meta-Llama-3.1-8B            : 0.7663 (rank 8)
LSX-UniWue/LLaMmlein_1B                 : 0.7630 (rank 9)
deepset/gbert-large                     : 0.7623 (rank 10)

Individual task correlations with overall ranking
============================================================

existing_massive_intents                : 0.9395
existing_polarity                       : 0.9387
existing_news_class                     : 0.9366
new_F-Class                             : 0.9234
existing_db_aspect                      : 0.9193
new_HC-c2a                              : 0.9191
new_HC-vio                              : 0.9176
existing_offensive_lang                 : 0.9114
existing_argument_mining                : 0.9105
new_SE-Class                            : 0.9041
existing_query_ad                       : 0.8989
existing_verbal_idioms                  : 0.8989
new_HC-dbo                              : 0.8975
existing_ner_legal                      : 0.8937
existing_hotel_aspect                   : 0.8868

Greedy subset selection
==================================================

Step 1: Added 'existing_massive_intents' -> Correlation: 0.9395
Step 2: Added 'new_SE-Class' -> Correlation: 0.9548
High correlation achieved (0.9548), stopping early.

Exhaustive search for subsets of size 1-5
==================================================

Size 1: Best correlation = 0.9395
  Tasks: ['existing_massive_intents']

Size 2: Best correlation = 0.9761
  Tasks: ['existing_ner_news', 'new_SE-Class']

Size 3: Best correlation = 0.9868
  Tasks: ['existing_offensive_lang', 'existing_ner_news', 'new_SE-Class']

Size 4: Best correlation = 0.9892
  Tasks: ['existing_pawsx', 'existing_ner_news', 'new_HC-dbo', 'new_SE-Class']

Size 5: Best correlation = 0.9906
  Tasks: ['existing_toxic_comments', 'existing_pawsx', 'existing_ner_news', 'new_llms4s', 'new_SE-Class']

================================================================================
RANKING COMPARISON TABLES
================================================================================

Subset Size 1 (Correlation: 0.9395)
Tasks: ['existing_massive_intents']
--------------------------------------------------------------------------------

Rank Model                               Full Avg Sub Avg  Sub Rank Diff
--------------------------------------------------------------------------------

1    LSX-UniWue/ModernGBERT_1B           0.810    0.886    1T       0
2    LSX-UniWue/LLaMmlein_7B             0.792    0.886    1T       -1
3    LeoLM/leo-hessianai-7b              0.783    0.878    4T       1
4    utter-project/EuroLLM-9B            0.781    0.881    3        -1
5    maxidl/DOSMo-7B-v0.2                0.779    0.878    4T       -1
6    DiscoResearch/Llama3-German-8B      0.773    0.878    4T       -2
7    malteos/bloom-6b4-clp-german        0.770    0.872    7        0
8    meta-llama/Meta-Llama-3.1-8B        0.766    0.864    9        1
9    LSX-UniWue/LLaMmlein_1B             0.763    0.861    11        2
10   deepset/gbert-large                 0.762    0.839    13        3
11   flair/bueble-lm-2b                  0.756    0.852    12        1
12   meta-llama/Llama-3.2-3B             0.752    0.871    8        -4
13   utter-project/EuroLLM-1.7B          0.749    0.838    14        1
14   Qwen/Qwen2.5-7B                     0.744    0.863    10        -4
15   LSX-UniWue/ModernGBERT_134M         0.742    0.822    15        0

Ranking difference statistics:
  Mean absolute difference: 2.86
  Max absolute difference: 14
  Models with exact rank match: 4
  Tied ranks in subset: [np.int64(1), np.int64(4), np.int64(23), np.int64(36), np.int64(38)]
  Number of models in ties: 15

Subset Size 2 (Correlation: 0.9761)
Tasks: ['existing_ner_news', 'new_SE-Class']
--------------------------------------------------------------------------------

Rank Model                               Full Avg Sub Avg  Sub Rank Diff
--------------------------------------------------------------------------------

1    LSX-UniWue/ModernGBERT_1B           0.810    0.746    1        0
2    LSX-UniWue/LLaMmlein_7B             0.792    0.676    2        0
3    LeoLM/leo-hessianai-7b              0.783    0.639    3        0
4    utter-project/EuroLLM-9B            0.781    0.598    6        2
5    maxidl/DOSMo-7B-v0.2                0.779    0.633    4        -1
6    DiscoResearch/Llama3-German-8B      0.773    0.621    5        -1
7    malteos/bloom-6b4-clp-german        0.770    0.592    7        0
8    meta-llama/Meta-Llama-3.1-8B        0.766    0.591    8        0
9    LSX-UniWue/LLaMmlein_1B             0.763    0.554    10        1
10   deepset/gbert-large                 0.762    0.531    13        3
11   flair/bueble-lm-2b                  0.756    0.537    11        0
12   meta-llama/Llama-3.2-3B             0.752    0.536    12        0
13   utter-project/EuroLLM-1.7B          0.749    0.580    9        -4
14   Qwen/Qwen2.5-7B                     0.744    0.529    14        0
15   LSX-UniWue/ModernGBERT_134M         0.742    0.480    16        1

Ranking difference statistics:
  Mean absolute difference: 2.00
  Max absolute difference: 6
  Models with exact rank match: 11
  Tied ranks in subset: [np.int64(30), np.int64(42)]
  Number of models in ties: 4

Subset Size 3 (Correlation: 0.9868)
Tasks: ['existing_offensive_lang', 'existing_ner_news', 'new_SE-Class']
--------------------------------------------------------------------------------

Rank Model                               Full Avg Sub Avg  Sub Rank Diff
--------------------------------------------------------------------------------

1    LSX-UniWue/ModernGBERT_1B           0.810    0.684    1        0
2    LSX-UniWue/LLaMmlein_7B             0.792    0.642    2        0
3    LeoLM/leo-hessianai-7b              0.783    0.611    3        0
4    utter-project/EuroLLM-9B            0.781    0.580    6        2
5    maxidl/DOSMo-7B-v0.2                0.779    0.601    4        -1
6    DiscoResearch/Llama3-German-8B      0.773    0.596    5        -1
7    malteos/bloom-6b4-clp-german        0.770    0.567    8        1
8    meta-llama/Meta-Llama-3.1-8B        0.766    0.569    7        -1
9    LSX-UniWue/LLaMmlein_1B             0.763    0.551    9        0
10   deepset/gbert-large                 0.762    0.522    11        1
11   flair/bueble-lm-2b                  0.756    0.520    12        1
12   meta-llama/Llama-3.2-3B             0.752    0.516    13        1
13   utter-project/EuroLLM-1.7B          0.749    0.544    10        -3
14   Qwen/Qwen2.5-7B                     0.744    0.509    14        0
15   LSX-UniWue/ModernGBERT_134M         0.742    0.453    16        1

Ranking difference statistics:
  Mean absolute difference: 1.51
  Max absolute difference: 6
  Models with exact rank match: 10
  Tied ranks in subset: [np.int64(42)]
  Number of models in ties: 2

Subset Size 4 (Correlation: 0.9892)
Tasks: ['existing_pawsx', 'existing_ner_news', 'new_HC-dbo', 'new_SE-Class']
--------------------------------------------------------------------------------

Rank Model                               Full Avg Sub Avg  Sub Rank Diff
--------------------------------------------------------------------------------

1    LSX-UniWue/ModernGBERT_1B           0.810    0.826    1        0
2    LSX-UniWue/LLaMmlein_7B             0.792    0.783    2        0
3    LeoLM/leo-hessianai-7b              0.783    0.772    3        0
4    utter-project/EuroLLM-9B            0.781    0.752    6        2
5    maxidl/DOSMo-7B-v0.2                0.779    0.760    4        -1
6    DiscoResearch/Llama3-German-8B      0.773    0.756    5        -1
7    malteos/bloom-6b4-clp-german        0.770    0.742    7        0
8    meta-llama/Meta-Llama-3.1-8B        0.766    0.741    8        0
9    LSX-UniWue/LLaMmlein_1B             0.763    0.719    10        1
10   deepset/gbert-large                 0.762    0.712    11T       1
11   flair/bueble-lm-2b                  0.756    0.711    13        2
12   meta-llama/Llama-3.2-3B             0.752    0.712    11T       -1
13   utter-project/EuroLLM-1.7B          0.749    0.737    9        -4
14   Qwen/Qwen2.5-7B                     0.744    0.710    14        0
15   LSX-UniWue/ModernGBERT_134M         0.742    0.675    16        1

Ranking difference statistics:
  Mean absolute difference: 1.37
  Max absolute difference: 4
  Models with exact rank match: 12
  Tied ranks in subset: [np.int64(11)]
  Number of models in ties: 2

Subset Size 5 (Correlation: 0.9906)
Tasks: ['existing_toxic_comments', 'existing_pawsx', 'existing_ner_news', 'new_llms4s', 'new_SE-Class']
--------------------------------------------------------------------------------

Rank Model                               Full Avg Sub Avg  Sub Rank Diff
--------------------------------------------------------------------------------

1    LSX-UniWue/ModernGBERT_1B           0.810    0.780    1        0
2    LSX-UniWue/LLaMmlein_7B             0.792    0.744    2        0
3    LeoLM/leo-hessianai-7b              0.783    0.731    3        0
4    utter-project/EuroLLM-9B            0.781    0.716    6        2
5    maxidl/DOSMo-7B-v0.2                0.779    0.722    4        -1
6    DiscoResearch/Llama3-German-8B      0.773    0.716    5        -1
7    malteos/bloom-6b4-clp-german        0.770    0.707    7        0
8    meta-llama/Meta-Llama-3.1-8B        0.766    0.705    8        0
9    LSX-UniWue/LLaMmlein_1B             0.763    0.691    10        1
10   deepset/gbert-large                 0.762    0.685    11        1
11   flair/bueble-lm-2b                  0.756    0.678    12        1
12   meta-llama/Llama-3.2-3B             0.752    0.676    14        2
13   utter-project/EuroLLM-1.7B          0.749    0.695    9        -4
14   Qwen/Qwen2.5-7B                     0.744    0.678    13        -1
15   LSX-UniWue/ModernGBERT_134M         0.742    0.649    16        1

Ranking difference statistics:
  Mean absolute difference: 1.26
  Max absolute difference: 5
  Models with exact rank match: 12
==================================================

SUMMARY
==================================================

Full correlation between existing and new task rankings: 0.8224

Minimal subsets with high correlation (>0.9):
  Size 1 (correlation 0.9395): ['existing_massive_intents']
  Size 2 (correlation 0.9761): ['existing_ner_news', 'new_SE-Class']
  Size 3 (correlation 0.9868): ['existing_offensive_lang', 'existing_ner_news', 'new_SE-Class']
  Size 4 (correlation 0.9892): ['existing_pawsx', 'existing_ner_news', 'new_HC-dbo', 'new_SE-Class']
  Size 5 (correlation 0.9906): ['existing_toxic_comments', 'existing_pawsx', 'existing_ner_news', 'new_llms4s', 'new_SE-Class']

Greedy selection achieved 0.9548 correlation with 2 tasks:
  ['existing_massive_intents', 'new_SE-Class']
