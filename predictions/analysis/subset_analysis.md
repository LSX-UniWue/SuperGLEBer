SuperGLEBer Task Subset Analysis
==================================================

Filtered from 47 to 38 models with complete data
Removed 9 models with incomplete data

Loaded data for 38 models
Existing tasks: 29
New tasks: 8
Total tasks: 37

Top 10 models by overall average:
LSX-UniWue/ModernGBERT_1B               : 0.8066 (rank 1)
LSX-UniWue/LLaMmlein2Vec_7B             : 0.7961 (rank 2)
LSX-UniWue/LLaMmlein_7B                 : 0.7888 (rank 3)
LeoLM/leo-hessianai-7b                  : 0.7796 (rank 4)
utter-project/EuroLLM-9B                : 0.7768 (rank 5)
maxidl/DOSMo-7B-v0.2                    : 0.7760 (rank 6)
DiscoResearch/Llama3-German-8B          : 0.7697 (rank 7)
malteos/bloom-6b4-clp-german            : 0.7669 (rank 8)
LSX-UniWue/LLaMmlein2Vec_1B             : 0.7629 (rank 9)
meta-llama/Meta-Llama-3.1-8B            : 0.7628 (rank 10)

Individual task correlations with overall ranking
============================================================

existing_massive_intents                : 0.9687
existing_polarity                       : 0.9577
existing_news_class                     : 0.9543
new_llms4s                              : 0.9327
existing_offensive_lang                 : 0.9250
new_HC-dbo                              : 0.9245
new_F-Class                             : 0.9191
existing_db_aspect                      : 0.9168
existing_hotel_aspect                   : 0.8962
new_SE-Class                            : 0.8890
existing_ner_legal                      : 0.8867
new_HC-c2a                              : 0.8838
existing_verbal_idioms                  : 0.8837
new_HC-vio                              : 0.8764
existing_query_ad                       : 0.8757

Greedy subset selection
==================================================

Step 1: Added 'existing_massive_intents' -> Correlation: 0.9687
High correlation achieved (0.9687), stopping early.

Exhaustive search for subsets of size 1-5
==================================================

Size 1: Best correlation = 0.9687
  Tasks: ['existing_massive_intents']

Size 2: Best correlation = 0.9781
  Tasks: ['existing_verbal_idioms', 'existing_massive_intents']

Size 3: Best correlation = 0.9899
  Tasks: ['existing_toxic_comments', 'existing_db_aspect', 'existing_ner_legal']

Size 4: Best correlation = 0.9919
  Tasks: ['existing_toxic_comments', 'existing_db_aspect', 'existing_verbal_idioms', 'existing_ner_legal']

Size 5: Best correlation = 0.9941
  Tasks: ['existing_offensive_lang', 'existing_db_aspect', 'existing_ner_news', 'existing_germanquad', 'new_HC-dbo']

================================================================================
RANKING COMPARISON TABLES
================================================================================

Subset Size 1 (Correlation: 0.9687)
Tasks: ['existing_massive_intents']
--------------------------------------------------------------------------------

Rank Model                               Full Avg Sub Avg  Sub Rank Diff
--------------------------------------------------------------------------------

1    LSX-UniWue/ModernGBERT_1B           0.807    0.886    1T       0
2    LSX-UniWue/LLaMmlein2Vec_7B         0.796    0.886    1T       -1
3    LSX-UniWue/LLaMmlein_7B             0.789    0.886    1T       -2
4    LeoLM/leo-hessianai-7b              0.780    0.878    5T       1
5    utter-project/EuroLLM-9B            0.777    0.881    4        -1
6    maxidl/DOSMo-7B-v0.2                0.776    0.878    5T       -1
7    DiscoResearch/Llama3-German-8B      0.770    0.878    5T       -2
8    malteos/bloom-6b4-clp-german        0.767    0.872    8        0
9    LSX-UniWue/LLaMmlein2Vec_1B         0.763    0.865    10        1
10   meta-llama/Meta-Llama-3.1-8B        0.763    0.864    11        1
11   LSX-UniWue/LLaMmlein_1B             0.760    0.861    13        2
12   deepset/gbert-large                 0.759    0.839    15        3
13   flair/bueble-lm-2b                  0.753    0.852    14        1
14   meta-llama/Llama-3.2-3B             0.749    0.871    9        -5
15   utter-project/EuroLLM-1.7B          0.746    0.838    16        1

Ranking difference statistics:
  Mean absolute difference: 2.00
  Max absolute difference: 10
  Models with exact rank match: 3
  Tied ranks in subset: [np.int64(1), np.int64(5), np.int64(25), np.int64(31)]
  Number of models in ties: 10

Subset Size 2 (Correlation: 0.9781)
Tasks: ['existing_verbal_idioms', 'existing_massive_intents']
--------------------------------------------------------------------------------

Rank Model                               Full Avg Sub Avg  Sub Rank Diff
--------------------------------------------------------------------------------

1    LSX-UniWue/ModernGBERT_1B           0.807    0.922    1        0
2    LSX-UniWue/LLaMmlein2Vec_7B         0.796    0.918    3        1
3    LSX-UniWue/LLaMmlein_7B             0.789    0.921    2        -1
4    LeoLM/leo-hessianai-7b              0.780    0.916    4T       0
5    utter-project/EuroLLM-9B            0.777    0.916    4T       -1
6    maxidl/DOSMo-7B-v0.2                0.776    0.914    6        0
7    DiscoResearch/Llama3-German-8B      0.770    0.914    7        0
8    malteos/bloom-6b4-clp-german        0.767    0.913    8        0
9    LSX-UniWue/LLaMmlein2Vec_1B         0.763    0.905    12        3
10   meta-llama/Meta-Llama-3.1-8B        0.763    0.907    10        0
11   LSX-UniWue/LLaMmlein_1B             0.760    0.902    13        2
12   deepset/gbert-large                 0.759    0.898    14        2
13   flair/bueble-lm-2b                  0.753    0.897    15        2
14   meta-llama/Llama-3.2-3B             0.749    0.909    9        -5
15   utter-project/EuroLLM-1.7B          0.746    0.887    16        1

Ranking difference statistics:
  Mean absolute difference: 1.71
  Max absolute difference: 6
  Models with exact rank match: 8
  Tied ranks in subset: [np.int64(4)]
  Number of models in ties: 2

Subset Size 3 (Correlation: 0.9899)
Tasks: ['existing_toxic_comments', 'existing_db_aspect', 'existing_ner_legal']
--------------------------------------------------------------------------------

Rank Model                               Full Avg Sub Avg  Sub Rank Diff
--------------------------------------------------------------------------------

1    LSX-UniWue/ModernGBERT_1B           0.807    0.799    1        0
2    LSX-UniWue/LLaMmlein2Vec_7B         0.796    0.761    5        3
3    LSX-UniWue/LLaMmlein_7B             0.789    0.778    2        -1
4    LeoLM/leo-hessianai-7b              0.780    0.767    3        -1
5    utter-project/EuroLLM-9B            0.777    0.758    6        1
6    maxidl/DOSMo-7B-v0.2                0.776    0.765    4        -2
7    DiscoResearch/Llama3-German-8B      0.770    0.749    10        3
8    malteos/bloom-6b4-clp-german        0.767    0.752    9        1
9    LSX-UniWue/LLaMmlein2Vec_1B         0.763    0.756    7        -2
10   meta-llama/Meta-Llama-3.1-8B        0.763    0.745    11        1
11   LSX-UniWue/LLaMmlein_1B             0.760    0.753    8        -3
12   deepset/gbert-large                 0.759    0.738    13        1
13   flair/bueble-lm-2b                  0.753    0.739    12        -1
14   meta-llama/Llama-3.2-3B             0.749    0.731    14        0
15   utter-project/EuroLLM-1.7B          0.746    0.718    17        2

Ranking difference statistics:
  Mean absolute difference: 1.21
  Max absolute difference: 4
  Models with exact rank match: 8

Subset Size 4 (Correlation: 0.9919)
Tasks: ['existing_toxic_comments', 'existing_db_aspect', 'existing_verbal_idioms', 'existing_ner_legal']
--------------------------------------------------------------------------------

Rank Model                               Full Avg Sub Avg  Sub Rank Diff
--------------------------------------------------------------------------------

1    LSX-UniWue/ModernGBERT_1B           0.807    0.839    1        0
2    LSX-UniWue/LLaMmlein2Vec_7B         0.796    0.808    5        3
3    LSX-UniWue/LLaMmlein_7B             0.789    0.822    2        -1
4    LeoLM/leo-hessianai-7b              0.780    0.814    3        -1
5    utter-project/EuroLLM-9B            0.777    0.806    6        1
6    maxidl/DOSMo-7B-v0.2                0.776    0.812    4        -2
7    DiscoResearch/Llama3-German-8B      0.770    0.800    10        3
8    malteos/bloom-6b4-clp-german        0.767    0.803    8        0
9    LSX-UniWue/LLaMmlein2Vec_1B         0.763    0.804    7        -2
10   meta-llama/Meta-Llama-3.1-8B        0.763    0.796    11        1
11   LSX-UniWue/LLaMmlein_1B             0.760    0.800    9        -2
12   deepset/gbert-large                 0.759    0.793    12        0
13   flair/bueble-lm-2b                  0.753    0.790    13        0
14   meta-llama/Llama-3.2-3B             0.749    0.786    14        0
15   utter-project/EuroLLM-1.7B          0.746    0.772    17        2

Ranking difference statistics:
  Mean absolute difference: 1.16
  Max absolute difference: 3
  Models with exact rank match: 6

Subset Size 5 (Correlation: 0.9941)
Tasks: ['existing_offensive_lang', 'existing_db_aspect', 'existing_ner_news', 'existing_germanquad', 'new_HC-dbo']
--------------------------------------------------------------------------------

Rank Model                               Full Avg Sub Avg  Sub Rank Diff
--------------------------------------------------------------------------------

1    LSX-UniWue/ModernGBERT_1B           0.807    0.769    1        0
2    LSX-UniWue/LLaMmlein2Vec_7B         0.796    0.747    2        0
3    LSX-UniWue/LLaMmlein_7B             0.789    0.735    3        0
4    LeoLM/leo-hessianai-7b              0.780    0.726    4        0
5    utter-project/EuroLLM-9B            0.777    0.717    5        0
6    maxidl/DOSMo-7B-v0.2                0.776    0.717    6        0
7    DiscoResearch/Llama3-German-8B      0.770    0.713    7        0
8    malteos/bloom-6b4-clp-german        0.767    0.713    8        0
9    LSX-UniWue/LLaMmlein2Vec_1B         0.763    0.708    9        0
10   meta-llama/Meta-Llama-3.1-8B        0.763    0.702    11        1
11   LSX-UniWue/LLaMmlein_1B             0.760    0.707    10        -1
12   deepset/gbert-large                 0.759    0.690    12        0
13   flair/bueble-lm-2b                  0.753    0.687    13        0
14   meta-llama/Llama-3.2-3B             0.749    0.682    14        0
15   utter-project/EuroLLM-1.7B          0.746    0.681    15        0

Ranking difference statistics:
  Mean absolute difference: 0.68
  Max absolute difference: 4
  Models with exact rank match: 22
==================================================

SUMMARY
==================================================

Full correlation between existing and new task rankings: 0.9129

Minimal subsets with high correlation (>0.9):
  Size 1 (correlation 0.9687): ['existing_massive_intents']
  Size 2 (correlation 0.9781): ['existing_verbal_idioms', 'existing_massive_intents']
  Size 3 (correlation 0.9899): ['existing_toxic_comments', 'existing_db_aspect', 'existing_ner_legal']
  Size 4 (correlation 0.9919): ['existing_toxic_comments', 'existing_db_aspect', 'existing_verbal_idioms', 'existing_ner_legal']
  Size 5 (correlation 0.9941): ['existing_offensive_lang', 'existing_db_aspect', 'existing_ner_news', 'existing_germanquad', 'new_HC-dbo']

Greedy selection achieved 0.9687 correlation with 1 tasks:
  ['existing_massive_intents']

Analysis complete! Results saved to 'subset_analysis_results.json'
