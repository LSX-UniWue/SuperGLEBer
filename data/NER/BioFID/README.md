# NER BIOFID

Dataset: [url](https://github.com/texttechnologylab/BIOfid)

Paper: [url](https://aclanthology.org/K19-1081.pdf)

### Task
Named Entity Recognition: TAX, OTHER, LOC, PER, TME, ORG

### Data

| Train | Dev  | Test  |
|-------|------|-------|
| 12668  | 1584 | 1584 |

### Example
```
über über APPR O
die der ART O
Nahrungsumstellung Nahrungsumstellung NN O
des der ART O
Graureihers Graureiher NN B-TAX
richtig richtig ADJD O
ist sein VAFIN O
, -- $, O
möchte möchten VMFIN O
ich ich PPER O
bezweifeln bezweifeln VVFIN O
. -- $. O
```

