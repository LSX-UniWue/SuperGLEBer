# Argument Mining

Dataset: [url](https://github.com/juliaromberg/cimt-argument-mining-dataset)

Paper: [url](https://aclanthology.org/2021.argmining-1.9.pdf)

### Task
Text Classification: premise, mpos, non-arg, mpos+premise

### Data

| Train   | Dev  | Test |
|---------|------|------|
| 12494   | 1785 | 3570 |


### Example 
```
id: 12	
document_id: B1596	
sentence_nr: 1	
content: Radwege neben den Gleisen	
code: mpos	
title/text: title	
curated/single curated: CD_B	
dataset url: https://www.raddialog.bonn.de/dialoge/bonner-rad-dialog/radwege-neben-den-gleisen
```

#### Given: 
Text: Radwege neben den Gleisen	

#### Predict: 
Label: mpos