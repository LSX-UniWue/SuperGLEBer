# XGLUE QADSM

Dataset: [url](https://github.com/microsoft/XGLUE)

XGLUE paper: [url](https://arxiv.org/abs/2004.01401)

### Task
Text Classification: Predict whether an advertisement (ad) is relevant to an input query (Good, Bad)

### Data

| Train | Dev  | Test  |
|-------|------|-------|
| 9000  | 1000 | 10000 |

### Example
<query, ad title, ad description, label>
````
query: immobilien von privat	
ad title: immobilie privat - Hier haben Sie mehr Info	
ad description: Bekommen Sie immobilie privat!	
label: Good
````
#### Given:
Query: immobilien von privat		

Ad title: immobilie privat - Hier haben Sie mehr Info

Ad description:	Bekommen Sie immobilie privat!		

#### Predict:
Label: Good


