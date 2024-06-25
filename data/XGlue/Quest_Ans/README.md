# XGLUE QAM

Dataset: [url](https://github.com/microsoft/XGLUE)

XGLUE paper: [url](https://arxiv.org/abs/2004.01401)

### Task
Sentence Classification: QA matching (True, False)

### Data

| Train | Dev  | Test  |
|-------|------|-------|
| 9000  | 1000 | 10000 |

### Example
 <question, passage, label>
````
question: gramm pro mol in gramm	
passage: Wenn Sie nun die gegebene Stoffmenge in mol mit der abgelesenen molekularen Masse g/mol multiplizieren, erhalten Sie die Masse in Gramm Ihres Stoffes oder Ihrer chemischen Verbindung.	
label: 0

````
#### Given:
Question: gramm pro mol in gramm	

Passage: Wenn Sie nun die gegebene Stoffmenge in mol mit der abgelesenen molekularen Masse g/mol multiplizieren, erhalten Sie die Masse in Gramm Ihres Stoffes oder Ihrer chemischen Verbindung.

#### Predict:
Label: 0



