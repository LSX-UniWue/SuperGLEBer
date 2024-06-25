# Germeval 2017 Dataset
### Shared Task on Aspect-based Sentiment in Social Media Customer Feedback

Dataset: [url](https://sites.google.com/view/germeval2017-absa/)

Proceedings: [url](https://drive.google.com/file/d/0B0IJZ0wwnhHDc1ZpcU05Mnh2N0U/view?resourcekey=0-UfVuudnLhY8V2QZv-Cg6Mw)

### Tasks

Task 1: Relevance Classification (True, False)

Task 2: Document-level Polarity (Positive, Negative, Neutral)

### Data

| Train | Dev  | Test |
|-------|------|------|
| 20941 | 2584 | 2566 |

### Example
```
url: http://twitter.com/michael66engel/statuses/639760318445547524	
text: @DB_Bahn Wie lang wird das dauern? Ca. ?	
relevance: true	
document-level polarity:  neutral	
labels: Allgemein#Haupt:neutral
`
```

#### Given: 
Text: @DB_Bahn Wie lang wird das dauern? Ca. ?	

#### Predict: 
Task 1: true
Task 2: neutral 