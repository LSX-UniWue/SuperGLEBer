# Verbal Idioms

Dataset: [url](https://github.com/rafehr/vid-disambiguation-sharedtask/tree/main)

Paper: [url](https://konvens.org/proceedings/2021/papers/KONVENS_2021_Disambiguation_ST-Shared_Task_on_the_Disambiguation_of_German_Verbal_Idioms_at_KONVENS_2021.pdf)

### Task
Classification: figuratively, literally, undecidable, both

### Data

| Train | Dev   | Test |
|-------|-------|------|
| 6902  | 1488   | 1511 |

### Example 
````
T970526.220.355	Atem anhalten	figuratively	Wir sind dabei - mit Kamera . Wenn Bremen den <b>Atem</b> <b>anhält</b> und tja - das ist nun die Frage . Läßt sich der " King of Pop " von Henning Thriller-Scherf umarmen ?
````
#### Given: 

Text 1: Atem anhalten
Text 2: Wir sind dabei - mit Kamera . Wenn Bremen den <b>Atem</b> <b>anhält</b> und tja - das ist nun die Frage . Läßt sich der " King of Pop " von Henning Thriller-Scherf umarmen ?

#### Predict: 

Labels: figuratively

