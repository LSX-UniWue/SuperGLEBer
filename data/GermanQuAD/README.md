# deepset/germanquad

Dataset: [url](https://huggingface.co/datasets/deepset/germanquad)

Paper: [url](https://aclanthology.org/2021.mrqa-1.4.pdf)

### Task
Question Answering

### Data

| Train | Test |
|-------|------|
| 11518 | 2204 | 

### Example

```
'id': 40369, 
'context': "Aufzugsanlage\n\n=== Seilloser Aufzug ===\nAn der RWTH Aachen im Institut für Elektrische Maschinen wurde ein seilloser Aufzug entwickelt und ein Prototyp aufgebaut. Die Kabine wird hierbei durch zwei elektromagnetische Synchron-Linearmotoren angetrieben und somit nur durch ein vertikal bewegliches Magnetfeld gehalten bzw. bewegt. Diese Arbeit soll der Entwicklung von Aufzugsanlagen für sehr hohe Gebäude dienen. Ein Ziel ist der Einsatz mehrerer Kabinen pro Schacht, die sich unabhängig voneinander steuern lassen. Bei Auswahl des Fahrtziels vor Fahrtantritt (d.\xa0h. noch außerhalb des Aufzug) wird ein bestimmter Fahrkorb in einem der Aufzugsschächte für die Fahrt ausgewählt, mit der sich der geplante Transport am schnellsten durchführen lässt. Der Platzbedarf für die gesamte Aufzugsanlage könnte somit um ein oder mehrere Schächte reduziert werden. Da die Kabinen seillos betrieben werden, ist ein Schachtwechsel ebenfalls denkbar. Hiermit können weitere Betriebsstrategien für die seillose Aufzugsanlage entwickelt werden, zum Beispiel ein moderner Paternosteraufzug mit unabhängig voneinander beweglichen Kabinen.\nIm Rahmen der Forschungen an dem seillosen Aufzug wird ebenfalls an der Entwicklung elektromagnetischer Linearführungen gearbeitet, um den Verschleiß der seillosen Aufzugsanlage bei hohem Fahrkomfort zu minimieren. Weltweit wird an verschiedenen Forschungseinrichtungen an seillosen Antriebslösungen für Aufzüge gearbeitet. Otis betreibt zu diesem Zweck seit 2007 den ''Shibayama Test Tower''. ThyssenKrupp Elevator weihte 2017 im süddeutschen Rottweil einen Testturm ein, in welchem die Technik des seillosen Aufzugs mit Synchron-Linearmotoren im Originalmaßstab getestet wird. Der erste Aufzug dieses Typs soll 2020 in Berlin in Betrieb gehen.", 'question': 'Was kann den Verschleiß des seillosen Aufzuges minimieren?', 
'answers': {'text': ['elektromagnetischer Linearführungen', ' elektromagnetischer Linearführungen', 'elektromagnetischer Linearführungen'], 'answer_start': [1205, 1204, 1205]}}
```
