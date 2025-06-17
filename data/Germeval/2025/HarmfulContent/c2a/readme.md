# Call to Action

This subdirectory contains the data for the first subtask of the shared task on "Harmful Content Detection in Social Media" in the context of GermEval 2025: the **binary detection of Call2Action**. 

## Data annotation

The data set contains all tweets for which there was a majority decision among the four annotators as to whether or not a tweet was a call to action (TRUE) or not (FALSE). A call to action is understood to be, based on the definition of the [Oxford Dictionaries](https://www.oxfordlearnersdictionaries.com/definition/english/call-to-action), an order or request for a specific action or behaviour. The behaviour that a call to action encourages or incites may but does not have to be, criminally relevant. For example, it may also be a call for a demonstration or political campaign such as distributing leaflets. 

## Origin and structure of the data 

The data set contains a total of 9822 German tweets. Most of the data set consists of posts and comments from a right-wing extremist movement from 12/12/2014 to 07/07/2016. The data set is provided as a CSV file. An entry has the following format: 

"id";"description";"C2A"<br />
"1064396393598783";"Oliver, ich guck doch schon mindestens einmal die Woche RTL2-NEWS.";FALSE

To anonymise the data mentions in the data set were replaced as follows:
- mentions of the press/press offices/news portals: [@PRE]
- mentions of the police/police authorities: [@POL]
- mentions of groups/organisations/associations: [@GRP]
- mentions of individuals: [@IND]

For example, the mentions of the organisation Greenpeace and the TV channel ARD in the following (fictitious) tweet would be replaced as follows:
*@greenpeace_de* Euch liegt bei euren Aktionen wohl etwas an Sicherheit. Da muss man sich ja nur die letzte Doku ansehen, um das zu merken *@ARDde* => *[@GRP]* Euch liegt bei euren Aktionen wohl etwas an Sicherheit. Da muss man sich ja nur die letzte Doku ansehen, um das zu merken *[@PRE]*

## Files 

-  `c2a_trial.csv`: Sample of the training data set consisting of approximately 1,000 tweets that have been available since the trial phase to familiarise yourself with the data set. 
-  `c2a_train.csv`: Complete training data set comprising 6840 tweets 
