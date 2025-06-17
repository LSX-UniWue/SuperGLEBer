# Detection of attacks on the basic democratic order

This subdirectory contains the data for the third subtask of the shared task on "Harmful Content Detection in Social Media" in the context of GermEval 2025: the **fine-grained detection of various attacks on the free and democratic basic order of the Federal Republic of Germany**. 

## Data annotation

The data set contains all tweets for which the four annotators could reach a majority decision regarding the form of attack. Specifically, the annotators assigned the tweets to one of the following classes: 
- **subversive:** A will is expressed to forcibly remove the existing government and overthrow it (e.g., through militant action, disruption of the power grid, etc.).   
- **agitation:** Agitative efforts are expressed. That includes the announcement of actions such as the dissemination of propaganda material of unconstitutional and terrorist organisations or the damaging of state symbols such as the flag of the Federal Republic of Germany.   
- **criticism:** Tweets in which legitimate criticism of the government, officials, government employees, authorities or parties was expressed were assigned to this class. 
- **nothing:** Tweets in this category contain neither criticism nor an attack against the free democratic basic order. However, neutral or positive statements on government decisions can be expressed in the tweets. 

## Origin and structure of the data 

The data set contains a total of 9.307 German tweets. Most of the data set consists of posts and comments from a right-wing extremist movement from 12/12/2014 to 07/07/2016. The data set is provided as a CSV file. An entry has the following format: 

"id";"description";"DBO"
"1064396393598783";"Oliver, ich guck doch schon mindestens einmal die Woche RTL2-NEWS.";"nothing"

To anonymise the data mentions in the data set were replaced as follows:
- mentions of the press/press offices/news portals: [@PRE]
- mentions of the police/police authorities: [@POL]
- mentions of groups/organisations/associations: [@GRP]
- mentions of individuals: [@IND]

For example, the mentions of the organisation Greenpeace and the TV channel ARD in the following (fictitious) tweet would be replaced as follows:
*@greenpeace_de* Euch liegt bei euren Aktionen wohl etwas an Sicherheit. Da muss man sich ja nur die letzte Doku ansehen, um das zu merken *@ARDde* => *[@GRP]* Euch liegt bei euren Aktionen wohl etwas an Sicherheit. Da muss man sich ja nur die letzte Doku ansehen, um das zu merken *[@PRE]*

## Files

-  `dbo_trial.csv`: Sample of the training data set consisting of approximately 1,000 tweets that have been available since the trial phase to familiarise yourself with the data set. 
-  `dbo_train.csv`: Complete training data set comprising 7454 tweets 
