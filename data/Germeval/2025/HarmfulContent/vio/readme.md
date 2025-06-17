# Violence Detection

This subdirectory contains the data for the second subtask of the shared task on "Harmful Content Detection in Social Media" in the context of GermEval 2025: the **binary detection of disturbing positive statements about violence**. 

## Data annotation

Initially, a fine-grained annotation was made into five subtypes of violence-related statements: 
- violence propensity, i.e. the will or desire to use violence oneself
- call to violence, i.e. inciting or calling on other people to commit a violent act. 
- violence support, i.e., i.e. positive approval of violence/a violent event 
- glorification, i.e. violence is presented as something particularly glorious and not just supported 
- other forms of worrying, violence-related statements 

However, the task was converted into a binary classification since some categories were severely underrepresented. Consequently, the dataset contains the majority decision as to whether or not a tweet contains any of the forms of questionable, violence-related statements (true/false).  

## Origin and structure of the data 

The data set contains a total of 10.933 German tweets. Most of the data set consists of posts and comments from a right-wing extremist movement from 12/12/2014 to 07/07/2016. The data set is provided as a CSV file. An entry has the following format: 

"id";"description";"VIO"
"1064396393598783";"Oliver, ich guck doch schon mindestens einmal die Woche RTL2-NEWS.";"nothing"

To anonymise the data mentions in the data set were replaced as follows:
- mentions of the press/press offices/news portals: [@PRE]
- mentions of the police/police authorities: [@POL]
- mentions of groups/organisations/associations: [@GRP]
- mentions of individuals: [@IND]

For example, the mentions of the organisation Greenpeace and the TV channel ARD in the following (fictitious) tweet would be replaced as follows:
*@greenpeace_de* Euch liegt bei euren Aktionen wohl etwas an Sicherheit. Da muss man sich ja nur die letzte Doku ansehen, um das zu merken *@ARDde* => *[@GRP]* Euch liegt bei euren Aktionen wohl etwas an Sicherheit. Da muss man sich ja nur die letzte Doku ansehen, um das zu merken *[@PRE]*

## Files

-  `vio_trial.csv`: Sample of the training data set consisting of approximately 1,000 tweets that have been available since the trial phase to familiarise yourself with the data set. 
-  `vio_train.csv`: Complete training data set comprising 7783 tweets 
