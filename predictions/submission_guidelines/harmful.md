Submission Guidelines
The results can be submitted using the test data during the competition phase from 16 June to 27 June.
Submission Format
Participants must submit their results as a **CSV file** with two columns:
the ID of the tweet and its prediction.
The classes are based on the annotations in the development and test data set.
The exact format of each entry depends on the subtask:

Subtask 1: Call2Action
The CSV file must contain the following two columns:
id: The unique identifier for the tweet as appear in the dataset (column id)
c2a: The predicted label. Possible values are: TRUE, FALSE
Each entry therefore has the following columns: id;c2a

Fictitious example for the file consisting of two predictions:
id  c2a
1234  FALSE
4321  TRUE
Subtask 2: Attacks on the Democratic Basic Order
The CSV file must contain the following two columns:
id: The unique identifier for the tweet as appear in the dataset (column id)
dbo: The predicted label. Possible values are: nothing, criticism, agitation, subversive
Each entry therefore has the following columns: id;dbo

Fictitious example for the file consisting of two predictions:
id  dbo
1234  nothing
4321  criticism
Subtask 3: Violence Detection
The CSV file must contain the following two columns: *id: The unique identifier for the tweet as appear in the dataset (column id)* vio: The predicted label. Possible values are: TRUE, FALSE **Each entry therefore has the following columns: id;vio**
Fictitious example for the file consisting of two predictions:
id  vio
1234  TRUE
4321  FALSE
How to Submit your Runs
Each team can submit a maximum of three runs, i.e. make three submissions. All predictions for a subtask must be packed into a zip file named:

[team_name][run].zip
The directory must contain one file per subtask named

[team_name][run]_[task].csv
The following abbreviations for the tasks are required:

task  abbreviation
Call2Action:  c2a
Attacks on Democratic Order:  dbo
Violence:  vio
Example
The team called COMFOR is participating in the Subtask 1 - Call2Action and Subtask 2 - Violence Detection subtasks. It then submits a zip file:

COMFOR1.zip
This contains the following files:

COMFOR1_c2a.csv
COMFOR1_dbo.csv
If the team intends to submit three runs for both subtasks, the first zip file must contain the first two runs for both tasks, the second zip file must contain the second two runs, and the third zip file must contain the third runs.

The zip directory containing the runs must be submitted via Codabench. Evaluation takes place within a few hours to a few days. The results are then displayed on the leaderboard. The ranking is determined using the macro F_1 metric.
