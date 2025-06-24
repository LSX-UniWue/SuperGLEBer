Classification:
Submission format
Creating the submission file
Your predictions should be submitted as a CSV file, which must be named task1-predicted.csv. This file must contain headers and exactly 3 columns as specified below:

document: unique identifier for the document
comment_id: unique identifier for the comment
flausch: your predicted label (yes if candy speech is present and no if candy speech is not present in the given comment)
Example:

document,comment_id,flausch
NDY-003,1,yes
NDY-003,6,no
Uploading the predictions
To upload your predictions:

Compress the CSV into a ZIP file with the extension .zip. The ZIP file can have any name.
Make sure that the ZIP only contains one file – the CSV file with your predictions (and does not contain any sub-directories).
Upload the ZIP file as a submission to this competition.
Submission limits:

The number of submissions per subtask from each participant/team is limited to 3, with at most 3 submissions per day
---

Tagging:
Submission format
Creating the submission file
Your predictions should be submitted as a CSV file, which must be named task2-predicted.csv. This file must contain headers and exactly 5 columns as specified below:

document: unique identifier for the document
comment_id: unique identifier for the comment
type: your predicted type of the identified candy speech expression (must be one of the 10 valid types)
start: your predicted onset of the identified candy speech expression
end: your predicted offset of the identified candy speech expression
Example:

document,comment_id,type,start,end
NDY-003,4,positive feedback,48,79
NDY-003,10,compliment,0,15
Uploading the predictions
To upload your predictions:

Compress the CSV into a ZIP file with the extension .zip. The ZIP file can have any name.
Make sure that the ZIP only contains one file – the CSV file with your predictions (and does not contain any sub-directories).
Upload the ZIP file as a submission to this competition.
