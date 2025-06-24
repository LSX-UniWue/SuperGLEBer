## Submission Guidelines

### Subtask 1 Overview Recap

Given a human-readable record, the developed system must classify it into one or more of the 28 predefined domains. The primary target of this task is to predict the values of `subject` property, where the available annotations will be hidden in the test dataset.

#### Data For Evaluation

During the evaluation phase, the participants will receive the test dataset consisting of technical records similar to those found in the training set, but with the subject domains removed, i.e. the `subject` property will be removed.

#### Evaluation Phase Submission

For the test dataset provided, the participant's systems should predict and classify the technical records into one or more of the 28 predefined subject domains.

##### Output Format

The expected output from participants should be a simple list of formatted domains as a list in `json` format. Each participant must submit their predictions in a file named identically to the corresponding test file, but containing only the predicted subject domains as a list in a json file.

##### Output Example

We use an example training file to illustrate this.

Consider  [shared-task-datasets/TIBKAT/all-subjects/data/train/Book/de/3A01265597X.jsonld](../shared-task-datasets/TIBKAT/all-subjects/data/train/Book/de/3A01265597X.jsonld),

A sample output file can be found in this repository. Refer to the following link: [3A01265597X.json](3A01265597X.json).

### Codabench Competition Submission Guidelines

To streamline the evaluation process, we have set up a Codabench competition. Participants are required to submit their predictions through this platform to be ranked on the leaderboard and compare results with other teams.

Participants can register in the competition using the following link: [https://www.codabench.org/competitions/8373](https://www.codabench.org/competitions/8373)

#### Submission Format

Participants must submit their predictions by following the structure described below:

1. The main folder should contain two subfolders: subtask_1 and subtask_2, corresponding to each subtask.
2. Each of these subfolders must follow the same folder structure as provided in the training dataset.
3. Once the structure is in place, zip the two folders together and submit the resulting archive to Codabench.

##### Folder Structure

```
team-name/
    ├── subtask_1/
    │   ├── Article/
    │   │   ├── en/
    │   │   └── de/
    │   ├── Book/
    │   ├── Conference/
    │   ├── Report/
    │   └── Thesis/
    └── subtask_2/
        ├── Article/
        │   ├── en/
        │   └── de/
        ├── Book/
        ├── Conference/
        ├── Report/
        └── Thesis/
```

A sample codabench submission can be found in this repository. Refer to the following link: [submission-format/team-germeval.zip](team-germeval.zip).
