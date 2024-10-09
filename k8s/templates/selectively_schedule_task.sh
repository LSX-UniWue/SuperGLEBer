# recursively search for all files named after the first argument and pipe the file to kubectl
# Usage: ./selectively_schedule_task.sh <task_name> => matches *<task_name>*.yaml

# get the task name
task_name=$1

# get the task files which also contain the authors lastname
task_files=$(find . -name "*$task_name*.yaml" | grep {{lastname}})

# loop through the task files and apply them
for task_file in $task_files
do
    kubectl apply -f $task_file
done
