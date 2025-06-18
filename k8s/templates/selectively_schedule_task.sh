# recursively search for all files named after the task name(s) and pipe the file to kubectl or sbatch
# Usage: ./selectively_schedule_task.sh <kubectl|sbatch> <task_name1,task_name2,...> => matches *<task_name>*.yaml

# get the command type (kubectl or sbatch)
command_type=$1

# get the task names (can be comma-separated)
task_names=$2

# validate command type
if [ "$command_type" != "kubectl" ] && [ "$command_type" != "sbatch" ]; then
    echo "Error: First argument must be either 'kubectl' or 'sbatch'"
    echo "Usage: ./selectively_schedule_task.sh <kubectl|sbatch> <task_name1,task_name2,...>"
    exit 1
fi

# split task names by comma and process each one
IFS=',' read -ra TASK_ARRAY <<< "$task_names"
for task_name in "${TASK_ARRAY[@]}"
do
    # trim whitespace from task name
    task_name=$(echo "$task_name" | xargs)

        echo "Processing task: $task_name"

    # set search parameters based on command type
    if [ "$command_type" = "kubectl" ]; then
        file_extension="*.yaml"
        search_path="./*/tasks_k8s"
    elif [ "$command_type" = "sbatch" ]; then
        file_extension="*.sh"
        search_path="./*/tasks_slurm"
    fi

    # get the task files which also contain the authors lastname
    task_files=$(find $search_path -name "*$task_name*$file_extension" | grep {{lastname}})

    # check if any files were found
    if [ -z "$task_files" ]; then
        echo "No files found for task: $task_name in $search_path"
        continue
    fi

    # loop through the task files and apply them with the specified command
    for task_file in $task_files
    do
        echo "Applying $task_file with $command_type"
        if [ "$command_type" = "kubectl" ]; then
            kubectl apply -f "$task_file"
        elif [ "$command_type" = "sbatch" ]; then
            sbatch "$task_file"
        fi
    done
done
