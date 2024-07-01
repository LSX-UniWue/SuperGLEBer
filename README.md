# ✨SuperGLEBer ✨

SuperGLEBer (German Language Understanding Evaluation Benchmark) is a broad Natural Language Understanding benchmark suite for the German language in order to create a better understanding of the current state of German LLMs.
Our benchmark consists of 29 different tasks ranging over different types like document classification, sequence tagging, sentence similarity, and question answering.

If you use this benchmark in your research, please cite the following paper:
<https://aclanthology.org/2024.naacl-long.438/>
For the current leaderboard and more information check out the [SuperGLEBer Website](https://supergleber.professor-x.de/) 🚀

This is the branch that was used to create the results reported in the paper and will be superseded by an updated and cleaned up version in the near future.

## Running Experiments

create all relevant files necessary to schedule runs on a k8s/slurm cluster:

```bash
python src/template_k8s.py
```

running all tasks once in sequence:

```bash
python src/train.py +model=gbert_base +train_args=a100 +task=news_class
```

override config keys via CLI:

```bash
python src/train.py +model=gbert_base +train_args=a100 +task=news_class train_args.batch_size=1
```
