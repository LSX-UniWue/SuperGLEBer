# ✨SuperGLEBer ✨

SuperGLEBer (German Language Understanding Evaluation Benchmark) is a broad Natural Language Understanding benchmark suite for the German language in order to create a better understanding of the current state of German LLMs.
Our benchmark consists of 29 different tasks ranging over different types like document classification, sequence tagging, sentence similarity, and question answering.

If you use this benchmark in your research, please cite the following paper:
<https://aclanthology.org/2024.naacl-long.438/>
For the current leaderboard and more information check out the [SuperGLEBer Website](https://supergleber.professor-x.de/) 🚀

This is the updated branch that contains the new and improved version of the SuperGLEBer benchmark.

## Updates
- We added 8 new tasks of the GermEval 2025 shared task. 
- Additionally, we added support for LLM2Vec models, with the integration of bidirectional masks (Thanks @vasqu)


## Running Experiments

Create all relevant files necessary to schedule runs on a k8s/slurm cluster:

```bash
python src/template_k8s.py
```

Running a model on a task:

```bash
python src/train.py +model=gbert_base +train_args=a100 +task=news_class
```

Override config keys via CLI:

```bash
python src/train.py +model=gbert_base +train_args=a100 +task=news_class train_args.batch_size=1
```

You can find valid parameters in the provided yaml configs: <https://github.com/LSX-UniWue/SuperGLEBer/tree/paper/src/conf>


## Contact

Feel free to reach out 💡:  
[supergleber@informatik.uni-wuerzburg.de](mailto:supergleber@informatik.uni-wuerzburg.de)




## Citation
```bib
@inproceedings{pfister-hotho-2024-supergleber,
    title = "{S}uper{GLEB}er: {G}erman Language Understanding Evaluation Benchmark",
    author = "Pfister, Jan  and
      Hotho, Andreas",
    editor = "Duh, Kevin  and
      Gomez, Helena  and
      Bethard, Steven",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-long.438/",
    doi = "10.18653/v1/2024.naacl-long.438",
    pages = "7904--7923",
    abstract = "We assemble a broad Natural Language Understanding benchmark suite for the German language and consequently evaluate a wide array of existing German-capable models in order to create a better understanding of the current state of German LLMs. Our benchmark consists of 29 different tasks ranging over different types such as document classification, sequence tagging, sentence similarity, and question answering, on which we evaluate 10 different German-pretrained models, thereby charting the landscape of German LLMs. In our comprehensive evaluation we find that encoder models are a good choice for most tasks, but also that the largest encoder model does not necessarily perform best for all tasks. We make our benchmark suite and a leaderboard publically available at https://supergleber.professor-x.de and encourage the community to contribute new tasks and evaluate more models on it (https://github.com/LSX-UniWue/SuperGLEBer)."
}
```

For our GermEval 2025 participation cite: 
```bib
@inproceedings{wunderle-etal-2025-die,
    title = "Die {S}uper{GLEB}er at {G}erm{E}val 2025 Shared Tasks: Growing Pains - When More Isn{'}t Always Better",
    author = "Wunderle, Julia  and
      Pfister, Jan  and
      Hotho, Andreas",
    editor = "Wartena, Christian  and
      Heid, Ulrich",
    booktitle = "Proceedings of the 21st Conference on Natural Language Processing (KONVENS 2025): Workshops",
    month = sep,
    year = "2025",
    address = "Hannover, Germany",
    publisher = "HsH Applied Academics",
    url = "https://aclanthology.org/2025.konvens-2.45/",
    pages = "479--493"
}
```

