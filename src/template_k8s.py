import shutil
from pathlib import Path

import jinja2
from loguru import logger

template_dir = Path("k8s/templates")
logger.info(f"template dir: {template_dir.absolute()}")

templateLoader = jinja2.FileSystemLoader(searchpath=template_dir.absolute())
templateEnv = jinja2.Environment(
    loader=templateLoader,
    # trim_blocks=True,
    # lstrip_blocks=True,
)

template_vars = [
    {
        # exemplary user, fill in your own values below
        # depending on your setup you might not need all of these values
        "lastname": "pfister",
        "affiliation": "lsx",
        "pullsecret": "multipull",
        "cluster_path": "/home/ls6/pfister/projects/Superkleber",
        "slurm_path": "/home/hpc/b185cb/b185cb10/projects/Superkleber",
        "user": "b185cb10",
        "mail_address": "supergleber@informatik.uni-wuerzburg.de",
        "uuid": 1243,
        "image_name": "ghcr.io/LSX-UniWue/SuperGLEBer:main",
        "image_tag": "main",
    },
    {
        "lastname": "wunderle",
        "affiliation": "lsx",
        "pullsecret": "multipull",
        "cluster_path": "/home/s386162/SuperGLEBer",
        "slurm_path": "/home/s386162/SuperGLEBer",
        "user": "b185cb13",
        "mail_address": "wunderle@informatik.uni-wuerzburg.de",
        "uuid": 1000,
        "image_name": "ghcr.io/LSX-UniWue/SuperGLEBer:main",
        "image_tag": "main",
    },
    {
        # this one is only used for creating the docker image on github actions
        "lastname": "supergleber",
        "affiliation": "lsx",
        "pullsecret": "multipull",
        "cluster_path": "/home/ls6/pfister/projects/Superkleber",
        "slurm_path": "/home/hpc/b185cb/b185cb10/projects/Superkleber",
        "user": "b185cb10",
        "mail_address": "supergleber@informatik.uni-wuerzburg.de",
        "uuid": 1000,
        "image_name": "ghcr.io/LSX-UniWue/SuperGLEBer:main",
        "image_tag": "main",
    },
]

logger.info("adding default values")
for template_config in template_vars:
    template_config["k8s_gpu_type"] = "a100"
    template_config["k8s_gpu_count"] = 1
    template_config["slurm_gpu_type"] = "h100"
    template_config["slurm_gpu_count"] = 1
    template_config["cpu_count"] = 8
    template_config["mem_amount"] = "48Gi"
    template_config["priorityClassName"] = "research-low"
    template_config["slurm_runtime"] = "24:00:00"
    template_config["grad_accum"] = 1
    template_config["k8s_excluded_hosts"] = [
        "uinen",  # on cuda 11.6
        "ulmo",  # on cuda 11.4
    ]

models_for_seed_study = ["gbert_base", "german_gpt2"]
tasks_for_seed_study = ["verbal_idioms", "embedding_pawsx", "ner_german_biofid"]
number_of_seeds = 4

for template_config in template_vars:
    logger.info(f"rendering templates for {template_config['lastname']}")
    out_dir: Path = Path("k8s") / template_config["lastname"]
    # delete beforehand recursively
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for file in template_dir.glob("**/*"):
        if not file.is_file() or file.stem == "train_template":
            continue
        logger.info(file.relative_to(template_dir))
        template = templateEnv.get_template(str(file.relative_to(template_dir)))
        outputText = template.render(template_config)
        # write to directory next to the template dir named after last_name but preserve structure
        out_file = out_dir / file.relative_to(template_dir)
        out_file.parent.mkdir(exist_ok=True)
        with open(out_file, "w") as f:
            f.write(outputText)

    for model_conf in Path("src/conf/model").glob("*.yaml"):
        logger.info(f"rendering model {model_conf.stem} for {template_config['lastname']}")
        template_config["model"] = model_conf.stem

        for task_conf in Path("src/conf/task").glob("*.yaml"):
            logger.info(f"rendering {task_conf.stem} with {model_conf.stem} for {template_config['lastname']}")

            template = templateEnv.get_template("train_template.yaml")
            outputText = template.render(
                template_config,
                task_name=task_conf.stem,
                job_name=f"superkleber-{model_conf.stem}-{task_conf.stem}".replace("_", "-").lower(),
            )
            out_file = out_dir / "tasks_k8s" / model_conf.stem / f"{task_conf.stem}.yaml"
            out_file.parent.mkdir(exist_ok=True, parents=True)
            with open(out_file, "w") as f:
                f.write(outputText)

            if model_conf.stem in models_for_seed_study and task_conf.stem in tasks_for_seed_study:
                # set disable_qlora to true once
                logger.info(
                    f"rendering {task_conf.stem} with {model_conf.stem} for {template_config['lastname']} without qlora"
                )

                template = templateEnv.get_template("train_template.yaml")
                outputText = template.render(
                    template_config,
                    task_name=task_conf.stem,
                    job_name=f"superkleber-{model_conf.stem}-{task_conf.stem}-no-qlora".replace("_", "-").lower(),
                    disable_qlora=True,
                    k8s_gpu_type="a100",
                )
                out_file = out_dir / "tasks_k8s" / "qlorant" / model_conf.stem / f"{task_conf.stem}-no-qlora.yaml"
                out_file.parent.mkdir(exist_ok=True, parents=True)
                with open(out_file, "w") as f:
                    f.write(outputText)

                for seed in range(number_of_seeds):
                    seed += 42 + 1
                    logger.info(
                        f"rendering {task_conf.stem} with {model_conf.stem} for {template_config['lastname']} with seed {seed}"
                    )
                    template = templateEnv.get_template("train_template.yaml")
                    outputText = template.render(
                        template_config,
                        task_name=task_conf.stem,
                        job_name=f"superkleber-{model_conf.stem}-{task_conf.stem}-{seed}".replace("_", "-").lower(),
                        seed=seed,
                    )
                    out_file = (
                        out_dir
                        / "tasks_k8s"
                        / "seeds"
                        / model_conf.stem
                        / task_conf.stem
                        / f"{task_conf.stem}-{seed}.yaml"
                    )
                    out_file.parent.mkdir(exist_ok=True, parents=True)
                    with open(out_file, "w") as f:
                        f.write(outputText)

        for task_conf in Path("src/conf/task").glob("*.yaml"):
            if "slurm_path" not in template_config:
                continue
            logger.info(f"rendering {task_conf.stem} with {model_conf.stem} for {template_config['lastname']}")
            template = templateEnv.get_template("train_template.sh")
            outputText = template.render(
                template_config,
                task_name=task_conf.stem,
                job_name=f"superkleber-{model_conf.stem}-{task_conf.stem}".replace("_", "-"),
            )
            out_file = out_dir / "tasks_slurm" / model_conf.stem / f"{task_conf.stem}.sh"
            out_file.parent.mkdir(exist_ok=True, parents=True)
            with open(out_file, "w") as f:
                f.write(outputText)
