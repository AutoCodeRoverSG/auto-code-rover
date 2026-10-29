"""
Analyze results from a SWE-bench run.
"""

import json
from glob import glob
from os import PathLike
from os.path import join as pjoin
from pathlib import Path
from pprint import pprint

import emojis

from app import utils as apputils
from app.api.validation import compare_fix_locations

SEPARATOR = "===================================="


def analyze(expr_dir):
    resolved = get_resolved(expr_dir)

    applicable = get_instance_names_from_dir(Path(expr_dir, "applicable_patch"))
    unmatched = get_instance_names_from_dir(Path(expr_dir, "raw_patch_but_unmatched"))
    unparsed = get_instance_names_from_dir(Path(expr_dir, "raw_patch_but_unparsed"))
    no_patch = get_instance_names_from_dir(Path(expr_dir, "no_patch"))

    # these should include all instances
    all_instances = set(applicable + unmatched + unparsed + no_patch)
    print(f"Total instances: {len(all_instances)}")
    print(SEPARATOR)

    print(f"No patch at all: {len(no_patch)}")
    pprint(no_patch)
    print(SEPARATOR)

    print(f"Unparsed: {len(unparsed)}")
    pprint(unparsed)
    print(SEPARATOR)

    print(f"Unmatched: {len(unmatched)}")
    pprint(unmatched)
    print(SEPARATOR)

    # interested in this
    applicable_but_not_resolved = sorted(list(set(applicable) - set(resolved)))
    print(f"Applicable but not resolved: {len(applicable_but_not_resolved)}")
    pprint(applicable_but_not_resolved)

    # for each applicable but not resolved, we analyze them further

    tasks_changed_same_loc = list()
    tasks_changed_diff_loc = list()

    for task in applicable_but_not_resolved:
        applicable_dir_names = glob(pjoin(expr_dir, "applicable_patch", f"{task}_*"))
        assert len(applicable_dir_names) == 1
        task_expr_dir = applicable_dir_names[0]
        model_changed_extra, changed_intersec, dev_changed_extra = analyze_one_task(
            task_expr_dir
        )
        if not changed_intersec:
            tasks_changed_same_loc.append(task)
        else:
            tasks_changed_diff_loc.append(
                (task, model_changed_extra, dev_changed_extra)
            )

    # print
    tasks_changed_same_loc.sort()
    tasks_changed_diff_loc.sort(key=lambda x: x[0])
    print(SEPARATOR)
    print("Analyzing applicable but unresolved tasks ...")

    print(SEPARATOR)
    msg = emojis.encode(
        f":thumbsup: Tasks with same changed methods: {len(tasks_changed_same_loc)}"
    )
    print(msg)
    pprint(tasks_changed_same_loc)

    print(SEPARATOR)
    print(f"Tasks with different changed methods: {len(tasks_changed_diff_loc)}")
    for task, model_changed_extra, dev_changed_extra in tasks_changed_diff_loc:
        msg = emojis.encode(f":collision: {task}:\n")
        if model_changed_extra:
            msg += f">>>>>> Model changed extra: {model_changed_extra}\n"
        if dev_changed_extra:
            msg += f">>>>>> Developer changed extra: {dev_changed_extra}\n"
        print(msg)


def analyze_one_task(task_expr_dir):
    """
    From result for one task instance, compare the fix location of the model patch
    and developer patch.

    Assumption: Model generated an applicable patch.
    """
    # (1) find out the model patch we should use
    extracted_patches = glob(pjoin(task_expr_dir, "extracted_patch_*.diff"))
    extracted_patches.sort(
        key=lambda name: int(name.removesuffix(".diff").split("_")[-1]),
        reverse=True,
    )
    model_patch = extracted_patches[0]

    # (2) get developer patch
    dev_patch = pjoin(task_expr_dir, "developer_patch.diff")

    # (3) get project path on disk
    meta_file = pjoin(task_expr_dir, "meta.json")
    with open(meta_file) as f:
        meta = json.load(f)
    project_path = meta["setup_info"]["repo_path"]

    # (4) prepare the codebase by checking out
    commit = meta["task_info"]["base_commit"]
    with apputils.cd(project_path):
        apputils.repo_reset_and_clean_checkout(commit)

    # (4) compare the two patches
    # print(f"Model patch: {model_patch} vs Developer patch: {dev_patch} on {project_path}")
    model_changed_extra, changed_intersec, dev_changed_extra = compare_fix_locations(
        model_patch, dev_patch, project_path
    )

    # results to be returned
    return model_changed_extra, changed_intersec, dev_changed_extra


# get resolved instances
def get_resolved(expr_dir):
    report_file = Path(expr_dir, "report", "report.json")
    with open(report_file) as f:
        report = json.load(f)
    return sorted(report["resolved"])


def get_instance_names_from_dir(dir_name: PathLike):
    inner_dirs = [f.name for f in Path(dir_name).iterdir() if f.is_dir()]
    names = [f.rsplit("_", 2)[0] for f in inner_dirs]
    return names
