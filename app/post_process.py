"""
Post-process the output of the inference workflow.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections import defaultdict
from collections.abc import Mapping
from enum import Enum
from glob import glob
from os import PathLike
from os.path import join as pjoin
from pathlib import Path
from shutil import move

from loguru import logger

from app import utils as apputils
from app.agents.patch_utils import apply_edit, parse_edits
from app.model import common
from app.search.search_utils import is_test_file

META_FILE_NAME = "meta.json"


def count_and_organize_tasks(
    task_list: list[str], task_list_name: str, task_exp_names: list[str], expr_dir: str
):
    """
    A helper for extract_diff_patches.
    Generate a message to log the number of tasks in one list.
    Also organizes tasks in this list to a new folder in the experiment directory.

    Args:
        - task_list: a list of task ids
        - task_list_name: name for this list (one of the four categories)
        - task_exp_names: list of individual experiment result dir names
        - expr_dir: the overall experiment directory.

    Returns:
        - message, a string message to be written to log file.
    """
    total_num_tasks = len(task_exp_names)

    # (1) get the message ready
    message = f"Total number of tasks in {task_list_name}: {len(task_list)}/{total_num_tasks}.\n"
    for task in task_list:
        message += f"\t {task}\n"

    # (2) create a new dir and move the experiment results of these tasks there
    new_dir = pjoin(expr_dir, task_list_name)
    os.makedirs(new_dir, exist_ok=True)
    for task_exp_name in task_exp_names:
        if any([task_exp_name.startswith(x) for x in task_list]):
            # this expr dir belongs to a task in the list
            old_dir = pjoin(expr_dir, task_exp_name)
            shutil.move(old_dir, new_dir)

    return message


# track status of patch extraction
class ExtractStatus(str, Enum):
    APPLICABLE_PATCH = "APPLICABLE_PATCH"
    MATCHED_BUT_EMPTY_ORIGIN = "MATCHED_BUT_EMPTY_ORIGIN"
    MATCHED_BUT_EMPTY_DIFF = "MATCHED_BUT_EMPTY_DIFF"
    RAW_PATCH_BUT_UNMATCHED = "RAW_PATCH_BUT_UNMATCHED"
    RAW_PATCH_BUT_UNPARSED = "RAW_PATCH_BUT_UNPARSED"
    NO_PATCH = "NO_PATCH"
    IS_VALID_JSON = "IS_VALID_JSON"
    NOT_VALID_JSON = "NOT_VALID_JSON"

    def __lt__(self, other):
        order = self._worst_to_best_order()
        return order.index(self) < order.index(other)

    def __gt__(self, other):
        order = self._worst_to_best_order()
        return order.index(self) > order.index(other)

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return hash(self.value)

    def to_dir_name(self, expr_dir: str):
        return pjoin(expr_dir, self.value.lower())

    @classmethod
    def _worst_to_best_order(cls) -> list[ExtractStatus]:
        return [
            cls.NO_PATCH,
            cls.RAW_PATCH_BUT_UNPARSED,
            cls.RAW_PATCH_BUT_UNMATCHED,
            cls.MATCHED_BUT_EMPTY_DIFF,
            cls.MATCHED_BUT_EMPTY_ORIGIN,
            cls.APPLICABLE_PATCH,
        ]

    @staticmethod
    def max(statuses):
        return sorted(statuses)[-1]


def record_extract_status(individual_expr_dir: str, extract_status: ExtractStatus):
    """
    Write extract status to file, so that we can read it again when
    classifying patches
    """
    # there is 1-to-1 correspondence between agent_patch_raw and extract_status
    # FIXME: it might be better to record these status in memory so they can be easily managed.
    record_file = pjoin(individual_expr_dir, "extract_status.json")
    if not os.path.isfile(record_file):
        # record for the first time
        with open(record_file, "w") as f:
            json.dump({"extract_status": [extract_status]}, f, indent=4)
    else:
        with open(record_file) as f:
            record = json.load(f)
        record["extract_status"].append(extract_status)
        with open(record_file, "w") as f:
            json.dump(record, f, indent=4)


def read_extract_status(
    individual_expr_dir: str,
) -> tuple[ExtractStatus, str | PathLike]:
    """
    Read extract status from file. If there are multiple status recorded, read the best one.
    Returns:
        - The best extract status
        - The index of the best status in the list of all statuses. (0-based)
    """
    # we should read from the all the record
    record_files = list(Path(individual_expr_dir).glob("**/extract_status.json"))

    if not record_files:
        return ExtractStatus.NO_PATCH, ""

    global_best_status = ExtractStatus.NO_PATCH
    global_best_files = []
    for record_file in record_files:
        record = json.loads(record_file.read_text())
        # convert string to enum type
        all_status = [ExtractStatus(s) for s in record["extract_status"]]

        best_status = ExtractStatus.max(all_status)

        if best_status < global_best_status:
            continue

        if best_status > global_best_status:
            global_best_files = []
            global_best_status = best_status

        # get index from behind
        best_idx = len(all_status) - all_status[::-1].index(best_status) - 1
        best_file = record_file.with_name(f"extracted_patch_{best_idx}.diff")
        global_best_files.append(best_file)

    def key(path: Path) -> int:
        try:
            return int(path.parent.name.rpartition("_")[2])
        except ValueError:
            # Just to be compatible with old output layouts. To be removed
            # in the future.
            return 0

    global_best_files.sort(key=key, reverse=True)
    return global_best_status, global_best_files[0]


def get_final_patch_path(individual_expr_dir: str) -> str | None:
    """
    Get the final patch path from the individual experiment directory.
    If there are multiple extracted patches, need to figure out which one is the best based
    on the patch extraction history.
    """
    best_status, best_file = read_extract_status(individual_expr_dir)
    return None if best_status == ExtractStatus.NO_PATCH else str(best_file)


def extract_diff_one_instance(
    raw_patch_file: str, extracted_file: str, standalone_mode: bool = False
) -> tuple[ExtractStatus, str]:
    """
    Extract .diff patches for one instance.
    Args:
        - raw_patch_file: Path to the raw patch file produced by model.
        - extracted_file: Path where the extracted diff file goes.
        - standalone_mode: If True, the function is called from the special --extract-patch mode.
                           Specify this to True if using this function as it is for testing.
    Returns:
        - ExtractStatus.
        - An additional string containing more explanation on how patch extraction failed.
          If everything is successful, this string is empty.
    """
    if not os.path.isfile(raw_patch_file):
        return ExtractStatus.NO_PATCH, "No raw patch file is found."

    raw_patch_path = Path(raw_patch_file)

    response = raw_patch_path.read_text()
    task_dir = str(raw_patch_path.parent)

    status, summary, diff_content = convert_response_to_diff(
        response, task_dir, standalone_mode
    )

    Path(extracted_file).write_text(diff_content)

    return status, summary


# TODO: move this to PatchWriter
def convert_response_to_diff(
    response: str, task_dir: str, standalone_mode: bool = False
) -> tuple[ExtractStatus, str, str]:
    patch_content = response

    # (1) get the meta data for this task
    # task_dir = os.path.dirname(raw_patch_file)
    meta_file = pjoin(task_dir, META_FILE_NAME)
    with open(meta_file) as f:
        meta = json.load(f)

    task_info = meta["task_info"]
    setup_info = meta["setup_info"]
    repo_path = setup_info["repo_path"]  # the project dir
    base_commit = task_info["base_commit"]  # the commit to checkout

    # if not os.path.isfile(raw_patch_file):
    # return ExtractStatus.NO_PATCH, "No raw patch file is found."

    # (2) try parsing the edits
    try:
        raw_edits = parse_edits(patch_content)
    except Exception as e:
        return (
            ExtractStatus.RAW_PATCH_BUT_UNPARSED,
            f"Exception {e} happend when parsing edits.",
            "",
        )

    # filter out edits to test files
    edits = []
    for idx, edit in enumerate(raw_edits):
        if is_test_file(edit.filename):
            logger.debug("filtered out edit {}, which changes tests", idx)
            continue
        edits.append(edit)

    if not edits:
        return ExtractStatus.RAW_PATCH_BUT_UNPARSED, "No edits can be parsed.", ""

    # (3) edit parsed. check whether it can match the original program
    with apputils.cd(repo_path):
        if standalone_mode:
            # in special --extract-patch mode
            apputils.repo_reset_and_clean_checkout(base_commit)
        else:
            # extracting patch in the write_patch loop
            # we should not reset to base commit, because previous we created a new commit
            # containing the test_patch content. We should just clean the changes until HEAD.
            apputils.repo_clean_changes()
        # try to match and apply each edit
        unmatched_edit_indexes = []
        for idx, edit in enumerate(edits):
            # NOTE: do not clean here, since we want to accumulate changes from all edits
            target_file = edit.filename
            # find the target file. The model may only use the short name of the file,
            # so we need to search for it here
            found_file = apputils.find_file(repo_path, target_file)
            if found_file is None:
                unmatched_edit_indexes.append(idx)
                continue
            # try to apply this edit and update the actual file content
            applied_file = apply_edit(edit, found_file)
            if applied_file is None:
                unmatched_edit_indexes.append(idx)
                continue

        if len(unmatched_edit_indexes) == len(edits):
            # non of the edits can be matched
            # there is obvious error, and we definitely cannot extract patch
            apputils.repo_clean_changes()
            return (
                ExtractStatus.RAW_PATCH_BUT_UNMATCHED,
                "None of the edits can match the original program.",
                "",
            )

        # let's have a message describing which edits can be matched
        if unmatched_edit_indexes:
            unmatched_msg = f"Edits number {','.join([str(x+1) for x in unmatched_edit_indexes])} cannot be matched to the original program. "
        else:
            unmatched_msg = ""

        # at this point, at least some of the edits could be applied (some others may be unmatched)
        # we first try to get the diff
        diff = apputils.run_command(
            ["git", "diff"], stdout=subprocess.PIPE
        ).stdout.decode()

        # After extracting diff, we have nothing more to do in the actual code base
        apputils.repo_clean_changes()

        if not diff:
            # diff file is empty, meaning the patched program is the same as original
            # effectively, there is no edits that matched and introduced a real diff
            msg = (
                unmatched_msg
                + "The matched edits do not introduce any change to the codebase."
            )
            return ExtractStatus.MATCHED_BUT_EMPTY_DIFF, msg, ""

        edits_with_empty_before = [
            str(idx + 1) for idx, edit in enumerate(edits) if not edit.before.strip()
        ]
        if edits_with_empty_before:
            numbers = ", ".join(edits_with_empty_before)
            msg = f"Please contain **non-whitespace** original code snippet in edits number {numbers}."
            return ExtractStatus.MATCHED_BUT_EMPTY_ORIGIN, msg, ""

        # the edits resulted in a non-empty diff. We should at least save and return it
        # with open(extracted_file, "w") as f:
        # f.write(diff)

        # if all edits are matched, the `unmatched_msg` is empty string
        return ExtractStatus.APPLICABLE_PATCH, unmatched_msg, diff


def organize_experiment_results(expr_dir: str):
    """
    Assuming patches have already been extracted, organize the experiment result
    directories into a few categories and move them there.
    """
    # (1) find all the task experiment directories
    task_exp_names = [
        x
        for x in os.listdir(expr_dir)
        if os.path.isdir(pjoin(expr_dir, x))
        and "__" in x  # for filtering out other dirs like "applicable_patch"
    ]
    task_exp_dirs = [pjoin(expr_dir, x) for x in task_exp_names]

    # start organizing
    for extract_status in ExtractStatus:
        os.makedirs(extract_status.to_dir_name(expr_dir), exist_ok=True)

    for task_dir in task_exp_dirs:
        extract_status, _ = read_extract_status(task_dir)
        corresponding_dir = extract_status.to_dir_name(expr_dir)
        shutil.move(task_dir, corresponding_dir)


# NOTE: only used in the special mode of only extracting patches
def extract_diffs_and_organize_tasks(expr_dir: str):
    """
    For extracting patches for all instances at one go.
    Now, it is mainly used in the special mode of only extracting patches,
    since patch extraction of individual instance is tied to the workflow now.

    Extract diff files from raw patches, and classify tasks into categories.
    """
    log_file = pjoin(expr_dir, "extract_patches.log")
    log_file_handle = open(log_file, "w")
    task_exp_names = [
        x
        for x in os.listdir(expr_dir)
        if os.path.isdir(pjoin(expr_dir, x))
        and "__" in x  # for filtering out other dirs like "applicable_patch"
    ]
    task_exp_dirs = [pjoin(expr_dir, x) for x in task_exp_names]
    task_exp_dirs = sorted(task_exp_dirs)

    # actuall we don't need this ....
    # BUT: if we want to record how many and which tasks are in each category,
    #      can use this data structure
    # mapping from ExtractStats to a list of task ids
    all_extract_stats: Mapping[ExtractStatus, list[str]] = defaultdict(list)

    # let's work on each individual task directory
    for task_dir in task_exp_dirs:
        # (1) gather some information from the meta file
        meta_file = pjoin(task_dir, META_FILE_NAME)
        with open(meta_file) as f:
            meta = json.load(f)
        task_id = meta["task_id"]

        log_file_handle.write(f"\n\n\nGoing to extract patch for task {task_id}.\n")

        # (2) find the latest raw patch file
        raw_patch_files = [
            x for x in os.listdir(task_dir) if x.startswith("agent_patch_raw_")
        ]
        if not raw_patch_files:
            record_extract_status(task_dir, ExtractStatus.NO_PATCH)
            all_extract_stats[ExtractStatus.NO_PATCH].append(task_id)
            # no patch files at all
            continue
        # find the most recent one
        numbers = [int(file.split("_")[-1]) for file in raw_patch_files]
        numbers.sort()
        if not numbers:
            # should not happen, but just in case
            record_extract_status(task_dir, ExtractStatus.NO_PATCH)
            all_extract_stats[ExtractStatus.NO_PATCH].append(task_id)
            continue

        all_status = []
        for num in numbers:
            raw_patch_file = pjoin(task_dir, f"agent_patch_raw_{num}")
            # print(f"Extracting patch for task {task_id} from {raw_patch_file}.")

            print(task_id, num)
            # (3) perform the actual extraction
            extracted_file = pjoin(task_dir, f"extracted_patch_{num}.diff")
            extract_status, _ = extract_diff_one_instance(
                raw_patch_file, extracted_file, standalone_mode=True
            )
            all_status.append(extract_status)

            record_extract_status(task_dir, extract_status)
            all_extract_stats[extract_status].append(task_id)

        log_file_handle.write(
            f"\tPatch extraction status: {ExtractStatus.max(all_status)}\n"
        )

    # tasks has been categorized, now move them to specific folder based on the result
    organize_experiment_results(expr_dir)
    log_file_handle.close()


def extract_swe_bench_input(dir: str):
    """
    After diff format patch files have been extracted, this function collects
    them and writes a single file that can be used by swe-bench.

    Returns:
        - path to swe-bench input file.
    """
    # only look into applicable_patch dir, since we have already done
    # the categorization
    applicable_res_dir = pjoin(dir, "applicable_patch")
    # figure out what tasks have applicable patch
    task_dirs = [
        x
        for x in os.listdir(applicable_res_dir)
        if os.path.isdir(pjoin(applicable_res_dir, x))
    ]
    task_dirs = [pjoin(applicable_res_dir, x) for x in task_dirs]
    patch_files = [pjoin(x, "agent_patch_raw") for x in task_dirs]
    patch_files = [os.path.abspath(x) for x in patch_files]

    # Diff files have the name extracted_patch_{1,2,3...}.diff
    # We take the one with the largest index. This is because
    # (1) if there is no validation, then there is at most one such file,
    #     so just take it
    # (2) if there is validation, only the one with the largest index may be correct
    diff_files = []
    for x in task_dirs:
        selection_file = Path(x, "selected_patch.json")
        if selection_file.is_file():
            diff_file = pjoin(
                x, json.loads(selection_file.read_text())["selected_patch"]
            )
        else:
            _, diff_file = read_extract_status(x)
        diff_files.append(diff_file)

    diff_files = [os.path.abspath(x) for x in diff_files]

    patch_files = [x for x in patch_files if os.path.isfile(x)]
    diff_files = [x for x in diff_files if os.path.isfile(x)]

    all_results = []
    for diff_file in diff_files:
        task_dir = os.path.dirname(diff_file)
        meta_file = pjoin(task_dir, META_FILE_NAME)
        with open(meta_file) as f:
            meta = json.load(f)

        this_result = {
            "instance_id": meta["task_id"],
            "model_name_or_path": common.SELECTED_MODEL.name,
        }
        diff_content = Path(diff_file).read_text()
        if not diff_content:
            # empty diff file, dont bother sending it to swe-bench
            continue
        this_result["model_patch"] = diff_content
        all_results.append(this_result)

    swe_input_file = pjoin(dir, "predictions_for_swebench.json")
    with open(swe_input_file, "w") as f:
        json.dump(all_results, f, indent=4)

    return swe_input_file


def is_valid_json(json_str: str) -> tuple[ExtractStatus, list | dict | None]:
    """
    Check whether a json string is valid.
    """
    try:
        data = json.loads(json_str)
    except json.decoder.JSONDecodeError:
        return ExtractStatus.NOT_VALID_JSON, None
    return ExtractStatus.IS_VALID_JSON, data


"""
Main entries of the module.
"""


def reextract_organize_and_form_inputs(expr_dir: str):
    """
    Move individual experiment dirs out of the categories (applicable_patch, etc.),
    before extracting patches and organizng again.
    """
    abs_expr_dir = os.path.abspath(expr_dir)
    un_classify_expr_dir(abs_expr_dir)
    extract_diffs_and_organize_tasks(abs_expr_dir)


def un_classify_expr_dir(expr_dir: str):
    individual_expr_dirs = []
    for individual_expr_dir in glob(pjoin(expr_dir, "*", "*__*")):
        assert "info.log" in os.listdir(
            individual_expr_dir
        ), f"{individual_expr_dir} has no info.log"
        individual_expr_dirs.append(individual_expr_dir)

    for d in individual_expr_dirs:
        move(d, expr_dir)


def extract_organize_and_form_input(expr_dir):
    """
    From a directory of raw experiment result dirs, extract diff patches, organize them into
    categories, and form a single file that can be used by swe-bench.
    Args:
        - expr_dir: the overall experiment directory.
    """
    abs_expr_dir = os.path.abspath(expr_dir)
    extract_diffs_and_organize_tasks(abs_expr_dir)
    extract_swe_bench_input(abs_expr_dir)


def organize_and_form_input(expr_dir):
    """
    Only organize the experiment directories into a few categories.
    Args:
        - expr_dir: the overall experiment directory.
    """
    organize_experiment_results(expr_dir)
    return extract_swe_bench_input(expr_dir)
