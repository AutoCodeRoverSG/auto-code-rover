import json
from collections import defaultdict
from collections.abc import Iterable
from itertools import cycle
from os import PathLike
from os.path import samefile
from pathlib import Path
from shutil import copy2

from loguru import logger
from natsort import natsorted

from app import config
from app.agents import agent_select
from app.agents.agent_common import InvalidLLMResponse
from app.agents.agent_reproducer import NoReproductionStep, TestAgent
from app.agents.agent_write_patch import PatchAgent
from app.api import validation
from app.api.review_manage import ReviewManager
from app.api.validation import evaluate_patch
from app.data_structures import BugLocation
from app.log import print_banner, print_issue
from app.manage import ProjectApiManager
from app.model.common import set_model
from app.task import Task


def write_patch_iterative_with_review(
    task: Task,
    output_dir: str,
    review_manager: ReviewManager,
    retries=3,
) -> bool:
    logger.info("Start generating patches with reviewer")
    patch_gen = review_manager.generator()

    eval_summary = None
    for _ in range(retries):
        try:
            patch_handle, patch_content = patch_gen.send(eval_summary)
            logger.info("Reviewer approved patch: {}", patch_handle)
        except StopIteration:
            break

        logger.info("Begin evaluating patch: {}", patch_handle)
        eval_passed, eval_summary = validation.evaluate_patch(
            task, patch_handle, patch_content, output_dir
        )

        if eval_passed:
            patch_gen.close()

            logger.info(
                "Patch {} passed evaluation. Ending patch generation", patch_handle
            )
            return True

        logger.info("Patch {} failed evaluation", patch_handle)

    return False


def write_patch_iterative(
    task: Task,
    output_dir: str,
    review_manager: ReviewManager,
    retries=3,
) -> bool:
    logger.info("Start generating patches without reviewer")

    patch_gen = review_manager.patch_only_generator()

    for _ in range(retries):
        try:
            patch_handle, patch_content = patch_gen.send(None)
            logger.info("Generated applicable patch: {}", patch_handle)
        except StopIteration:
            break

        logger.info("Begin evaluating patch: {}", patch_handle)
        eval_passed, _ = validation.evaluate_patch(
            task, patch_handle, patch_content, output_dir
        )

        if eval_passed:
            patch_gen.close()

            logger.info(
                "Patch {} passed evaluation. Ending patch generation", patch_handle
            )
            return True

        logger.info("Patch {} failed evaluation", patch_handle)

    return False


def run_one_task(task: Task, output_dir: str, model_names: Iterable[str]) -> bool:
    """
    Main entry point to run inference on one task.
    Args:
        output_dir (str): Path to the output directory.
        api_manager (ProjectApiManager): The already-initialized API manager.
        problem_stmt (str): The original problem statement submitted to the task issue.
    """
    assert model_names

    model_name_cycle = cycle(model_names)

    for idx in range(config.overall_retry_limit):
        model_name = next(model_name_cycle)
        set_model(model_name)

        logger.info("Starting overall retry {} with model {}", idx, model_name)

        out_dir = Path(output_dir, f"output_{idx}")

        out_dir.mkdir(parents=True, exist_ok=True)

        # meta.json is used later by convert_response_to_diff(),
        # so it needs to be copied over
        meta_file = Path(output_dir, "meta.json")
        if meta_file.exists():
            copy2(meta_file, out_dir)

        api_manager = ProjectApiManager(task, str(out_dir))

        if _run_one_task(str(out_dir), api_manager, task.get_issue_statement()):
            logger.info("Overall retry {} succeeded; ending workflow", idx)
            break

        logger.info("Overall retry {} failed; proceeding to next retry", idx)

    logger.info("Starting patch selection")

    selected, details = select_patch(task, output_dir)
    Path(output_dir, "selected_patch.json").write_text(json.dumps(details, indent=4))

    logger.info("Selected patch {}. Reason: {}", selected, details["reason"])

    return True


def _find_reviewer_approved_patch(patches: list, candidate_patches: list):
    for p in candidate_patches:
        index = p.with_suffix("").name.rpartition("_")[2]
        reviews = natsorted(
            list(p.parent.glob(f"review_p{index}_t*.json")), reverse=True
        )
        if not reviews:
            continue
        assert len(reviews) == 1, p
        if json.loads(reviews[0].read_text())["patch-correct"] == "yes":
            last_patch = natsorted(patches)[-1]
            assert samefile(
                p, last_patch
            ), f"{p} is approved and passes validation, but the last patch was {last_patch}"
            return p
    return None


def _select_among_patches(task: Task, patches: list, reason_suffix: str):
    agent_comment = None
    thread = None
    try:
        index, agent_comment, thread = agent_select.run(
            task.get_issue_statement(), [p.read_text() for p in patches]
        )
        reason = f"agent-selected,{reason_suffix}"
    except Exception:
        index = -1
        reason = f"agent-error,{reason_suffix}"
    return patches[index], reason, agent_comment, thread


def select_patch(task: Task, output_dir: str | PathLike) -> tuple[str, dict]:

    patches = natsorted(list(Path(output_dir).glob("**/extracted_patch_*.diff")))

    # TODO: These candidate patches must have been dismissed by reviewer. Maybe an
    # assertion should be added to confirm this.
    candidate_patches = [p for p in patches if may_pass_regression_tests(task, p)]

    agent_comment = None
    thread = None

    selected_patch = _find_reviewer_approved_patch(patches, candidate_patches)
    if selected_patch is not None:
        reason = "reviewer-approved"
    elif len(candidate_patches) > 1:
        selected_patch, reason, agent_comment, thread = _select_among_patches(
            task, candidate_patches, "multiple-pass-regression"
        )
    elif len(candidate_patches) == 1:
        selected_patch = candidate_patches[0]
        reason = "no-agent,single-pass-regression"
    else:
        selected_patch, reason, agent_comment, thread = _select_among_patches(
            task, patches, "none-pass-regression"
        )

    rel_selected_patch = str(selected_patch.relative_to(output_dir))

    result = {
        "selected_patch": rel_selected_patch,
        "reason": reason,
    }

    if agent_comment is not None:
        result["agent_comment"] = agent_comment

    if thread is not None:
        thread.save_to_file(Path(output_dir, "agent_selection.json"))

    return str(selected_patch.relative_to(output_dir)), result


def may_pass_regression_tests(task: Task, patch_file: str | PathLike) -> bool:
    if not config.enable_validation:
        return True

    patch_file = Path(patch_file)

    patch_idx = patch_file.with_suffix("").name.rpartition("_")[2]

    regression_file = patch_file.with_name(f"regression_{patch_idx}.json")
    if regression_file.exists():
        return json.loads(regression_file.read_text())["no_additional_failure"]

    task.reset_project()
    pass_evaluation, _ = evaluate_patch(
        task, patch_idx, patch_file.read_text(), str(patch_file.parent)
    )

    return pass_evaluation


def _run_one_task(
    output_dir: str, api_manager: ProjectApiManager, problem_stmt: str
) -> bool:
    print_banner("Starting AutoCodeRover on the following issue")
    print_issue(problem_stmt)

    test_agent = TestAgent(api_manager.task, output_dir)

    repro_result_map = {}
    repro_stderr = ""
    reproduced = False
    reproduced_test_content = None
    try:
        test_handle, test_content, orig_repro_result = (
            test_agent.write_reproducing_test_without_feedback()
        )
        test_agent.save_test(test_handle)

        coord = (PatchAgent.EMPTY_PATCH_HANDLE, test_handle)
        repro_result_map[coord] = orig_repro_result

        if orig_repro_result.reproduced:
            repro_stderr = orig_repro_result.stderr
            reproduced = True
            reproduced_test_content = test_content
        # TODO: utilize the test for localization
    except NoReproductionStep:
        logger.info(
            "Test agent decides that the issue statement does not contain "
            "reproduction steps; skipping reproducer tracing"
        )
    except InvalidLLMResponse:
        logger.warning("Failed to write a reproducer test; skipping reproducer tracing")

    if config.enable_sbfl:
        sbfl_result, *_ = api_manager.fault_localization()
    else:
        sbfl_result = ""

    bug_locs: list[BugLocation]
    bug_locs, search_msg_thread = api_manager.search_manager.search_iterative(
        api_manager.task, sbfl_result, repro_stderr, reproduced_test_content
    )

    logger.info("Search completed. Bug locations: {}", bug_locs)

    # logger.info("Additional class context code: {}", class_context_code)
    # done with search; dump the tool calls used for recording
    api_manager.search_manager.dump_tool_call_layers_to_file()

    # Write patch
    print_banner("PATCH GENERATION")
    logger.debug("Gathered enough information. Invoking write_patch.")

    review_manager = ReviewManager(
        search_msg_thread,
        bug_locs,
        api_manager.search_manager,
        api_manager.task,
        output_dir,
        test_agent,
        repro_result_map,
    )

    if config.reproduce_and_review and reproduced:
        try:
            return write_patch_iterative_with_review(
                api_manager.task, output_dir, review_manager
            )
        # this exception can arise when writing new reproducers
        except NoReproductionStep:
            pass

    result = write_patch_iterative(api_manager.task, output_dir, review_manager)
    logger.info(
        "Invoked write_patch. Since there is no reproducer, the workflow will be terminated."
    )
    return result


if __name__ == "__main__":
    from app.raw_tasks import RawSweTask

    config.enable_validation = True

    applicable_path = Path(
        "/media/media0/haifeng/projects/reverse-prompt/acr-plus/experiment/06-13-docker-val-loop-lite-try-2-rand/applicable_patch/"
    )
    task_dirs = list(applicable_path.glob("*"))
    for task_dir in task_dirs:
        meta = json.loads(task_dir.joinpath("meta.json").read_text())
        raw_task = RawSweTask(meta["task_id"], meta["setup_info"], meta["task_info"])
        task = raw_task.to_task()
        selected_patch, reason = select_patch(task, task_dir)

        task_dir.joinpath("selected_patch.json").write_text(
            json.dumps({"selected_patch": selected_patch, "reason": reason}, indent=4)
        )
