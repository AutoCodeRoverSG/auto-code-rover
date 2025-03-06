import datetime
import json
import os
import sys
import threading
import time
from queue import Queue

from flask import Flask, Response, jsonify, request
from flask_cors import cross_origin

sys.path.append("/opt/auto-code-rover/")
from demo_vis.old_test_data import RawGithubTask_for_debug, test_generate_data

from app import globals, log
from app.main import get_args, run_raw_task
from app.model import common
from app.model.register import register_all_models
from app.raw_tasks import RawGithubTask

app = Flask(__name__)


def initialize_thread_cost():
    common.thread_cost.process_cost = 0.0
    common.thread_cost.process_input_tokens = 0
    common.thread_cost.process_output_tokens = 0


@app.route("/api/run_github_issue", methods=["POST"])
@cross_origin(origin="http://localhost:3000")  # nextjs cross origin
def run_github_issue():

    if not request.is_json:
        return
    register_all_models()

    data = request.get_json()
    args = get_args(
        """github-issue --output-dir output --setup-dir setup --model gpt-4-0125-preview --model-temperature 0.2"""
    )
    args.task_id = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    args.clone_link = data["repository_link"]
    args.commit_hash = data["commit_hash"]
    args.issue_link = data["issue_link"]

    globals.output_dir = args.output_dir
    if globals.output_dir is not None:
        globals.output_dir = os.path.abspath(globals.output_dir)
    # set whether brief or verbose log
    print_stdout: bool = not args.no_print
    log.print_stdout = print_stdout
    # model related
    common.set_model(args.model)
    # FIXME: make temperature part of the Model class
    common.MODEL_TEMP = args.model_temperature
    # acr related
    globals.conv_round_limit = args.conv_round_limit
    globals.enable_layered = args.enable_layered
    globals.enable_sbfl = args.enable_sbfl
    globals.enable_validation = args.enable_validation
    globals.enable_angelic = args.enable_angelic
    globals.enable_perfect_angelic = args.enable_perfect_angelic
    globals.only_save_sbfl_result = args.save_sbfl_result

    try:
        if app.debug:
            task = RawGithubTask_for_debug(
                args.task_id,
                args.clone_link,
                args.commit_hash,
                args.issue_link,
                os.path.abspath(args.setup_dir),
            )
        else:
            task = RawGithubTask(
                args.task_id,
                args.clone_link,
                args.commit_hash,
                args.issue_link,
                os.path.abspath(args.setup_dir),
            )
    except RuntimeError as e:
        error_form = {"message": str(e)}
        return jsonify(error_form), 400

    def stream_print():
        print_queue = Queue()

        def callback(data: dict):
            print_queue.put(f"{json.dumps(data)}</////json_end>")

        def run(task, callback):
            initialize_thread_cost()
            callback(
                {"category": "issue_info", "problem_statement": task.problem_statement}
            )
            run_raw_task(task, callback)

        thread = threading.Thread(target=run, args=(task, callback))
        if app.debug:
            callback(
                {"category": "issue_info", "problem_statement": task.problem_statement}
            )
            thread = threading.Thread(
                target=test_generate_data, args=(callback,)
            )  # debug
            time.sleep(1)
        else:
            thread = threading.Thread(target=run, args=(task, callback))
        thread.start()

        while thread.is_alive() or not print_queue.empty():
            if not print_queue.empty():
                yield print_queue.get()
            else:
                time.sleep(0.1)

    return Response(stream_print(), mimetype="text/event-stream")


# http://localhost:5000/
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
