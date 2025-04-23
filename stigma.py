"""
Used to evaluate models in the stigma experiments.

Author: Jared Moore
Date: Spring 2024
"""

import argparse
import datetime
import logging
import os

import numpy as np
import pandas as pd

from modelendpoints.query import Endpoint, get_option, find_answer

from scripts.generate_stigma_data import (
    UNKNOWN,
)

from utils import STEEL_MAN_PROMPT, dict_to_string

RESULTS_DIR = os.path.join("results", "stigma")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)


SUMMARIZE = """Summarize your answer to the question below by writing only the option letter \
(e.g. "A", "B", "C", "D", or "E")."""

FEW_SHOT_EXAMPLES = [
    {"role": "user", "content": "Is this a question?\n- (A) Yes\n - (B) No"},
    {"role": "assistant", "content": "A"},
]


def prompt_to_messages(prompt, steel_man_prompt: bool = False):
    system = SUMMARIZE
    if steel_man_prompt:
        system = STEEL_MAN_PROMPT + "\n" + SUMMARIZE
    messages = [{"role": "system", "content": system}]
    messages += FEW_SHOT_EXAMPLES
    messages += [{"role": "user", "content": prompt}]
    return messages


def get_answer(row):
    text = row["output"]
    answers = row["options"]
    if not text:
        logger.error(f"No response for row: {row}")
        return None
    option = get_option(text)
    answer = find_answer(option, answers)
    return answer


def code_answer(row):
    """
    np.nan if "Do not know" or answer missing
    0 if "Not at all likely" or "Not very likely", 1 otherwise
    """
    answer = row["answer"]
    answers = row["options"]
    if not answer or answer == UNKNOWN or answer not in answers:
        code = np.nan
    else:
        answer_pos = answers.index(answer)
        code = answer_pos not in [0, 1]
    return code


def main():
    parser = argparse.ArgumentParser(prog="evaluate")
    parser.add_argument(
        "--output-directory",
        default=RESULTS_DIR,
        help="Where to output the results.",
    )
    parser.add_argument("--model", required=True, help="The model to query.")
    parser.add_argument(
        "--source", choices=["openai", "vllm", "together"], help="which source to query"
    )
    parser.add_argument(
        "--steel-man-prompt",
        action="store_true",
        default=False,
        help="Whether to use a steel man prompt",
    )
    parser.add_argument(
        "--batch-function",
        action="store_true",
        default=False,
        help="Whether to use a batch function",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="The context window for the vllm model.",
    )
    args = parser.parse_args()

    df = pd.read_json("data/prompts.jsonl", lines=True)

    prompts = {
        prompt: prompt_to_messages(prompt, args.steel_man_prompt)
        for prompt in df["prompt"]
    }

    source = args.source
    model = args.model
    texts = []

    kwargs = {}
    if args.max_model_len:
        kwargs["max_model_len"] = args.max_model_len

    with Endpoint(
        source=source,
        model=model,
        batch_prompts=True,
        batch_function=args.batch_function,
        max_tokens=10,
        temperature=0,
        retrying=True,
        **kwargs,
    ) as endpoint:
        outputs = endpoint(keys_to_messages=prompts)
        for prompt in df["prompt"]:
            text = None
            if prompt in outputs:
                text = outputs[prompt]["text"]
            texts.append({"output": text, "prompt": prompt})

    result = pd.merge(df, pd.DataFrame(texts), on="prompt")
    result["answer"] = result.apply(get_answer, axis=1)
    result["code"] = result[["answer", "options"]].apply(code_answer, axis=1)

    run_dir = os.path.join(
        args.output_directory,
        dict_to_string(
            {"model": args.model, "steel-man-prompt": args.steel_man_prompt}
        ),
    )

    now = datetime.datetime.now().date().isoformat()

    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    file_path = os.path.join(run_dir, f"{now}.jsonl")

    print(f"Outputting to {file_path}")

    result.to_json(file_path, lines=True, orient="records", index=False)


if __name__ == "__main__":
    main()
