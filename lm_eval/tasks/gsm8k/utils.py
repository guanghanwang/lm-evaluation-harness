import logging
import re
import signal
from importlib.metadata import version
from typing import Dict, List, Optional

import datasets
import numpy as np


eval_logger = logging.getLogger(__name__)


try:
    import antlr4
    import sympy
    from math_verify import parse, verify
    from sympy.parsing.latex import parse_latex

    assert version("antlr4-python3-runtime").startswith("4.11")
except (ModuleNotFoundError, AssertionError) as e:
    raise type(e)(
        "`sympy`, `math_verify` and `antlr4-python3-runtime==4.11` are required for generating translation task prompt templates. "
        "Please install the required packages via pip install lm-eval[math] or pip install -e .[math]"
    ) from e


def get_last_number(output):
    output = re.sub(r"(\d),(\d)", r"\1\2", output)
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
    if numbers:
        return numbers[-1]
    else:
        return "NaN"


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        out_doc = {
            "question": doc["question"],
            "answer": doc["answer"],
            "result": get_last_number(doc["answer"]),
        }
        if getattr(doc, "few_shot", None) is not None:
            out_doc["few_shot"] = True
        return out_doc

    return dataset.map(_process_doc)


def get_exact_number(text):
    text = re.sub(r"(\d),(\d)", r"\1\2", text)
    pattern = r"#### (\-?[0-9\.,]+)"
    matches = re.findall(pattern, text)

    # Take the first match if available
    first_match = matches[0] if matches else "NaN"
    return first_match


def is_equiv(x1: str, x2: str) -> bool:
    """
    x1 and x2 are normalized latex string
    """
    try:
        with timeout(seconds=5):
            try:
                parsed_x1 = parse_latex(x1)
                parsed_x2 = parse_latex(x2)
            except (
                sympy.parsing.latex.errors.LaTeXParsingError,
                sympy.SympifyError,
                TypeError,
            ):
                eval_logger.debug(f"couldn't parse one of {x1} or {x2}")
                return False

            try:
                diff = parsed_x1 - parsed_x2
            except TypeError:
                eval_logger.debug(f"couldn't subtract {x1} and {x2}")
                return False

            try:
                if sympy.simplify(diff) == 0:
                    return True
                else:
                    return False
            except ValueError:
                eval_logger.debug(
                    f"Had some trouble simplifying when comparing {x1} and {x2}"
                )
    except TimeoutError:
        eval_logger.debug(f"Timed out comparing {x1} and {x2}")
        return False
    except ImportError as e:
        eval_logger.error(e)
        raise
    except Exception as e:
        eval_logger.debug(f"Failed comparing {x1} and {x2} with {e}")
        return False


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    candidates = results[0]

    exact_correct, flexible_correct = 0, 0
    for i in range(10):
        answer_string = candidates[i]
        exact_number = get_exact_number(answer_string)
        if is_equiv(doc['result'], exact_number):
            exact_correct += 1
        flexible_number = get_last_number(answer_string)
        if is_equiv(doc['result'], flexible_number):
            flexible_correct += 1

    exact_pass10 = 1 if exact_correct > 0 else 0
    flexible_pass10 = 1 if flexible_correct > 0 else 0
    if exact_correct == 10:
        exact_pass1 = 1.
    else:
        exact_pass1 = 1.0 - np.prod(1.0 - 1 / np.arange(10 - exact_correct + 1, 10 + 1))
    if flexible_correct == 10:
        flexible_pass1 = 1.
    else:
        flexible_pass1 = 1.0 - np.prod(1.0 - 1 / np.arange(10 - flexible_correct + 1, 10 + 1))

    return {
        "exact_match_pass@1": exact_pass1, 
        "flexible_match_pass@1": flexible_pass1,
        "exact_match_pass@10": exact_pass10, 
        "flexible_match_pass@10": flexible_pass10
    }


class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)