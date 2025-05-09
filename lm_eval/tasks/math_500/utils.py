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


SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]
REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """
    Normalize a final answer to a quantitative reasoning question.

    Copied character for character from appendix D of Lewkowycz et al. (2022)
    """
    final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer


def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        out_doc = {
            "problem": doc["problem"],
            "solution": doc["solution"],
            "answer": normalize_final_answer(
                remove_boxed(last_boxed_only_string(doc["solution"]))
            ),
        }
        if getattr(doc, "few_shot", None) is not None:
            out_doc["few_shot"] = True
        return out_doc

    return dataset.map(_process_doc)


def doc_to_text(doc: dict) -> str:
    return "Problem: " + doc["problem"] + "\n\n" + "Solution:"


def get_last_number(output):
    output = re.sub(r"(\d),(\d)", r"\1\2", output)
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
    if numbers:
        return numbers[-1]
    else:
        return "NaN"
    

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
        if "\\boxed" in answer_string:
            last_box = last_boxed_only_string(answer_string)
            try:
                final = remove_boxed(last_box)
                final = normalize_final_answer(final)
                if is_equiv(doc['answer'], final):
                    exact_correct += 1
            except Exception as e:
                final = get_last_number(answer_string)
        else:
            final = get_last_number(answer_string)
        if is_equiv(doc['answer'], final):
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
        "flexible_match_pass@10": flexible_pass10}


def list_fewshot_samples() -> list[dict]:
    return [
        {
            "problem": "The areas of three squares are 16, 49 and 169.  What is the average (mean) of their side lengths?",
            "solution": "ince the areas of the three squares are 16, 49 and 169, then their side lengths are $\\sqrt{16}=4$, $\\sqrt{49}=7$ and $\\sqrt{169}=13$, respectively.\n\nThus, the average of their side lengths is $$\\frac{4+7+13}{3}=\\boxed{8}.$$",
            "few_shot": "1",
        },
        {
            "problem": "Find all $x$ such that $\\lfloor \\lfloor 2x \\rfloor - 1/2 \\rfloor = \\lfloor x + 2 \\rfloor.$",
            "solution": "Observe that $\\lfloor 2x \\rfloor$ is an integer, so it follows that $\\lfloor \\lfloor 2x \\rfloor - 1/2 \\rfloor = \\lfloor 2x \\rfloor - 1$. Also, $\\lfloor x + 2 \\rfloor = \\lfloor x \\rfloor + 2$. Thus, our equation becomes $$\\lfloor 2x \\rfloor = \\lfloor x \\rfloor + 3.$$Let $n = \\lfloor x \\rfloor,$ so $n \\le x < n + 1.$\n\nIf $x < n + \\frac{1}{2},$ then $2n \\le x < 2n + 1,$ so $\\lfloor 2x \\rfloor = 2n,$ and\n\\[2n = n + 3,\\]which means $n = 3.$\n\nIf $x \\ge n + \\frac{1}{2},$ then $2n + 1 \\le x < 2n + 2,$ so $\\lfloor 2x \\rfloor = 2n + 1,$ and\n\\[2n + 1 = n + 3,\\]which means $n = 2.$\n\nTherefore, the set of solutions is $x \\in \\boxed{\\left[ \\frac{5}{2}, \\frac{7}{2} \\right)}.$",
            "few_shot": "1",
        },
        {
            "problem": "Sequence $A$ is a geometric sequence. Sequence $B$ is an arithmetic sequence. Each sequence stops as soon as one of its terms is greater than $300.$ What is the least positive difference between a number selected from sequence $A$ and a number selected from sequence $B?$\n\n$\\bullet$ Sequence $A:$ $2,$ $4,$ $8,$ $16,$ $32,$ $\\ldots$\n\n$\\bullet$ Sequence $B:$ $20,$ $40,$ $60,$ $80,$ $100,$ $\\ldots$",
            "solution": "The terms of sequence $A$ are $2,$ $4,$ $8,$ $16,$ $32,$ $64,$ $128,$ $256,$ $512.$ The terms of sequence $B$ start from $20$ and go up by $20$ each time, so sequence $B$ is precisely all multiples of $20$ from $20$ to $320.$ We thus need to see which term in sequence $A$ is closest to a multiple of $20.$ $16,$ $64,$ and $256$ are the closest, each being $4$ away from a multiple of $20.$ So the least positive difference between a term in sequence $A$ and one in sequence $B$ is $\\boxed{4}.$",
            "few_shot": "1",
        },
        {
            "problem": "Find the domain of the function $f(x) = \\tan(\\arccos(x^2)).$",
            "solution": "For $\\arccos (x^2)$ to be defined, we must have $-1 \\le x^2 \\le 1,$ which is satisfied only for $-1 \\le x \\le 1.$  Then $\\arccos (x^2)$ will always return an angle between 0 and $\\frac{\\pi}{2}.$  Then $\\tan (\\arccos(x^2))$ is defined, unless $\\arccos(x^2) = \\frac{\\pi}{2}.$  This occurs only when $x = 0.$\n\nTherefore, the domain of $f(x)$ is $\\boxed{[-1,0) \\cup (0,1]}.$",
            "few_shot": "1",
        },
    ]


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



