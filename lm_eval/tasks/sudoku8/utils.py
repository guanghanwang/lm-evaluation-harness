from typing import Dict, List
import re
import datasets
import numpy as np


def doc_to_text(doc: dict) -> str:
    return "Problem: Solve this 4x4 Sudoku puzzle. Rules:\n- Grid: 4 rows, 4 columns, four 2x2 subgrids.\n- Fill digits 1-4 such that no row, column, or subgrid repeats.\n- Input: 16-character string (0=empty, row-major order).\n- Output: 16-character solved string.\nNow solve this puzzle:\n" + "- Puzzle: " + doc['puzzle'] + "\n  Solution:"


def list_fewshot_samples() -> list[dict]:
    return [
        {
            "puzzle": "0002120400434001",
            "solution": "3412123421434321",
            "few_shot": "1",
        },
        {
            "puzzle": "2031034002003020",
            "solution": "2431134242133124",
            "few_shot": "1",
        },
        {
            "puzzle": "0001204304020210",
            "solution": "4321214314323214",
            "few_shot": "1",
        },
        {
            "puzzle": "0400230432010020",
            "solution": "1432231432414123",
            "few_shot": "1",
        },
    ]


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    candidates = results[0]

    correct = 0
    for i in range(10):
        answer_string = candidates[i]
        answer = answer_string.split(' ')[1].split('\n')[0].strip()
        if answer == doc['solution']:
            correct += 1
    
    pass1 = 0
    pass10 = 0
    if correct > 0:
        pass10 = 1
    if correct == 10:
        pass1 = 1
    else:
        pass1 = 1.0 - np.prod(1.0 - 1 / np.arange(10 - correct + 1, 10 + 1))

    return {"pass@1": pass1, "pass@10": pass10}