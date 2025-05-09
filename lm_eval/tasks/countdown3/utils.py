from typing import Dict, List
import re
import numpy as np


def doc_to_text(doc: dict) -> str:
    return "Question: Using the numbers [" + doc["input"].split(',')[0] + ", " + doc["input"].split(',')[1] + ", " + doc["input"].split(',')[2] + \
        "], create an equation that equals " + doc["input"].split(',')[3] + "." + "\n" + "Answer: "


def list_fewshot_samples() -> list[dict]:
    return [
        {
            "input": "44,2,54,64",
            "output": "2*54=108,108-44=64",
            "few_shot": "1",
        },
        {
            "input": "22,11,44,26",
            "output": "44/11=4,22+4=26",
            "few_shot": "1",
        },
        {
            "input": "2,62,96,79",
            "output": "62+96=158,158/2=79",
            "few_shot": "1",
        },
        {
            "input": "52,20,21,53",
            "output": "52-20=32,21+32=53",
            "few_shot": "1",
        },
    ]


def extract_numbers(input_string):
    numbers = re.findall(r'\d+', input_string)  
    return [int(num) for num in numbers]


def is_correct(equation):
    if '=' not in equation:
        return False  # Not a valid equation format
    left, right = equation.split('=')
    try:
        return eval(left.strip()) == int(right.strip())
    except Exception:
        return False  


def judge_number_used(numbers, required_numbers):
    if len(numbers) != 5:
        return False
    for rn in required_numbers:
        if rn not in numbers:
            return False
    for number in numbers:
        if number not in required_numbers:
            if numbers.count(number) == 1:
                return False
    return True


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    candidates = results[0]
    
    correct = 0
    for i in range(10):
        answer = candidates[i]
        # check number usage
        try:
            final_number = int(answer.split("=")[-1])
        except Exception:
            continue
        before_last_equal = answer.rpartition('=')[0]

        final_number_used = True if final_number == int(doc['input'].split(',')[-1]) else False
        numbers = extract_numbers(before_last_equal)
        required_numbers = [int(doc["input"].split(',')[i]) for i in range(3)]

        numbers_used = judge_number_used(numbers, required_numbers)

        # check whether equations are correct
        equations = [equation.strip() for equation in answer.split(',')]
        equation_correct = True
        for equation in equations:
            if not is_correct(equation):
                equation_correct = False
                break

        if equation_correct and numbers_used and final_number_used:
            correct += 1
    
    pass10 = 1 if correct > 0 else 0
    if correct == 10:
        pass1 = 1.
    else:
        pass1 = 1.0 - np.prod(1.0 - 1 / np.arange(10 - correct + 1, 10 + 1))

    return {"pass@1": pass1, "pass@10": pass10}


