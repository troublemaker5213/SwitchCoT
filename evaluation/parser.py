import random
import regex
import re
import sympy
from latex2sympy2 import latex2sympy
from typing import TypeVar, Iterable, List, Union, Any, Dict
from word2number import w2n
from utils import *



def extract_multi_choice_answer(pred_str):
    if "Problem:" in pred_str:
        pred_str = pred_str.split("Problem:", 1)[0]
    pred_str = pred_str.replace("choice is", "answer is")
    patt = regex.search(r"answer is \(?(?P<ans>[abcde])\)?", pred_str.lower())
    if patt is not None:
        return patt.group("ans").upper()
    return "placeholder"


direct_answer_trigger_for_fewshot = ("choice is", "answer is")


def choice_answer_clean(pred: str):

    # Determine if this is ICL, if so, use \n\n to split the first chunk.
    ICL = False
    for trigger in direct_answer_trigger_for_fewshot:
        if pred.count(trigger) > 1:
            ICL = True
    if ICL:
        pred = pred.split("\n\n")[0]

    # Split the trigger to find the answer.
    preds = re.split("|".join(direct_answer_trigger_for_fewshot), pred)
    if len(preds) > 1:
        answer_flag = True
        pred = preds[-1]
    else:
        answer_flag = False

    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")

    # Clean the answer based on the dataset
    tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())


    if tmp:
        pred = tmp
    else:
        pred = [pred.strip().strip(".")]

    if len(pred) == 0:
        pred = ""
    else:
        if answer_flag:
            # choose the first element in list ...
            pred = pred[0]
        else:
            # choose the last e
            pred = pred[-1]

    # Remove the period at the end, again!
    pred = pred.rstrip(".").rstrip("/")

    return pred


def find_box(pred_str: str):
    ans = pred_str.split("boxed")[-1]
    if not ans:
        return ""
    if ans[0] == "{":
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()
    return a


def clean_units(pred_str: str):
    # 清理数字中的单位表示
    """Clean the units in the number."""

    def convert_pi_to_number(code_string):
        code_string = code_string.replace("\\pi", "π")
        # Replace \pi or π not preceded by a digit or } with 3.14
        code_string = re.sub(r"(?<![\d}])\\?π", "3.14", code_string)
        # Replace instances where π is preceded by a digit but without a multiplication symbol, e.g., "3π" -> "3*3.14"
        code_string = re.sub(r"(\d)(\\?π)", r"\1*3.14", code_string)
        # Handle cases where π is within braces or followed by a multiplication symbol
        # This replaces "{π}" with "3.14" directly and "3*π" with "3*3.14"
        code_string = re.sub(r"\{(\\?π)\}", "3.14", code_string)
        code_string = re.sub(r"\*(\\?π)", "*3.14", code_string)
        return code_string

    pred_str = convert_pi_to_number(pred_str)
    pred_str = pred_str.replace("%", "/100")
    pred_str = pred_str.replace("$", "")
    pred_str = pred_str.replace("¥", "")
    pred_str = pred_str.replace("°C", "")
    pred_str = pred_str.replace(" C", "")
    pred_str = pred_str.replace("°", "")
    return pred_str


def extract_theoremqa_answer(pred: str, answer_flag: bool = True):
    if any([option in pred.lower() for option in ["yes", "true"]]):
        pred = "True"
    elif any([option in pred.lower() for option in ["no ", "false"]]):
        pred = "False"
    elif any(
        [
            option in pred.lower()
            for option in ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
        ]
    ):
        pass
    else:
        # Some of the models somehow get used to boxed output from pre-training
        if "boxed" in pred:
            pred = find_box(pred)

        if answer_flag:
            # Extract the numbers out of the string
            pred = pred.split("=")[-1].strip()
            pred = clean_units(pred)
            try:
                tmp = str(latex2sympy(pred))
                pred = str(eval(tmp))
            except Exception:
                if re.match(r"-?[\d\.]+\s\D+$", pred):
                    pred = pred.split(" ")[0]
                elif re.match(r"-?[\d\.]+\s[^\s]+$", pred):
                    pred = pred.split(" ")[0]
        else:
            # desparate search over the last number
            preds = re.findall(r"-?\d*\.?\d+", pred)
            if len(preds) >= 1:
                pred = preds[-1]
            else:
                pred = ""

    return pred

def extract_answer_from(pred_str, key, before=True):
    pred_lower = pred_str.lower()
    if key in pred_lower:
        idx = pred_lower.find(key)
        if before:
            segment = pred_str[:idx].strip()
        else:
            segment = pred_str[idx + len(key):].strip()
        match = re.search(r"\b([A-E])\b", segment.upper())
        if match:
            return [match.group(1)]
    return []


def extract_answer(pred_str , data_name):
    # 支持提取 boxed、Final Answer、the answer is 等形式
    # print("1")
    pred_str = pred_str.replace("\u043a\u0438", "")
    if "</think>" in pred_str:
        pred_str = pred_str.split("</think>")[1]
    pred_str = pred_str.strip("\n")
    pred = ""
    pred_lower = pred_str.lower()

    if data_name =="how-answer":       
        match = re.search(r"\b([A-C])\b", pred_str)
        if match:
            return match.group(1)
        
    for option in ["answer:", "answer is", "**answer**"]:
        if option in pred_lower:
            # 找到原始位置，截取后面部分，保留大小写
            idx = pred_lower.find(option)
            suffix = pred_str[idx + len(option):].strip()

            # 尝试正则提取大写 A~E，独立出现的
            match = re.search(r"\b([A-E])\b", suffix.upper())
            if match:
                return match.group(1)

    if "final answer is $" in pred_lower and "$. I hope" in pred_str:
        tmp = pred_str.split("final answer is $", 1)[1]
        pred = tmp.split("$. I hope", 1)[0].strip()
    elif "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1]
        if len(ans) == 0:
            pred =  ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
            pred = "".join(re.findall(r'[A-E]', a))
        
        else:
            a = ans.split("$")[0].strip()
            pred = a
    if pred ==  "":
        if "**Answer**:" in pred_str:
            pred = pred_str.split("**Answer**")[-1].strip()
        elif "answer:" in pred_lower:
            pred = pred_lower.split("answer:")[-1].strip()
        elif "final answer" in pred_lower:
            pred = pred_lower.split("final answer")[-1].strip()
        elif "the answer is" in pred_lower:
            # Handle Chinese few-shot multiple choice problem answer extraction
            pred = pred_lower.split("the answer is")[1].strip().split("\n\n")[0].strip()
        else:
            pred = ""

        if pred != "":
            if data_name == "gsm8k" or data_name == "math-500" or data_name == "amc23" or data_name == "aime25":
                match = re.findall(r"[-+]?\d*\.\d+|\d+", pred)
                if match:
                    pred = match[0]
                else:
                    pred = ""
        pred = pred.split("$")[0].strip()

        if pred == "":
            if data_name == "gsm8k" or data_name == "math-500" or data_name == "amc23" or data_name == "aime25":
                numbers = re.findall(r"[-+]?\d*\.\d+|\d+", pred_str)
                if numbers:
                    pred = numbers[-1]  # 取最后一个

            
    return pred


STRIP_EXCEPTIONS = ["carp_en", "minerva_math"]


def parse_ground_truth(example: Dict[str, Any], data_name):

    if data_name == "how-answer":
        abc = "ABC"
        gt_cot, gt_ans = None, abc[example["label"]]
    
    else:
        raise ValueError(f"Unknown data name: {data_name}")

    return gt_cot, gt_ans


def parse_question(example, data_name):
    question = ""


    for key in ["question", "problem", "Question", "input"]:
        if key in example:
            question = example[key]
            break

    _, gt_ans = parse_ground_truth(example, data_name)
    if isinstance(gt_ans, str):
        gt_lower = gt_ans.lower()
        if gt_lower in ["true", "false"]:
            question += " (True or False)"
        if gt_lower in ["yes", "no"]:
            question += " (Yes or No)"
    return question.strip()


def run_execute(executor, result, data_name, execute=False):
    if not result or result == "error":
        return None, None
    report = None

    prediction = extract_answer(result, data_name)

    # prediction = strip_string(prediction, skip_unit=data_name == "carp_en")
    prediction = strip_string(prediction, skip_unit=data_name in STRIP_EXCEPTIONS)
    return prediction, report


def _test_extract_answer():
    text = "To solve the equation \\(18 + p = 29\\), I need to isolate the variable \\(p\\).\n\nFirst, I'll subtract 18 from both sides of the equation to get rid of the constant term on the left side.\n\nThis gives me \\(p = 29 - 18\\).\n\nCalculating the right side, \\(29 - 18\\) equals 11.\n\nTherefore, the value of \\(p\\) is 11.\n</think>\n\nTo solve the equation \\(18 + p = 29\\), follow these steps:\n\n1. **Isolate the variable \\(p\\):**\n   \\[\n   p = 29 - 18\n   \\]\n\n2. **Calculate the right side:**\n   \\[\n   p = 11\n   \\]\n\n**Final Answer:**\n\n\\(\\boxed{11}\\)"
    print(extract_answer(text, "open-animals", use_last_number=True))
    # print(choice_answer_clean(r"\mathrm{(D)\}1,008,016"))
    # should output a dict


if __name__ == "__main__":
    _test_extract_answer()
