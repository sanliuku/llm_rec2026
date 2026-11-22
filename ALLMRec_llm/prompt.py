"""第二阶段的 prompt 模板与拼装。
"""

from typing import Dict, List


def get_prompt():
    instruction_prompt = (
        "The description of user is: <user_des>. "
        "Given a list of concepts the user has learned before, please recommend a new concept "
        "from the candidates that the user likes to the user."
    )
    history_prompt = (
        "The user has learned the following concepts before: {historical_concepts}\n"
        "The candidates you need to select: {candidates}\n"
        "Please return your choice."
    )
    return instruction_prompt, history_prompt


def generate_prompt(instruction: str, input_text: str) -> str:
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""


def format_history(his_ids: List[int], id2title: Dict[int, str]) -> str:
    """`1.(title, <his_des>),2.(title, <his_des>)`，每个 concept 后跟一个占位符。"""
    return ','.join(f'{i + 1}.({id2title[c - 1]}, <his_des>)' for i, c in enumerate(his_ids))


def format_candidates(cand_ids: List[int], id2title: Dict[int, str]) -> str:
    """`(title, <concept_des>), (title, <concept_des>)`，按采样顺序平铺。"""
    return ', '.join(f'({id2title[c - 1]}, <concept_des>)' for c in cand_ids)


def build_input(his_ids: List[int], cand_ids: List[int], id2title: Dict[int, str],
                history_prompt: str) -> str:
    """把历史与候选填进 history 模板，得到 prompt 的 Input 段。"""
    return history_prompt.format(
        historical_concepts=format_history(his_ids, id2title),
        candidates=format_candidates(cand_ids, id2title),
    )
