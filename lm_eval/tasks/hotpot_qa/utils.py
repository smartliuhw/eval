import datasets
import sacrebleu
import numpy as np
import evaluate
import string
import re
import collections

from rouge_score import rouge_scorer, scoring
import random

f1_gen = evaluate.load("./metrics/f1")
exact_match = evaluate.load("./metrics/exact_match")

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()

def process_docs_gen(dataset: datasets.Dataset) -> datasets.Dataset:

    return dataset.map(preprocess_function)


def preprocess_function(examples):
    def _extract_facts(context):
        facts = []
        ''' rk_len = len(retrieved_knowledges)
        for i in range(rk_len):
            fact = retrieved_knowledges[i]
            facts.append(fact)
        return facts'''
        context_len = len(context["title"])
        sentences = []
        random.seed(725)
        for i in range(context_len):
            title = context["title"][i]
            # text = "\n".join(context["sentences"][i])
            sentence = context["sentences"][i]
            sentences.extend(f"{i + 1}. {sentence}")
            # sentences.extend(f"{i + 1}. {context["sentences"][i]}")
            # facts.append(title + ":\n" + text)
        return "\n\n".join(random.sample(sentences, min(5, len(sentences))))
        

    facts = _extract_facts(examples["context"])
   # facts = "\n\n".join(list(set(facts)))
    

    
#    facts = "" # set to empty space as no knowledge
    # print("question: ", examples["question"].strip(), "answer: ", examples["answer"].strip(), "facts: ", facts)
    return {
        "question": examples["question"].strip(),
        "answer": examples["answer"].strip(),
        "facts": facts.strip(),
    }
    
def process_results(doc, results):
    completion = results[0]
    ans = doc["answer"]
    exact_score = exact_match(references=[ans], predictions=[completion])
    ans_toks = get_tokens(ans)
    completion_toks = get_tokens(completion)
    common = collections.Counter(ans_toks) & collections.Counter(completion_toks)
    num_same = sum(common.values())
    if len(ans_toks) == 0 or len(completion_toks) == 0:
        f1_score = 0
    else:
        precision = 1.0 * num_same / len(completion_toks)
        recall = 1.0 * num_same / len(ans_toks)
        f1_score = (2 * precision * recall) / (precision + recall)
    return {"exact_match": exact_score, "f1": f1_score}
    

def f1(**kwargs):
    references = kwargs["references"]
    predictions = kwargs["predictions"]
    ref_toks = get_tokens(references[0])
    pred_toks = get_tokens(predictions[0])
    # print("ref_toks: ", ref_toks, "pred_toks: ", pred_toks)
    common = collections.Counter(ref_toks) & collections.Counter(pred_toks)
    # print("common: ", common)
    num_same = sum(common.values())
    if len(ref_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(ref_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(ref_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
