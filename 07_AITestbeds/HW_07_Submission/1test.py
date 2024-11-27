"""
The following example takes pre-trained MiniLM v2 from
huggingface models repository and executes against STS benchmark dataset
on CPU and GroqChip1 through GroqFlow.
"""
import os
from transformers import AutoTokenizer, AutoModel
import torch
from demo_helpers.compute_performance import compute_performance
from demo_helpers.args import parse_args
import numpy as np
from groqflow import groqit


def evaluate_minilm(rebuild_policy=None, should_execute=True):
    # set seed for consistency
    torch.manual_seed(0)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # load pre-trained torch model
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    # PROVIDE INPUT TEXT
    text = ["Pretend you're a comedian doing stand-up comedy in the style of John Mulaney.", "Who is the funniest person you know?"]


    # TOKENIZE AND ENCODE INPUT TEXT
    encoded_input = tokenizer(
        text,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=256
    )   

    # TEST
    # encoded_input is a <class 'transformers.tokenization_utils_base.BatchEncoding'>
    print(type(encoded_input))
    #print(encoded_input.size())

    # dummy inputs to generate the groq model
    max_seq_length = 256
    inputs = {
        "input_ids": encoded_input["input_ids"].to(dtype=torch.long),
        "token_type_ids": encoded_input["token_type_ids"].to(dtype=torch.long),
        "attention_mask": encoded_input["attention_mask"].to(dtype=torch.bool),
    }

    # generate groq model
    groq_model = groqit(model, inputs, rebuild=rebuild_policy)

    # compute performance on CPU and GroqChip
    if should_execute:
        compute_performance(
            groq_model,
            model,
            dataset="stsb_multi_mt",
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            task="sentence_similarity",
        )

    print(f"Proof point {__file__} finished!")


if __name__ == "__main__":
    evaluate_minilm(**parse_args())
