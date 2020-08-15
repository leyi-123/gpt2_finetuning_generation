from transformers import GPT2LMHeadModel,GPT2Tokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
import argparse
import logging
import numpy as np
import torch
def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
def select_top_k(predictions,k):
    predicted_index = random.choice(predictions.sort(descending = True)[1][:k]).item()
    return predicted_index
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--w_path",
        default=None,
        type=str,
        required=True,
        help="The source test path "
    )
    parser.add_argument(
        "--r_path",
        default=None,
        type=str,
        required=True,
        help="The path to store result"
    )
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        required=True,
        help="The path of trained model"
    )
    parser.add_argument(
        "--k",
        default=1,
        type=int,
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    set_seed(args)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model.to(args.device)
    max_length = 20
    w_f = open(args.w_path,"r")
    r_f = open(args.r_path,"a")
    all_text = w_f.read()
    all_text = all_text.split("\n")
    for text in all_text:
        text = text.split("\t")
        if len(text) == 3:
            persona = text[0]
            context = text[1]
            p = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(persona))
            if len(p) < 50:
                for i in range(50 - len(p)):
                    p.append(50257)
            elif len(p) > 50:
                p = p[:50]
            c = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(context))
            if len(c) < 50:
                for i in range(50 - len(c)):
                    c.append(50257)
            elif len(c) > 50:
                c = c[-50:]
            t = p + c
            input_tokens = torch.tensor([t]).to(args.device)
            length = 0
            response = list()
            while(True):
                output = model(input_tokens)
                predictions = output[0][0, -1, :]
                predicted_index = select_top_k(predictions, args.k)
                length += 1
                response.append(predicted_index)
                if (predicted_index == 50256) or (predicted_index == 50257) or (length >= 20):
                    break
                t.append(predicted_index)
                input_tokens = torch.tensor([t]).to(args.device)
            response = tokenizer.decode(response, clean_up_tokenization_spaces=True)
            r_f.write(response)
            print(response)
if __name__ == "__main__":
    main()