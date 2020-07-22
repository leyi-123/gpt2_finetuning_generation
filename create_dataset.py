import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm
from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    LineByLineTextDataset,
    PreTrainedTokenizer,
    TextDataset,
    Trainer,
    TrainingArguments,
    set_seed,
)

import pickle
import time
from torch.utils.data import DataLoader
import torch
from filelock import FileLock
from torch.utils.data.dataset import Dataset

from transformers import GPT2Tokenizer
import re

logger = logging.getLogger(__name__)
tokenizer = GPT2Tokenizer.from_pretrained('./gpt2/')
file_path = "E:/dialogue/personprofile/personachat/test_both_original.txt"
class MyTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(
        self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, overwrite_cache=False,
    ):
        assert os.path.isfile(file_path)

        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, "cached_lm_{}_{}_{}".format(tokenizer.__class__.__name__, str(block_size), filename,),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )

            else:
                logger.info(f"Creating features from dataset file at {directory}")
                self.examples = []
                with open(file_path, encoding="utf-8") as f:
                    text = f.read()
                text = text.split('\n')

                def Persona1(text):
                    persona1_l = list()
                    p1 = ''
                    flag = 0
                    flag2 = 0
                    for sentence in text:
                        if "partner's persona:" in sentence:
                            if flag == 1:
                                flag2 = 0
                                flag = 0
                                p1 = ''
                            sentence = re.sub('\d+ your persona:', '', sentence)
                            p1 = p1 + ' ' + sentence
                        elif "your persona:" not in sentence:
                            flag = 1
                            if flag2 == 0:
                                flag2 = 1
                                persona1 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(p1))
                                if len(persona1) > 50:
                                    persona1 = persona1[:50]
                                elif len(persona1) < 50:
                                    for i in range(50 - len(persona1)):
                                        persona1.append(50257)
                                persona1_l.append(persona1)
                            else:
                                persona1 = persona1_l[len(persona1_l)-1]
                                persona1_l.append(persona1)
                    return persona1_l
                def Persona2(text):
                    persona2_l = list()
                    p2 = ''
                    flag = 0
                    flag2 = 0
                    for sentence in text:
                        if "your persona:" in sentence:
                            if flag == 1:
                                flag2 = 0
                                flag = 0
                                p2 = ''
                            sentence = re.sub('\d+ your persona:', '', sentence)
                            p2 = p2 + ' ' + sentence
                        elif "partner's persona:" not in sentence:
                            flag = 1
                            if flag2 == 0:
                                flag2 = 1
                                persona2 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(p2))
                                if len(persona2) > 50:
                                    persona2 = persona2[:50]
                                elif len(persona2) < 50:
                                    for i in range(50 - len(persona2)):
                                        persona2.append(50257)
                                persona2_l.append(persona2)
                            else:
                                persona2 = persona2_l[len(persona2_l)-1]
                                persona2_l.append(persona2)
                    return persona2_l
                def Context1(text):
                    d1 = ''
                    context1_l = list()
                    for sentence in text:
                        if ("your persona:" in sentence) or ("partner's persona:" in sentence):
                            d1 = ''
                        elif ("your persona:" not in sentence) and ("partner's persona:" not in sentence):
                            chat = sentence.split('\t')
                            if len(chat) > 2:
                                context1 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(d1))
                                if len(context1) > 50:
                                    context1 = context1[-50:]
                                elif len(context1) < 50:
                                    for i in range(50 - len(context1)):
                                        context1.append(50257)
                                context1_l.append(context1)
                                r1 = chat[0] + ' <|endoftext|>'
                                r2 = chat[1] + " <|endoftext|>"
                                d1 = d1 + ' ' + r1 + ' ' + r2
                    return context1_l
                def Context2(text):
                    d2 = ''
                    context2_l = list()
                    for sentence in text:
                        if ("your persona:" in sentence) or ("partner's persona:" in sentence):
                            d2 = ''
                        elif ("your persona:" not in sentence) and ("partner's persona:" not in sentence):
                            chat = sentence.split('\t')
                            if len(chat) > 2:
                                r1 = chat[0] + ' <|endoftext|>'
                                r2 = chat[1] + " <|endoftext|>"
                                d2 = d2 + ' ' + r1
                                context2 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(d2))
                                if len(context2) > 50:
                                    context2 = context2[-50:]
                                elif len(context2) < 50:
                                    for i in range(50 - len(context2)):
                                        context2.append(50257)
                                context2_l.append(context2)
                                d2 = d2 + ' ' + r2
                    return context2_l
                def Response1(text):
                    response1_l = list()
                    for sentence in text:
                        chat = sentence.split('\t')
                        if len(chat) > 2:
                            r1 = chat[0] + ' <|endoftext|>'
                            r_t = ''
                            for s in range(len(r1)):
                                if (r1[s] in "1234567890") and (s <= 3):
                                    continue
                                else:
                                    r_t += r1[s]
                            r1 = r_t
                            response1 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(r1))
                            if len(response1) < 20:
                                for i in range(20 - len(response1)):
                                    response1.append(50257)
                            elif len(response1) > 20:
                                response1 = response1[:20]
                            response1_l.append(response1)
                    return response1_l
                def Response2(text):
                    response2_l = list()
                    for sentence in text:
                        chat = sentence.split('\t')
                        if len(chat) > 2:
                            r2 = chat[0] + ' <|endoftext|>'
                            r_t = ''
                            for s in range(len(r2)):
                                if (r2[s] in "1234567890") and (s <= 3):
                                    continue
                                else:
                                    r_t += r2[s]
                            r2 = r_t
                            response2 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(r2))
                            if len(response2) < 20:
                                for i in range(20 - len(response2)):
                                    response2.append(50257)
                            elif len(response2) > 20:
                                response2 = response2[:20]
                            response2_l.append(response2)
                    return response2_l
                persona1 = Persona1(text)
                persona2 = Persona2(text)
                context1 = Context1(text)
                context2 = Context2(text)
                response1 = Response1(text)
                response2 = Response2(text)
                for i in range(len(response1)):
                    t = persona1[i] + context1[i] + response1[i]
                    self.examples.append(t)
                for i in range(len(response2)):
                    t = persona2[i] + context2[i] + response2[i]
                    self.examples.append(t)
                # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
                # If your dataset is small, first you should loook for a bigger one :-) and second you
                # can change this behavior by adding (model specific) padding.
                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)
