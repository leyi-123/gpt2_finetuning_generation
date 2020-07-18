import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional

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

import torch
from filelock import FileLock
from torch.utils.data.dataset import Dataset

from transformers import GPT2Tokenizer
import re

logger = logging.getLogger(__name__)
tokenizer = GPT2Tokenizer.from_pretrained('./gpt2/')
file_path = "E:/dialogue/personprofile/personachat/train_both_original.txt"
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
                p1 = ''
                p2 = ''
                d1 = ''
                d2 = ''
                t1 = ''
                t2 = ''
                flag = 0
                for sentence in text:
                    if "your persona:" in sentence:
                        if flag == 1:
                            flag = 0
                            p1 = ''
                            p2 = ''
                            d1 = ''
                            d2 = ''
                        sentence = re.sub('\d+ your persona:', '', sentence)
                        p2 += (sentence)
                    elif "partner's persona" in sentence:
                        if flag == 1:
                            flag = 0
                            p1 = ''
                            p2 = ''
                            d1 = ''
                            d2 = ''
                        sentence = re.sub("\d+ partner's persona:", '', sentence)
                        p1 += (sentence)
                    else:
                        chat = sentence.split('\t')
                        flag = 1
                        if len(chat) >= 2:
                            context1 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(d1))
                            persona1 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(p1))
                            if (len(context1) + len(persona1)) > 100:
                                if len(persona1) <= 100:
                                    t = context1[-(100 - len(persona1)):]
                                    context1 = persona1 + t
                                else:
                                    context1 = persona1[-100:]
                            elif (len(context1) + len(persona1)) < 100:
                                d_pad = list()
                                for i in range(100 - len(persona1) - len(context1)):
                                    d_pad.append(50257)
                                context1 = persona1 + context1 + d_pad
                            elif (len(context1) + len(persona1)) == 100:
                                context1 = persona1 + context1
                            r1 = chat[0]
                            r_t = ''
                            for s in range(len(r1)):
                                if (r1[s] in "1234567890") and (s <= 3):
                                    continue
                                else:
                                    r_t += r1[s]
                            r1 = r_t
                            response1 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(r1))
                            if len(response1) < 20:
                                r1_pad = r1
                                for i in range(20 - len(response1)):
                                    r1_pad += '<pad>'
                                response1 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(r1_pad))
                            elif len(response1) > 20:
                                response1 = response1[-20:]
                            traindata1 = context1 + response1
                            self.examples.append(traindata1)
                            d2 = d1 + chat[0]
                            d1 = d1 + chat[0] + chat[1]
                            context2 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(d2))
                            persona2 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(p2))
                            if (len(context2) + len(persona2)) > 100:
                                if len(persona2) <= 100:
                                    context2 = context2[-(100 - len(persona2)):]
                                    context2 = persona2 + context2
                                else:
                                    context2 = persona2[-100:]
                            elif (len(context2) + len(persona2)) < 100:
                                d_pad = list()
                                for i in range(100 - len(persona2) - len(context2)):
                                    d_pad.append(50257)
                                context2 = context2 + persona2 + d_pad
                            elif (len(context2) + len(persona2)) == 100:
                                context2 = context2 + persona2
                            r2 = chat[1]
                            r_t = ''
                            for s in range(len(r2)):
                                if (r2[s] in "1234567890") and (s <= 3):
                                    continue
                                else:
                                    r_t += r2[s]
                            r2 = r_t
                            response2 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(r2))
                            if len(response2) < 20:
                                r2_pad = r2
                                for i in range(20 - len(response2)):
                                    r2_pad += '<pad>'
                                response2 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(r2_pad))
                            elif len(response2) > 20:
                                response2 = response2[-20:]
                            traindata2 = context2 + response2
                            d2 = d2 + chat[1]
                            self.examples.append(traindata2)
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