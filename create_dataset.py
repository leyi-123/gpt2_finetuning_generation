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
w_path = "E:/dialogue/personprofile/personachat/alter_file/w_test_both_original.txt"
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
                def change_data(text):
                    flag = 0
                    persona1 = ''
                    persona2 = ''
                    context1 = ''
                    context2 = ''
                    for sentence in text:
                        if "your persona:" in sentence:
                            if flag == 1:
                                flag = 0
                                persona1 = ''
                                persona2 = ''
                                context1 = ''
                                context2 = ''
                            sentence = re.sub("\d+ your persona: ", '', sentence)
                            persona2 = persona2 + ' ' + sentence
                        elif "partner's persona:" in sentence:
                            sentence = re.sub("\d+ partner's persona: ", '', sentence)
                            persona1 = persona1 + ' ' + sentence
                        else:
                            chat = sentence.split("\t")
                            if len(chat) >= 2:
                                flag = 1
                                r = ''
                                for i in range(len(chat[0])):
                                    if (i <= 3) and (chat[0][i] in '0123456789'):
                                        continue
                                    else:
                                        r += chat[0][i]
                                chat[0] = r
                                r = ''
                                for i in range(len(chat[1])):
                                    if (i <= 3) and (chat[1][i] in '0123456789'):
                                        continue
                                    else:
                                        r += chat[1][i]
                                chat[1] = r
                                response1 = chat[0] + ' <|endoftext|>'
                                response2 = chat[1] + ' <|endoftext|>'
                                context2 = context1 + ' ' + chat[0] + ' <|endoftext|>'
                                w_f = open(w_path, 'a', encoding="utf-8")
                                s1 = persona1 + "\t" + context1 + "\t" + response1 + "\n"
                                s2 = persona2 + "\t" + context2 + "\t" + response2 + "\n"
                                w_f.write(s1)
                                w_f.write(s2)
                                w_f.close()
                                context1 = context1 + ' ' + chat[0] + ' <|endoftext|>' + ' ' + chat[1] + ' <|endoftext|>'
                change_data(text)
                w_f = open(w_path, encoding="utf-8")
                w_text = w_f.read()
                for sentence in w_text.split("\n"):
                    if len(sentence) == 3:
                        persona = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence[0]))
                        context = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence[1]))
                        response = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence[2]))
                        if len(persona) < 50:
                            for i in range(50 - len(persona)):
                                persona.append(50257)
                        elif len(persona) > 50:
                            persona = persona[:50]
                        if len(context) < 50:
                            for i in range(50 - len(context)):
                                context.append(50257)
                        elif len(context) > 50:
                            context = context[-50:]
                        if len(response) < 20:
                            for i in range(20 - len(response)):
                                response.append(50257)
                        elif len(response) > 20:
                            response = response[:20]
                        t = persona + context + response
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
