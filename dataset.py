import torch
from torch.utils.data import DataLoader, Dataset
import config


class EntityDataset(Dataset):

    def __init__(self, texts, pos, tag):

        # texts is list of lists
        # [[hi, my, name, is, ahmed],[hi, this, me"]]
        # tags : [[1,2,3,4],[1,3,4]]
        self.texts = texts
        self.pos = pos
        self.tag = tag

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        pos = self.pos[item]
        tag = self.tag[item]

        ids = []
        target_pos = []
        target_tag = []

        for i, s in enumerate(text):
            inputs = config.TOKENZIER.encode(
                s,
                # we don't need to add special tokens such as CLS ,
                add_special_tokens=False
            )
            # tokenizer could get a word like gonna, it will tokenize it into going to
            # we should add these word after tokenizing it in ids
            input_len = len(inputs)
            ids.extend(inputs)
            # gonna is noung then going and to should be noun
            target_pos.extend([pos[i]] * input_len)
            target_tag.extend([tag[i]] * input_len)

        # '-2' because we need to add special tokens such as CLS
        ids = ids[:config.MAX_LEN - 2]
        target_pos = target_pos[:config.MAX_LEN - 2]
        target_tag = target_tag[:config.MAX_LEN - 2]

        # 101 is CLS token, 102 is SEP token
        ids = [101] + ids + [102]
        target_pos = [0] + target_pos + [0]
        target_tag = [0] + target_tag + [0]

        # we create the mask to ignore the padded elements in the sequences.
        attention_mask = [1] * len(ids)

        token_type_ids = [0] * len(ids)

        padding_len = config.MAX_LEN - len(ids)
        # padding the input if the input is smaller than max_len
        ids = ids + ([0] * padding_len)
        target_tag = target_tag + ([0] * padding_len)
        target_pos = target_pos + ([0] * padding_len)
        attention_mask = attention_mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "target_pos": torch.tensor(target_pos, dtype=torch.long),
            "target_tag": torch.tensor(target_tag, dtype=torch.long),
            "mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_types_ids": torch.tensor(token_type_ids, dtype=torch.long)
        }








