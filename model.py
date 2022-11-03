import torch
import torch.nn as nn
import transformers

import config


def loss_fn(output, target, mask, num_labels):
    lfn = nn.CrossEntropyLoss()
    # we don't need to compute the loss to the whole sentence
    # we need to compute the loss where there is not any padding
    active_loss = mask.view(-1) == 1
    active_logtis = output.view(-1, num_labels)
    # if active loss is zero or false then return the ignore index of lfn
    active_labels = torch.where(
        active_loss,
        target.view(-1),
        torch.tensor(lfn.ignore_index).type_as(target)
    )
    loss = lfn(active_logtis, active_labels)
    return loss


class NerModel(nn.Module):

    def __init__(self, num_tag, num_pos):
        super(NerModel, self).__init__()

        self.num_tag = num_tag
        self.num_pos = num_pos
        # load bert model
        self.bert = transformers.BertModel.from_pretrained(config.MODEL_BASE_PATH, return_dict=False)

        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.5)

        self.out_tag = nn.Linear(768, self.num_tag)
        self.out_pos = nn.Linear(768, self.num_pos)

    def forward(self, ids, mask, token_types_ids, target_pos, target_tag):
        model_output,  _ = self.bert(ids, mask, token_types_ids)

        tag_net = self.drop1(model_output)
        pos_net = self.drop2(model_output)

        tag = self.out_tag(tag_net)
        pos = self.out_pos(pos_net)

        loss_tag = loss_fn(tag, target_tag, mask, self.num_tag)

        loss_pos = loss_fn(pos, target_pos, mask, self.num_pos)

        loss = (loss_tag + loss_pos) / 2

        return tag, pos, loss

