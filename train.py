import pandas as pd
import numpy as np

import joblib
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import config
import dataset
from model import NerModel
import engine


def process_data(data_path):
    # read data
    data = pd.read_csv(data_path, encoding='latin-1')

    # fill null values with last non-null value
    data['Sentence #'].fillna('ffill', inplace=True)

    tag_encoder = LabelEncoder()
    pos_encoder = LabelEncoder()
    # encoder pos and tags
    data['POS'] = pos_encoder.fit_transform(data['POS'])
    data['Tag'] = tag_encoder.fit_transform(data['Tag'])

    # group words by sentence column
    sentence = data.groupby('Sentence #')['Word'].apply(list).values
    pos = data.groupby('Sentence #')['POS'].apply(list).values
    tag = data.groupby('Sentence #')['Tag'].apply(list).values

    return sentence, pos, tag, tag_encoder, pos_encoder


if __name__ == "__main__":
    sentence, pos, tag, tag_ecoder, pos_encoder = process_data(config.TRAINING_FILE)

    meta_data = {
        'pos_encoder': pos_encoder,
        'tag_encoder': tag_ecoder
    }

    joblib.dump(meta_data, 'meta.bin')

    num_tags = len(tag_ecoder.classes_)
    num_pos = len(pos_encoder.classes_)

    (train_sentence,
     test_sentence,
     train_pos,
     test_pos,
     train_tag,
     test_tag) = train_test_split(sentence, pos, tag, test_size=0.10, random_state=42)

    train_dataset = dataset.EntityDataset(train_sentence, train_pos, train_tag)

    test_dataset = dataset.EntityDataset(test_sentence, test_pos, test_tag)

    train_loader = DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NerModel(num_tags, num_pos)
    model = model.to(device)

    num_train_Steps = int(len(train_dataset) / config.EPOCHS * config.TRAIN_BATCH_SIZE)
    optimizer = AdamW(model.parameters(), lr=config.LEARING_RATE)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=num_train_Steps)
    best_loss = 0

    for epoch in range(config.EPOCHS):
        print(f"Epoch {epoch + 1 }/{config.EPOCHS} :")
        traning_loss = engine.training_fn(train_loader, model, optimizer=optimizer,
                                        device=device, schduler=scheduler)

        validation_loss, tag_accuracy, pos_accuracy = engine.eval_fn(test_loader, model, device)

        print(f"Traning Loss : {traning_loss}")
        print(f"Validation Loss : {validation_loss}")
        print(f"Validation Tag Accuracy : {tag_accuracy}")
        print(f"Validation Pos Accuracy : {pos_accuracy}")

        if best_loss < traning_loss:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_loss = traning_loss

