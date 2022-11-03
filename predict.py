import joblib
import torch

import config
from model import NerModel
from dataset import EntityDataset

if __name__ == '__main__':

    meta_data = joblib.load("meta.bin")
    tag_encoder =meta_data['tag_encoder']
    pos_encoder = meta_data['pos_encoder']
    num_tags = len(list(tag_encoder.classes_))
    num_pos = len(pos_encoder.classes_)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NerModel(num_tags, num_pos)
    #model = model.load_state_dict(torch.load(config.MODEL_PATH))
    model = model.to(device)

    sentence = "Hello this egypt the country of ahmed "

    tokenized_sentence = config.TOKENZIER.encode(sentence)

    sentence = sentence.split()

    print(sentence)
    print(tokenized_sentence)

    data_set = EntityDataset(
        texts=[sentence],
        pos=[[0] * len(sentence)],
        tag=[[0] * len(sentence)]
    )

    with torch.no_grad():
        data = data_set[0]
        for k, v in data.items():
            data[k] = v.to(device).unsqueeze(0)

        tag, pos, _ = model(**data)

    print(
        tag_encoder.inverse_transform(
            tag.argmax(2).cpu().numpy().reshape(-1)
        )[:len(tokenized_sentence)]
    )
    print(
        pos_encoder.inverse_transform(
            pos.argmax(2).cpu().numpy().reshape(-1)
        )[:len(tokenized_sentence)]
    )




    