import torch
from tqdm import tqdm


def training_fn(train_loader, model, optimizer, device, schduler):
    # put the model into training mode
    model.train()

    final_loss = 0

    for data in tqdm(train_loader, total=len(train_loader)):
        for k, v in data.items():
            # assine every data in data dictionary into device
            data[k] = v.to(device)

        # perfrom forward propagation
        _, _, loss = model(**data)

        # perfrom backpropagation
        loss.backward()

        optimizer.step()
        schduler.step()

        final_loss += loss

        optimizer.zero_grad()

    return final_loss / len(train_loader)


def eval_fn(data_loader, model, device):
    # put the model into evalution mode
    model.eval()
    final_loss = 0
    correct_tag = 0
    correct_pos = 0

    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)

        tag, pos, loss = model(**data)
        # shape of tag is 128 * 8 * 17
        # shape of pos is 128 * 8 * 42
        # sequence len * batch size * num_tag or num_pos

        tag_predictions = tag.argmax(2)
        pos_predictions = pos.argmax(2)

        target_tag = data['target_tag']
        target_pos = data['target_pos']
        # print(pos.shape)
        # print(tag_predictions.shape)
        correct_tag += torch.sum(tag_predictions == target_tag).item()
        correct_pos += torch.sum(pos_predictions == target_pos).item()

        final_loss += loss.item()
    return final_loss / len(data_loader), correct_tag / len(data_loader), correct_pos / len(data_loader)
