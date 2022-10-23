import time
import torch


def train(model, optimizer, criterion, dataloader, epoch=-1):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(batch['text_padded'])#, batch['lengths'])
        loss = criterion(predicted_label, batch['labels'])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == batch['labels']).sum().item()
        total_count += batch['labels'].size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc/total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()


def evaluate(model, criterion, dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            predicted_label = model(batch['text_padded'])#, batch['lengths'])
            # loss = criterion(predicted_label, batch['labels'])
            total_acc += (predicted_label.argmax(1) == batch['labels']).sum().item()
            total_count += batch['labels'].size(0)
    return total_acc/total_count
