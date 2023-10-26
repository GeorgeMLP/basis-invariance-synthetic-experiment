from tqdm import tqdm
import torch
from torch import optim
from torch.utils.data import DataLoader, random_split
from utils import unique_basis
from model import DeepSets, DeepSetsSignNet
from dataset import EigenspaceClassification

N = 7
EPOCHS = 1000
NUM_TRIALS = 4
DATASET_SIZE = 2000
TRAIN_RATIO = 0.9
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
FACTOR = 0.5
PATIENCE = 25
USE_MAP = False
USE_RANDOM_SIGN = False
USE_SIGNNET = False
DEVICE = torch.device("cuda")


def train(model, device, loader, optimizer, criterion) -> float:
    model.train()
    epoch_loss = step = 0
    for step, batch in enumerate(loader):
        x, y = batch[0].to(device), batch[1].to(device)
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
    epoch_loss /= step + 1
    return epoch_loss


def validate(model, device, loader, criterion) -> float:
    model.eval()
    epoch_loss = step = 0
    for step, batch in enumerate(loader):
        x, y = batch[0].to(device), batch[1].to(device)
        with torch.no_grad():
            pred = model(x)
            loss = criterion(pred, y)
        epoch_loss += loss.detach().item()
    epoch_loss /= step + 1
    return epoch_loss


def eval(model, device, loader) -> float:
    model.eval()
    total = correct = 0
    for step, batch in enumerate(loader):
        x, y = batch[0].to(device), batch[1].to(device)
        with torch.no_grad():
            out = model(x)
        y_pred = out.max(dim=-1)[1]
        correct += torch.eq(y_pred, y).sum().item()
        total += y.shape[0]
    correct /= total
    return correct


test_accs = []
for _ in range(NUM_TRIALS):
    dataset = EigenspaceClassification(N, DATASET_SIZE, pre_transform=unique_basis) if USE_MAP \
        else EigenspaceClassification(N, DATASET_SIZE)
    train_size = int(DATASET_SIZE * TRAIN_RATIO)
    test_size = DATASET_SIZE - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_set, BATCH_SIZE, True)
    test_loader = DataLoader(test_set, BATCH_SIZE, True)
    if USE_SIGNNET:
        model = DeepSetsSignNet(N // 2, 2, True, 10, 10, readout="mean")
    else:
        model = DeepSets(N // 2, 2, True, 0.0, 10, 10, torch.nn.LeakyReLU(),
                         readout="mean", random_sign=USE_RANDOM_SIGN)
    model = model.to(DEVICE)
    # num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print("number of model parameters:", num_param)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=FACTOR, patience=PATIENCE)
    criterion = torch.nn.CrossEntropyLoss()
    try:
        for t in (pbar := tqdm(range(EPOCHS))):
            train_loss = train(model, DEVICE, train_loader, optimizer, criterion)
            val_loss = validate(model, DEVICE, test_loader, criterion)
            scheduler.step(val_loss)
            train_perf = eval(model, DEVICE, train_loader)
            test_perf = eval(model, DEVICE, test_loader)
            pbar.set_postfix(train_loss=train_loss, val_loss=val_loss, train_acc=train_perf, test_acc=test_perf)
    except KeyboardInterrupt:
        print('Exiting from training early because of KeyboardInterrupt')
    test_acc = eval(model, DEVICE, test_loader)
    test_accs.append(test_acc)
test_accs = torch.Tensor(test_accs)
print("Final result:")
print(f"Test acc: {test_accs.mean()} ± {test_accs.std()}")
