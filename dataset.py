import random
import torch
from torch import Tensor
from torch.utils.data import Dataset
from typing import Tuple
from utils import random_orthonormal_matrix, random_permutation_matrix


class EigenspaceClassification(Dataset):
    def __init__(self, n: int = 7, dataset_size: int = 2000, transform=None, target_transform=None, pre_transform=None):
        dataset = []
        U_1 = random_orthonormal_matrix(n, n // 2)
        for _ in range(dataset_size // 2):
            P = random_permutation_matrix(n)
            Q = random_orthonormal_matrix(n // 2, n // 2)
            U = P @ U_1 @ Q
            dataset.append((U, torch.tensor(0)))
        U_2 = random_orthonormal_matrix(n, n // 2)
        for _ in range(dataset_size // 2):
            P = random_permutation_matrix(n)
            Q = random_orthonormal_matrix(n // 2, n // 2)
            U = P @ U_2 @ Q
            dataset.append((U, torch.tensor(1)))
        random.shuffle(dataset)
        if pre_transform is not None:
            dataset = [(pre_transform(x), y) for x, y in dataset]
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        x, y = self.dataset[idx]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y
