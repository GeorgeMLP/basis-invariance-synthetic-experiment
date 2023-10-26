import numpy as np
import torch
from torch import Tensor


def orthogonalize(U: Tensor) -> Tensor:
    """
    Orthogonalize a set of linear independent vectors using Gram–Schmidt process.

    >>> U = torch.nn.functional.normalize(torch.randn(5, 3), dim=0)
    >>> U = orthogonalize(U)
    >>> torch.allclose(U.T @ U, torch.eye(3), atol=1e-06)
    True

    :param U: Tensor of shape [n, d], d <= n.
    :return: Tensor of shape [n, d].
    """
    Q, R = torch.linalg.qr(U)
    return Q


def find_complementary_space(U: Tensor, u_span: Tensor) -> Tensor:
    """
    Find the orthogonal complementary space of u_span in the linear space U.

    >>> U = Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    >>> u_span = Tensor([[0, 0], [1, 0], [0, 0], [0, 0], [0, 1]])
    >>> find_complementary_space(U, u_span)
    tensor([[1., 0.],
            [0., 0.],
            [0., 0.],
            [0., 1.],
            [0., 0.]])

    :param U: Tensor of shape [n, d].
    :param u_span: Tensor of shape [n, s], where s <= d.
    :return: Tensor of shape [n, d - s].
    """
    n, d = U.shape
    s = u_span.shape[1]
    u_base = u_span.clone()
    for j in range(d):
        i = u_base.shape[1]
        u_j = U[:, j].unsqueeze(dim=1)  # shape [n, 1]
        u_temp = torch.cat([u_base, u_j], dim=1)  # shape [n, d'] where i <= d' <= d
        if torch.linalg.matrix_rank(u_temp) == i + 1:  # u_temp are linear independent
            u_base = u_temp
        if u_base.shape[1] == d:
            break
    u_base = orthogonalize(u_base)
    u_perp = u_base[:, s:d]
    return u_perp


def unique_basis(U_i: Tensor) -> Tensor:
    """
    Eliminating basis ambiguity of the input eigenvectors.

    :param U_i: Tensor of shape [n, d]. Each column of U is an eigenvector.
    :return: Tensor of shape [n, d].
    """
    n, d = U_i.shape
    E = torch.eye(n)
    J = torch.ones(n)
    P = U_i @ U_i.T
    Pe = [torch.linalg.vector_norm(P[:, i]).round(decimals=14).item() for i in range(n)]
    Pe = [i for i in enumerate(Pe)]
    Pe.sort(key=lambda x: x[1])
    indices = [i[0] for i in Pe]
    lengths = [i[1] for i in Pe]
    _, counts = np.unique(lengths, return_counts=True)
    assert len(counts) >= d  # basis assumption 1
    X = torch.zeros([d, n])  # [x_1, ..., x_d]
    step = -1
    for i in range(1, d + 1):
        x = torch.zeros(n)
        for _ in range(counts[-i]):
            x += E[indices[step]]
            step -= 1
        X[i - 1] = x + 10 * J
    U_0 = torch.zeros([n, d])  # the unique basis
    u_span = torch.empty([n, 0])  # span(u_1, ..., u_{i-1})
    u_perp = U_i.clone()  # orthogonal complementary space
    for i in range(d):
        P_perp = u_perp @ u_perp.T
        u_i = P_perp @ X[i]
        assert torch.linalg.vector_norm(u_i) != 0  # basis assumption 2
        u_i = torch.nn.functional.normalize(u_i, dim=0)
        U_0[:, i] = u_i
        u_span = torch.cat([u_span, u_i.unsqueeze(dim=1)], dim=1)
        u_perp = find_complementary_space(U_i, u_span)
    return U_0


def random_orthonormal_matrix(n: int, d: int) -> Tensor:
    """
    Randomly generate an orthonormal matrix of shape [n, d].

    >>> U = random_orthonormal_matrix(5, 3)
    >>> I = torch.eye(3)
    >>> torch.allclose(U.T @ U, I, atol=1e-06)
    True

    :param n: The first dimension of the random orthonormal matrix.
    :param d: The second dimension of the random orthonormal matrix.
    :return: Random orthonormal matrix of shape [n, d].
    """
    A = torch.randn([n, n])
    _, U = torch.linalg.eigh(A)
    return U[:, :d]


def random_permutation_matrix(n: int) -> Tensor:
    """
    Generate a random permutation matrix.

    :param n: The order of the permutation matrix.
    :return: Tensor of shape [n, n].
    """
    P = torch.eye(n)
    sigma = torch.randperm(n)
    return P[sigma]
