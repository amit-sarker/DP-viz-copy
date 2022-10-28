import numpy as np
from ektelo import workload, matrix
from hdmm import templates


def get_domain(W):
    if isinstance(W, workload.Kronecker):
        return tuple(Q.shape[1] for Q in W.matrices)
    elif isinstance(W, workload.Weighted):
        return get_domain(W.base)
    elif isinstance(W, workload.VStack):
        return get_domain(W.matrices[0])
    else:
        return W.shape[1]


class HDMM:
    def __init__(self, W, x, eps, prng):
        self.domain = get_domain(W)
        self.W = W
        self.x = x
        self.eps = eps
        self.prng = prng

    def optimize(self, restarts=100):
        W = self.W
        if type(self.domain) is tuple:  # kron or union kron workload
            ns = self.domain
            krons = []
            for n, mat in zip(ns, W.matrices):
                if isinstance(mat, matrix.Ones):  # if mat is a Total() matrix
                    krons.append(templates.Total(n))
                else:
                    if n < 16:
                        p = 1
                    else:
                        p = n // 16
                    krons.append(templates.PIdentity(p, n))
            kron = templates.Kronecker(krons)
            kron._params = kron.strategy()
            optk, lossk = kron.restart_optimize(W, restarts)

            # multiplicative factor puts losses on same scale
            self.strategy = optk

        else:
            n = self.domain
            pid = templates.PIdentity(max(1, n // 6), n)
            optp, loss = pid.restart_optimize(W, restarts)
            self.strategy = optp

    def run(self):
        A = self.strategy
        A1 = A.pinv()
        delta = self.strategy.sensitivity()
        noise = self.prng.laplace(loc=0.0, scale=delta / self.eps, size=A.shape[0])
        self.ans = A.dot(self.x) + noise
        self.xest = A1.dot(self.ans)
        return self.xest
