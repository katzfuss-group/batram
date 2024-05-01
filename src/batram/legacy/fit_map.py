# flake8: noqa  ## lagacy file

import sys

import torch
from gpytorch.kernels import MaternKernel
from pyro.distributions import InverseGamma
from scipy.stats import norm, t
from torch.distributions import MultivariateNormal, Normal
from torch.distributions.studentT import StudentT
from torch.nn.parameter import Parameter


def nug_fun(i, theta, scales):
    return torch.exp(torch.log(scales[i]).mul(theta[1]).add(theta[0]))


def scaling_fun(k, theta):
    return torch.sqrt(torch.exp(k.mul(theta[2])))


def sigma_fun(i, theta, scales):
    return torch.exp(torch.log(scales[i]).mul(theta[4]).add(theta[3]))


def range_fun(theta):
    return torch.exp(theta[5])


def varscale_fun(i, theta, scales):
    return torch.exp(torch.log(scales[i]).mul(theta[7]).add(theta[6]))


def con_fun(i, theta, scales):
    return torch.exp(torch.log(scales[i]).mul(theta[9]).add(theta[8]))


def m_threshold(theta, mMax):
    below = scaling_fun(torch.arange(mMax).add(1), theta) < 0.01
    if below.sum().equal(torch.tensor(0)):
        m = torch.tensor(mMax)
    else:
        m = torch.argmax(below.type(torch.DoubleTensor))
    return torch.maximum(m, torch.tensor(1))


def kernel_fun(X1, theta, sigma, smooth, nuggetMean=None, X2=None):
    N = X1.shape[1]
    if X2 is None:
        X2 = X1
    if nuggetMean is None:
        nuggetMean = 1
    X1s = X1.mul(scaling_fun(torch.arange(1, N + 1).unsqueeze(0), theta))
    X2s = X2.mul(scaling_fun(torch.arange(1, N + 1).unsqueeze(0), theta))
    lin = X1s @ X2s.t()
    MaternObj = MaternKernel(smooth.item())
    MaternObj._set_lengthscale(1.0)
    lenScal = range_fun(theta) * smooth.mul(2).sqrt()
    nonlin = MaternObj.forward(X1s.div(lenScal), X2s.div(lenScal)).mul(sigma.pow(2))
    return (lin + nonlin).div(nuggetMean)


class TransportMap(torch.nn.Module):
    def __init__(self, thetaInit, linear=False, tuneParm=None):
        super().__init__()
        if tuneParm is None:
            self.nugMult = torch.tensor(4.0)
            self.smooth = torch.tensor(1.5)
        else:
            self.nugMult = tuneParm[0]
            self.smooth = tuneParm[1]
        self.theta = Parameter(thetaInit)
        self.linear = linear

    def forward(self, data, NNmax, mode, m=None, inds=None, scal=None):
        # theta as intermediate var
        if self.linear:
            theta = torch.cat((self.theta, torch.tensor([-float("inf"), 0.0, 0.0])))
        else:
            theta = self.theta
        # default opt parms
        n, N = data.shape
        if m is None:
            m = m_threshold(theta, NNmax.shape[1])
        if inds is None:
            inds = torch.arange(N)
        Nhat = inds.shape[0]
        if scal is None:
            scal = torch.div(torch.tensor(1), torch.arange(N).add(1))  # N
        NN = NNmax[:, :m]
        # init tmp vars
        K = torch.zeros(Nhat, n, n)
        G = torch.zeros(Nhat, n, n)
        loglik = torch.zeros(Nhat)
        # Prior vars
        nugMean = torch.relu(nug_fun(inds, theta, scal).sub(1e-5)).add(1e-5)  # Nhat,
        nugSd = nugMean.mul(self.nugMult)  # Nhat,
        alpha = nugMean.pow(2).div(nugSd.pow(2)).add(2)  # Nhat,
        beta = nugMean.mul(alpha.sub(1))  # Nhat,
        # nll
        for i in range(Nhat):
            if inds[i] == 0:
                G[i, :, :] = torch.eye(n)
            else:
                ncol = torch.minimum(inds[i], m)
                X = data[:, NN[inds[i], :ncol]]  # n X ncol
                K[i, :, :] = kernel_fun(
                    X, theta, sigma_fun(inds[i], theta, scal), self.smooth, nugMean[i]
                )  # n X n
                G[i, :, :] = K[i, :, :] + torch.eye(n)  # n X n
        try:
            GChol = torch.linalg.cholesky(G)
        except RuntimeError as inst:
            print(inst)
            if mode == "fit":
                sys.exit("chol failed")
            else:
                return torch.tensor(float("-inf"))
        yTilde = torch.linalg.solve_triangular(
            GChol, data[:, inds].t().unsqueeze(2), upper=False
        ).squeeze()  # Nhat X n
        alphaPost = alpha.add(n / 2)  # Nhat,
        betaPost = beta + yTilde.square().sum(dim=1).div(2)  # Nhat,
        if mode == "fit":
            # variable storage has been done through batch operations
            pass
        elif mode == "intlik":
            # integrated likelihood
            logdet = GChol.diagonal(dim1=-1, dim2=-2).log().sum(dim=1)  # nHat,
            loglik = (
                -logdet
                + alpha.mul(beta.log())
                - alphaPost.mul(betaPost.log())
                + alphaPost.lgamma()
                - alpha.lgamma()
            )  # nHat,
        else:
            # profile likelihood
            nuggetHat = betaPost.div(alphaPost.add(1))  # nHat
            fHat = (
                torch.triangular_solve(K, GChol, upper=False)[0]
                .bmm(yTilde.unsqueeze(2))
                .squeeze()
            )  # nHat X n
            uniNDist = Normal(loc=fHat, scale=nuggetHat.unsqueeze(1))
            mulNDist = MultivariateNormal(loc=torch.zeros(1, n), covariance_matrix=K)
            invGDist = InverseGamma(concentration=alpha, rate=beta)
            loglik = (
                uniNDist.log_prob(data[:, inds].t()).sum(dim=1)
                + mulNDist.log_prob(fHat)
                + invGDist.log_prob(nuggetHat)
            )
        if mode == "fit":
            tuneParm = torch.tensor([self.nugMult, self.smooth])
            return {
                "Chol": GChol,
                "yTilde": yTilde,
                "nugMean": nugMean,
                "alphaPost": alphaPost,
                "betaPost": betaPost,
                "scal": scal,
                "data": data,
                "NN": NN,
                "theta": theta,
                "tuneParm": tuneParm,
            }
        else:
            return loglik.sum().neg()


def fit_map_mini(
    data,
    NNmax,
    scal=None,
    linear=False,
    maxEpoch=10,
    batsz=128,
    tuneParm=None,
    lr=1e-5,
    dataTest=None,
    NNmaxTest=None,
    scalTest=None,
    track_loss=False,
    **kwargs,
):
    # default initial values
    thetaInit = torch.tensor(
        [data[:, 0].square().mean().log(), 0.2, -1.0, 0.0, 0.0, -1.0]
    )
    if linear:
        thetaInit = thetaInit[0:3]

    print(f"inital theta: {thetaInit}")
    transportMap = TransportMap(thetaInit, linear=linear, tuneParm=tuneParm)
    # optimizer = torch.optim.SGD(transportMap.parameters(), lr=lr, momentum=0.9)
    optimizer = torch.optim.Adam(transportMap.parameters(), lr=lr)
    if dataTest is None:
        dataTest = data[:, : min(data.shape[1], 5000)]
        NNmaxTest = NNmax[: min(data.shape[1], 5000), :]
        if scal is not None:
            scalTest = scal[: min(data.shape[1], 5000)]
    # optimizer = torch.optim.Adam(transportMap.parameters(), lr=lr)
    epochIter = int(data.shape[1] / batsz)
    losses = []
    for i in range(maxEpoch):
        for _ in range(epochIter):
            inds = torch.multinomial(torch.ones(data.shape[1]), batsz)
            optimizer.zero_grad()
            try:
                loss = transportMap(
                    data, NNmax, "intlik", inds=inds, scal=scal, **kwargs
                )
                loss.backward()
            except RuntimeError as inst:
                print("Warning: the current optimization iteration failed")
                print(inst)
                continue
            optimizer.step()
        print("Epoch ", i + 1, "\n")
        for name, parm in transportMap.named_parameters():
            print(f"{name}: {parm.data}")
        if i == 0:
            with torch.no_grad():
                scrPrev = transportMap(dataTest, NNmaxTest, "intlik", scal=scalTest)
                print("Current test score is ", scrPrev, "\n")
        else:
            with torch.no_grad():
                scrCurr = transportMap(dataTest, NNmaxTest, "intlik", scal=scalTest)
                print("Current test score is ", scrCurr, "\n")
            if scrCurr > scrPrev:
                losses.append(scrCurr)
                break
            scrPrev = scrCurr
        losses.append(scrPrev)
    with torch.no_grad():
        tmval = transportMap(data, NNmax, "fit", scal=scal, **kwargs)
        if track_loss:
            return tmval, losses
        return tmval


def cond_samp(fit, mode, obs=None, xFix=torch.tensor([]), indLast=None):
    data = fit["data"]
    NN = fit["NN"]
    theta = fit["theta"]
    scal = fit["scal"]
    nugMult = fit["tuneParm"][0]
    smooth = fit["tuneParm"][1]
    nugMean = fit["nugMean"]
    chol = fit["Chol"]
    yTilde = fit["yTilde"]
    betaPost = fit["betaPost"]
    alphaPost = fit["alphaPost"]
    n, N = data.shape
    m = NN.shape[1]
    if indLast is None:
        indLast = N
    # loop over variables/locations
    xNew = scr = torch.cat((xFix, torch.zeros(N - xFix.size(0))))
    for i in range(xFix.size(0), indLast):
        # predictive distribution for current sample
        if i == 0:
            cStar = torch.zeros(n)
            prVar = torch.tensor(0.0)
        else:
            ncol = min(i, m)
            X = data[:, NN[i, :ncol]]
            if mode in ["score", "trans", "scorepm"]:
                XPred = obs[NN[i, :ncol]].unsqueeze(0)
            else:
                XPred = xNew[NN[i, :ncol]].unsqueeze(0)
            cStar = kernel_fun(
                XPred, theta, sigma_fun(i, theta, scal), smooth, nugMean[i], X
            ).squeeze()
            prVar = kernel_fun(
                XPred, theta, sigma_fun(i, theta, scal), smooth, nugMean[i]
            ).squeeze()
        cChol = torch.linalg.solve_triangular(
            chol[i, :, :], cStar.unsqueeze(1), upper=False
        ).squeeze()
        meanPred = yTilde[i, :].mul(cChol).sum()
        varPredNoNug = prVar - cChol.square().sum()
        # evaluate score or sample
        if mode == "score":
            initVar = betaPost[i] / alphaPost[i] * (1 + varPredNoNug)
            STDist = StudentT(2 * alphaPost[i])
            scr[i] = (
                STDist.log_prob((obs[i] - meanPred) / initVar.sqrt())
                - 0.5 * initVar.log()
            )
        elif mode == "scorepm":
            nugget = betaPost[i] / alphaPost[i].sub(1)
            uniNDist = Normal(loc=meanPred, scale=nugget.sqrt())
            scr[i] = uniNDist.log_prob(obs[i])
        elif mode == "fx":
            xNew[i] = meanPred
        elif mode == "freq":
            nugget = betaPost[i] / alphaPost[i].add(1)
            uniNDist = Normal(loc=meanPred, scale=nugget.sqrt())
            xNew[i] = uniNDist.sample()
        elif mode == "bayes":
            invGDist = InverseGamma(concentration=alphaPost[i], rate=betaPost[i])
            nugget = invGDist.sample()
            uniNDist = Normal(loc=meanPred, scale=nugget.mul(1 + varPredNoNug).sqrt())
            xNew[i] = uniNDist.sample()
        elif mode == "trans":
            initVar = betaPost[i] / alphaPost[i] * (1 + varPredNoNug)
            xStand = (obs[i] - meanPred) / initVar.sqrt()
            xNew[i] = norm(0, 1).ppf(t(2 * alphaPost[i]).cdf(xStand))
        elif mode == "invtrans":
            initVar = betaPost[i] / alphaPost[i] * (1 + varPredNoNug)
            xNew[i] = (
                meanPred
                + t(2 * alphaPost[i]).ppf(norm(0, 1).cdf(obs[i])) * initVar.sqrt()
            )
    if mode in ["score", "scorepm"]:
        return scr[xFix.size(0) :].sum()
    else:
        return xNew


# locsOdr: each row is one location
# NN: each row represents one location
def compute_scal(locsOdr, NN):
    locsOdr = torch.as_tensor(locsOdr)
    N = locsOdr.shape[0]
    scal = (locsOdr[1:, :] - locsOdr[NN[1:, 0], :]).square().sum(1).sqrt()
    scal = torch.cat((scal[0].square().div(scal[4]).unsqueeze(0), scal))
    scal = scal.div(scal[0])
    return scal
