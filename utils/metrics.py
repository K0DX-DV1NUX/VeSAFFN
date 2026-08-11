import numpy as np

EPS = 1e-8


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def RSE(pred, true):
    num = np.sqrt(np.sum((true - pred) ** 2))
    den = np.sqrt(np.sum((true - true.mean()) ** 2)) + EPS
    return num / den


def CORR(pred, true):
    pred_mean = pred.mean(0)
    true_mean = true.mean(0)

    num = ((pred - pred_mean) * (true - true_mean)).sum(0)

    den = np.sqrt(
        ((pred - pred_mean) ** 2).sum(0) *
        ((true - true_mean) ** 2).sum(0)
    ) + EPS

    return (num / den).mean()


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / (true + EPS)))


def MSPE(pred, true):
    return np.mean(((pred - true) / (true + EPS)) ** 2)


def SMAPE(pred, true):
    return np.mean(
        2 * np.abs(pred - true) /
        (np.abs(pred) + np.abs(true) + EPS)
    )


def R_squared(pred, true):
    ss_res = np.sum((pred - true) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2) + EPS
    return 1 - ss_res / ss_tot


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = np.sqrt(mse)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)
    r2 = R_squared(pred, true)

    return mae, mse, rmse, mape, mspe, rse, corr, r2