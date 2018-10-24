# 同じ時間のアクションとしては、来客→回収の順番で処理

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt


def strong_function(base, ker):
    def strong_function_t(t):
        y = base
        for i in range(len(ker)):
            mu = ker[i][0]
            sigma = ker[i][1]
            y = y + (math.exp(-(t - mu) ** 2 / (2 * sigma))) / math.sqrt(2 * math.pi * sigma)
        return y

    return strong_function_t


def make_day_data(day):
    # パラメータの設定
    dt = 0.010  # 時間幅、単位は時間
    T = 16  # 時間の総量、7時～23時の16時間をイメージ
    F = 2  # フロア数

    S = np.zeros(F).astype('int')
    S[0] = 1000
    A = np.zeros((F, F)).astype('int')

    f = []
    f.append(strong_function(0.2, [[0.2, 0.0005], [0.6, 0.0005]]))
    f.append(strong_function(0.2, [[0.4, 0.0005], [0.8, 0.0005]]))

    df = pd.DataFrame(columns=(['day', 'time'] + ['S' + str(i) for i in range(F)]))
    df_move = pd.DataFrame(columns=(['day', 'time', 'in', 'out', 'passenger'] + ['S' + str(i) for i in range(F)]))

    for t in range(int(T / dt)):
        for f_in in range(F):
            for f_out in range(F):
                if f_in != f_out:
                    if f_in < 0:
                        A[f_in, f_out] = A[f_in, f_out] + f[0](t * dt / T) * np.random.poisson(40 * dt)
                    else:
                        # 付け焼刃的に直したので、スマートにしたい
                        A[f_in, f_out] = min(A[f_in, f_out] + f[f_in](t * dt / T) * np.sqrt(
                            S[f_in] - np.sum(A[f_in, :])) * np.random.poisson(20 * dt), (S[f_in] - np.sum(A[f_in, :])))

        for f_in in range(F):
            for f_out in range(F):
                if (f_in != f_out):
                    e_come = np.random.poisson(20 * dt)
                    if (e_come > 0):
                        if A[f_in, f_out] > 0:
                            df_move.loc[len(df_move) + 1] = (
                                        [day, dt * t] + [f_in, f_out, A[f_in, f_out]] + [S[i] for i in range(F)])
                            S[f_out] = S[f_out] + A[f_in, f_out]
                            S[f_in] = S[f_in] - A[f_in, f_out]
                            A[f_in, f_out] = 0

        df.loc[t] = ([day, dt * t] + [S[i] for i in range(F)])
    return df, df_move


def main():
    for i in range(100):
        df, df_move = make_day_data(i)
        if i == 0:
            df_all = df_move
        else:
            df_all = pd.concat([df_all, df_move])
        print(i, len(df_move))
    df_all.to_csv('df_all.csv', index=False)
    print(len(df_all))
    print(df_all.head())


if __name__ == '__main__':
    main()