import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt


def search_redist_2(df, is_1_add=False, is_2_add=False):

    coocked_df = pd.DataFrame(
        columns=['0_', '1_', 'lower0', 'lower1', 'upper0', 'upper1', 'fl0', 'fl1', 'fu0', 'fu1', 'dif_is_l', 'dif_is_u', 'redist', 'day']
    )

    df.day = df['day'].apply(lambda x: pd.to_datetime(x))  # индекс == дата
    df = df.set_index('day')

    for i in range(0, df.shape[1]):
        df.iloc[:, i] = df.iloc[:, i] + ((df.iloc[:, i].min() + 1) / (1000000 * df.iloc[:, i].max() + 1000000))  # избавляемся от 0

    if not is_1_add:

        decompose_result = seasonal_decompose(df[['0_']], model="multiplicative")  # считаем мультипликативную сезонку
        seasonal = decompose_result.seasonal
        df['msd'] = df['0_'] / seasonal

        decompose_result = seasonal_decompose(df[['0_']], model="additive")  # считаем аддитивную сезонку
        seasonal = decompose_result.seasonal
        df['asd'] = df['0_'] - seasonal

        dftest1 = adfuller(df['msd'])
        dftest2 = adfuller(df['asd'])
        if dftest1[1] >= dftest2[1]:  # сравниваем два десезонированных ряда и берем тот, что стабильнее
            df['0_'] = df['msd']
        else:
            df['0_'] = df['asd']

    if not is_2_add:

        decompose_result = seasonal_decompose(df[['1_']], model="multiplicative")
        seasonal = decompose_result.seasonal
        df['msd'] = df['1_'] / seasonal

        decompose_result = seasonal_decompose(df[['1_']], model="additive")
        seasonal = decompose_result.seasonal
        df['asd'] = df['1_'] - seasonal

        dftest1 = adfuller(df['msd'])
        dftest2 = adfuller(df['asd'])
        if dftest1[1] >= dftest2[1]:
            df['1_'] = df['msd']
        else:
            df['1_'] = df['asd']

    df['0_'] = df['0_'].rolling(3, center=True, min_periods=1).median()  # медианный фильтр
    df['1_'] = df['1_'].rolling(3, center=True, min_periods=1).median()

    df['e0_'] = df['0_'].rolling(7, min_periods=1, closed='left').mean()  # математическое ожидание
    df['e1_'] = df['1_'].rolling(7, min_periods=1, closed='left').mean()

    df = df.iloc[1:]

    df['d0_'] = df['0_'].rolling(14, min_periods=1, closed='left').std()  # стандартное отклонение
    df['d1_'] = df['1_'].rolling(14, min_periods=1, closed='left').std()

    df = df.iloc[2:]

    df['upper0'] = df['e0_'] + 3 * df['d0_']  # создание доверительного интервала
    df['lower0'] = df['e0_'] - 3 * df['d0_']
    df['upper1'] = df['e1_'] + 3 * df['d1_']
    df['lower1'] = df['e1_'] - 3 * df['d1_']

    df = df[['0_', '1_', 'e0_', 'e1_', 'lower0', 'lower1', 'upper0', 'upper1']].copy()

    df['dif0'] = abs(df['e0_'] - df['0_'])  # подсчет разницы между текущим значением и мат ожиданием
    df['dif1'] = abs(df['e1_'] - df['1_'])

    df['l0'] = ((df['0_'] < df['lower0']) & (df['dif0'] / df['e0_'] >= 0.05)) * 1  # проверка условия на вылет из доверительного интервала(выход за предел и изменение хотя-бы на 5%)
    df['shiftl0'] = df['l0'].shift(periods=1, fill_value=0)

    df['l1'] = ((df['1_'] < df['lower1']) & (df['dif1'] / df['e1_'] >= 0.05)) * 1
    df['shiftl1'] = df['l1'].shift(periods=1, fill_value=0)

    df['u0'] = ((df['0_'] > df['upper0']) & (df['dif0'] / df['e0_'] >= 0.05)) * 1
    df['shiftu0'] = df['u0'].shift(periods=1, fill_value=0)

    df['u1'] = ((df['1_'] > df['upper1']) & (df['dif1'] / df['e1_'] >= 0.05)) * 1
    df['shiftu1'] = df['u1'].shift(periods=1, fill_value=0)

    df = df.iloc[1:]

    df['fl0'] = ((df['l0'] != 0) & (df['shiftl0'] == 0)) * 1
    df['fl1'] = ((df['l1'] != 0) & (df['shiftl1'] == 0)) * 1
    df['fu0'] = ((df['u0'] != 0) & (df['shiftu0'] == 0)) * 1
    df['fu1'] = ((df['u1'] != 0) & (df['shiftu1'] == 0)) * 1

    df = df.reset_index()

    df['is_l'] = df['fl0'] + df['fl1']  # был ли вылет вниз
    df['is_u'] = df['fu0'] + df['fu1']  # был ли вылет вверх
    df['dif_is_l'] = df['dif0'] * df['fl0'] + df['dif1'] * df['fl1']  # насколько вылетело вниз
    df['dif_is_u'] = df['dif0'] * df['fu0'] + df['dif1'] * df['fu1']  # насколько вылетело вверх
    df['is_same_dif'] = ((abs(df['dif_is_l'] - df['dif_is_u']) <= (df['dif_is_l'] / 2)) & (abs(df['dif_is_l'] - df['dif_is_u']) <= (df['dif_is_u'] / 2))) * 1  # одного ли порядка вылеты

    df['redist'] = ((df['is_same_dif'] != 0) & (df['is_l'] != 0) & (df['is_u'] != 0) & (df['dif_is_u'] >= 10) & (df['dif_is_l'] >= 10)) * 1  # проверка на все условия для перераспределения

    df['2_'] = 0
    df['lower2'] = 0
    df['upper2'] = 0
    df['fl2'] = 0
    df['fu2'] = 0

    df = df[['0_', '1_', 'lower0', 'lower1', 'upper0', 'upper1', 'fl0', 'fl1', 'fu0', 'fu1', 'dif_is_l', 'dif_is_u', 'redist', 'day']].copy()

    df = df.iloc[16:]

    coocked_df = pd.concat([coocked_df, df], ignore_index=True)
    return coocked_df


def search_redist_3(df, is_1_add=False, is_2_add=False, is_3_add=False):

    coocked_df = pd.DataFrame(
        columns=['0_', '1_', '2_', 'lower0', 'lower1', 'lower2', 'upper0', 'upper1', 'upper2', 'fl0', 'fl1', 'fl2', 'fu0', 'fu1', 'fu2', 'dif_is_l', 'dif_is_u', 'redist', 'day']
    )

    df.day = df['day'].apply(lambda x: pd.to_datetime(x))  # индекс == дата
    df = df.set_index('day')

    for i in range(0, df.shape[1]):
        df.iloc[:, i] = df.iloc[:, i] + ((df.iloc[:, i].min() + 1) / (1000000 * df.iloc[:, i].max() + 1000000))  # избавляемся от 0

    if not is_1_add:

        decompose_result = seasonal_decompose(df[['0_']], model="multiplicative")
        seasonal = decompose_result.seasonal
        df['msd'] = df['0_'] / seasonal

        decompose_result = seasonal_decompose(df[['0_']], model="additive")
        seasonal = decompose_result.seasonal
        df['asd'] = df['0_'] - seasonal

        dftest1 = adfuller(df['msd'])
        dftest2 = adfuller(df['asd'])
        if dftest1[1] >= dftest2[1]:
            df['0_'] = df['msd']
        else:
            df['0_'] = df['asd']

    if not is_2_add:

        decompose_result = seasonal_decompose(df[['1_']], model="multiplicative")
        seasonal = decompose_result.seasonal
        df['msd'] = df['1_'] / seasonal

        decompose_result = seasonal_decompose(df[['1_']], model="additive")
        seasonal = decompose_result.seasonal
        df['asd'] = df['1_'] - seasonal

        dftest1 = adfuller(df['msd'])
        dftest2 = adfuller(df['asd'])
        if dftest1[1] >= dftest2[1]:
            df['1_'] = df['msd']
        else:
            df['1_'] = df['asd']

    if not is_3_add:

        decompose_result = seasonal_decompose(df[['2_']], model="multiplicative")
        seasonal = decompose_result.seasonal
        df['msd'] = df['2_'] / seasonal

        decompose_result = seasonal_decompose(df[['2_']], model="additive")
        seasonal = decompose_result.seasonal
        df['asd'] = df['2_'] - seasonal

        dftest1 = adfuller(df['msd'])
        dftest2 = adfuller(df['asd'])
        if dftest1[1] >= dftest2[1]:
            df['2_'] = df['msd']
        else:
            df['2_'] = df['asd']

    df['0_'] = df['0_'].rolling(3, center=True, min_periods=1).median()
    df['1_'] = df['1_'].rolling(3, center=True, min_periods=1).median()
    df['2_'] = df['2_'].rolling(3, center=True, min_periods=1).median()

    df['e0_'] = df['0_'].rolling(7, min_periods=1, closed='left').mean()
    df['e1_'] = df['1_'].rolling(7, min_periods=1, closed='left').mean()
    df['e2_'] = df['2_'].rolling(7, min_periods=1, closed='left').mean()

    df = df.iloc[1:]

    df['d0_'] = df['0_'].rolling(14, min_periods=1, closed='left').std()
    df['d1_'] = df['1_'].rolling(14, min_periods=1, closed='left').std()
    df['d2_'] = df['2_'].rolling(14, min_periods=1, closed='left').std()

    df = df.iloc[2:]

    df['upper0'] = df['e0_'] + 3 * df['d0_']
    df['lower0'] = df['e0_'] - 3 * df['d0_']
    df['upper1'] = df['e1_'] + 3 * df['d1_']
    df['lower1'] = df['e1_'] - 3 * df['d1_']
    df['upper2'] = df['e2_'] + 3 * df['d2_']
    df['lower2'] = df['e2_'] - 3 * df['d2_']

    df = df[['0_', '1_', '2_', 'e0_', 'e1_', 'e2_', 'lower0', 'lower1', 'lower2', 'upper0', 'upper1', 'upper2']].copy()

    df['dif0'] = abs(df['e0_'] - df['0_'])
    df['dif1'] = abs(df['e1_'] - df['1_'])
    df['dif2'] = abs(df['e2_'] - df['2_'])

    df['l0'] = ((df['0_'] < df['lower0']) & (df['dif0'] / df['e0_'] >= 0.05)) * 1
    df['shiftl0'] = df['l0'].shift(periods=1, fill_value=0)
    df['l1'] = ((df['1_'] < df['lower1']) & (df['dif1'] / df['e1_'] >= 0.05)) * 1
    df['shiftl1'] = df['l1'].shift(periods=1, fill_value=0)
    df['l2'] = ((df['2_'] < df['lower2']) & (df['dif2'] / df['e2_'] >= 0.05)) * 1
    df['shiftl2'] = df['l2'].shift(periods=1, fill_value=0)

    df['u0'] = ((df['0_'] > df['upper0']) & (df['dif0'] / df['e0_'] >= 0.05)) * 1
    df['shiftu0'] = df['u0'].shift(periods=1, fill_value=0)
    df['u1'] = ((df['1_'] > df['upper1']) & (df['dif1'] / df['e1_'] >= 0.05)) * 1
    df['shiftu1'] = df['u1'].shift(periods=1, fill_value=0)
    df['u2'] = ((df['2_'] > df['upper2']) & (df['dif2'] / df['e2_'] >= 0.05)) * 1
    df['shiftu2'] = df['u2'].shift(periods=1, fill_value=0)

    df = df.iloc[1:]

    df['fl0'] = ((df['l0'] != 0) & (df['shiftl0'] == 0)) * 1
    df['fl1'] = ((df['l1'] != 0) & (df['shiftl1'] == 0)) * 1
    df['fl2'] = ((df['l2'] != 0) & (df['shiftl2'] == 0)) * 1
    df['fu0'] = ((df['u0'] != 0) & (df['shiftu0'] == 0)) * 1
    df['fu1'] = ((df['u1'] != 0) & (df['shiftu1'] == 0)) * 1
    df['fu2'] = ((df['u2'] != 0) & (df['shiftu2'] == 0)) * 1

    df = df.reset_index()

    df['is_l'] = df['fl0'] + df['fl1'] + df['fl2']
    df['is_u'] = df['fu0'] + df['fu1'] + df['fu2']
    df['dif_is_l'] = df['dif0'] * df['fl0'] + df['dif1'] * df['fl1'] + df['dif2'] * df['fl2']
    df['dif_is_u'] = df['dif0'] * df['fu0'] + df['dif1'] * df['fu1'] + df['dif2'] * df['fu2']
    df['is_same_dif'] = ((abs(df['dif_is_l'] - df['dif_is_u']) <= (df['dif_is_l'] / 2)) & (abs(df['dif_is_l'] - df['dif_is_u']) <= (df['dif_is_u'] / 2))) * 1

    df['redist'] = ((df['is_same_dif'] != 0) & (df['is_l'] != 0) & (df['is_u'] != 0) & (df['dif_is_u'] >= 10) & (df['dif_is_l'] >= 10)) * 1  # проверка на все условия для перераспределения

    df = df[['0_', '1_', '2_', 'lower0', 'lower1', 'lower2', 'upper0', 'upper1', 'upper2', 'fl0', 'fl1', 'fl2', 'fu0', 'fu1', 'fu2', 'dif_is_l', 'dif_is_u', 'redist', 'day']].copy()

    df = df.iloc[16:]

    coocked_df = pd.concat([coocked_df, df], ignore_index=True)
    return coocked_df


def main():
    df = pd.read_csv('1.csv')
    max_value = max(df['0_'].max(), df['1_'].max())
    if df.shape[0] > 0:
        coocked_df = search_redist_2(df)
        if coocked_df[coocked_df['redist'] == 1].shape[0] > 0:
            ans = coocked_df[coocked_df['redist'] == 1].copy()
            plt.figure(figsize=(16, 12), dpi=80)
            plt.plot(coocked_df['day'], coocked_df['0_'], coocked_df['day'], coocked_df['1_'])
            plt.vlines(x=ans['day'], ymin=0, ymax=1.1*max_value, colors='purple', ls='dotted', lw=2,
                       label='vline_multiple - full height')
            plt.show()

    # Для трех значений признака
    # df = pd.read_csv('2.csv')
    # max_value = max(df['0_'].max(), df['1_'].max(), df['2_'].max())
    # if df.shape[0] > 0:
    #     coocked_df = search_redist_3(df)
    #     if coocked_df[coocked_df['redist'] == 1].shape[0] > 0:
    #         ans = coocked_df[coocked_df['redist'] == 1].copy()
    #         plt.figure(figsize=(16, 12), dpi=80)
    #         plt.plot(coocked_df['day'], coocked_df['0_'], coocked_df['day'], coocked_df['1_'], coocked_df['day'], coocked_df['2_'])
    #         plt.vlines(x=ans['day'], ymin=0, ymax=1.1*max_value, colors='purple', ls='dotted', lw=2,
    #                    label='vline_multiple - full height')
    #         plt.show()


if __name__ == "__main__":
    main()
