# 导入所需的包
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import click

sns.set()


def smooth(data,weight=0.75):
    smoothed = []
    last = data[0]
    for point in data:
        smooth_value = last * weight + (1-weight) * point
        smoothed.append(smooth_value)
        last = smooth_value
    return smoothed


@click.command()
@click.option('--path_end', default='avg_acc.npy', type=str)
@click.option('--train_or_eval', default='Train', type=str)
@click.option('--y_label', default="success rate", type=str)
@click.option('--picture_name', default="avg_acc_2", type=str)
@click.option('--plot_result', default=True, type=bool)
def plot(path_end, train_or_eval, y_label, picture_name, plot_result):

    # path_share = 'results/simple_uav_task'
    path_share = 'results/complex_uav_task'
    algo_name_1 = 'MP_DQN_HER'
    algo_name_2 = 'PA_DQN_HER'
    algo_name_3 = 'MP_DQN'

    df = []
    # 导入npy文件路径位置
    MP_DQN_HER_0 = np.load('{}/{}/0/{}'.format(path_share, algo_name_1, algo_name_1 + path_end))
    MP_DQN_HER_1 = np.load('{}/{}/1/{}'.format(path_share, algo_name_1, algo_name_1 + path_end))
    MP_DQN_HER_2 = np.load('{}/{}/20/{}'.format(path_share, algo_name_1, algo_name_1 + path_end))
    MP_DQN_HER_3 = np.load('{}/{}/40/{}'.format(path_share, algo_name_1, algo_name_1 + path_end))
    MP_DQN_HER_4 = np.load('{}/{}/60/{}'.format(path_share, algo_name_1, algo_name_1 + path_end))
    MP_DQN_HER_5 = np.load('{}/{}/80/{}'.format(path_share, algo_name_1, algo_name_1 + path_end))
    MP_DQN_HER_6 = np.load('{}/{}/100/{}'.format(path_share, algo_name_1, algo_name_1 + path_end))
    MP_DQN_HER_0 = smooth(MP_DQN_HER_0)
    MP_DQN_HER_1 = smooth(MP_DQN_HER_1)
    MP_DQN_HER_2 = smooth(MP_DQN_HER_2)
    MP_DQN_HER_3 = smooth(MP_DQN_HER_3)
    MP_DQN_HER_4 = smooth(MP_DQN_HER_4)
    MP_DQN_HER_5 = smooth(MP_DQN_HER_5)
    MP_DQN_HER_6 = smooth(MP_DQN_HER_6)

    PA_DQN_HER_0 = np.load('{}/{}/0/{}'.format(path_share, algo_name_2, algo_name_2 + path_end))
    PA_DQN_HER_1 = np.load('{}/{}/1/{}'.format(path_share, algo_name_2, algo_name_2 + path_end))
    PA_DQN_HER_2 = np.load('{}/{}/20/{}'.format(path_share, algo_name_2, algo_name_2 + path_end))
    PA_DQN_HER_3 = np.load('{}/{}/40/{}'.format(path_share, algo_name_2, algo_name_2 + path_end))
    PA_DQN_HER_4 = np.load('{}/{}/60/{}'.format(path_share, algo_name_2, algo_name_2 + path_end))
    PA_DQN_HER_5 = np.load('{}/{}/80/{}'.format(path_share, algo_name_2, algo_name_2 + path_end))
    PA_DQN_HER_6 = np.load('{}/{}/100/{}'.format(path_share, algo_name_2, algo_name_2 + path_end))
    PA_DQN_HER_0 = smooth(PA_DQN_HER_0)
    PA_DQN_HER_1 = smooth(PA_DQN_HER_1)
    PA_DQN_HER_2 = smooth(PA_DQN_HER_2)
    PA_DQN_HER_3 = smooth(PA_DQN_HER_3)
    PA_DQN_HER_4 = smooth(PA_DQN_HER_4)
    PA_DQN_HER_5 = smooth(PA_DQN_HER_5)
    PA_DQN_HER_6 = smooth(PA_DQN_HER_6)

    MP_DQN_0 = np.load('{}/{}/0/{}'.format(path_share, algo_name_3, algo_name_3 + path_end))
    MP_DQN_1 = np.load('{}/{}/1/{}'.format(path_share, algo_name_3, algo_name_3 + path_end))
    MP_DQN_2 = np.load('{}/{}/20/{}'.format(path_share, algo_name_3, algo_name_3 + path_end))
    MP_DQN_3 = np.load('{}/{}/40/{}'.format(path_share, algo_name_3, algo_name_3 + path_end))
    MP_DQN_4 = np.load('{}/{}/60/{}'.format(path_share, algo_name_3, algo_name_3 + path_end))
    MP_DQN_5 = np.load('{}/{}/80/{}'.format(path_share, algo_name_3, algo_name_3 + path_end))
    MP_DQN_6 = np.load('{}/{}/100/{}'.format(path_share, algo_name_3, algo_name_3 + path_end))
    MP_DQN_0 = smooth(MP_DQN_0)
    MP_DQN_1 = smooth(MP_DQN_1)
    MP_DQN_2 = smooth(MP_DQN_2)
    MP_DQN_3 = smooth(MP_DQN_3)
    MP_DQN_4 = smooth(MP_DQN_4)
    MP_DQN_5 = smooth(MP_DQN_5)
    MP_DQN_6 = smooth(MP_DQN_6)

    if plot_result:
        # MP_DQN_HER
        MP_DQN_HER = np.vstack((MP_DQN_HER_0, MP_DQN_HER_1, MP_DQN_HER_2, MP_DQN_HER_3,
                                   MP_DQN_HER_4, MP_DQN_HER_5, MP_DQN_HER_6))
        df_MP_DQN_HER = pd.DataFrame(MP_DQN_HER).melt(var_name='epoch', value_name='exp')
        # PA_DQN_HER
        PA_DQN_HER = np.vstack(
            (PA_DQN_HER_0, PA_DQN_HER_1, PA_DQN_HER_2,
             PA_DQN_HER_3, PA_DQN_HER_4, PA_DQN_HER_5,
             PA_DQN_HER_6))
        df_PA_DQN_HER = pd.DataFrame(PA_DQN_HER).melt(var_name='epoch', value_name='exp')
        # MP_DQN
        MP_DQN = np.vstack(
            (MP_DQN_0, MP_DQN_1, MP_DQN_2, MP_DQN_3, MP_DQN_4, MP_DQN_5, MP_DQN_6))
        df_MP_DQN = pd.DataFrame(MP_DQN).melt(var_name='epoch', value_name='exp')

        label = ['HER-MPDQN(our)','HER-PDQN', 'MP-DQN']
        df.append(df_MP_DQN_HER)
        df.append(df_PA_DQN_HER)
        df.append(df_MP_DQN)

        for i in range(3):
            df[i]['Algorithm'] = label[i]

        df = pd.concat(df, ignore_index=True)  # 合并
        print('start plot, wait...')
        sns.lineplot(x="epoch", y="exp", hue="Algorithm", data=df)

        plt.legend(loc='lower right')
        plt.xlabel('epoch')
        plt.ylabel('{}'.format(y_label))
        plt.savefig("{}.png".format(picture_name), dpi=500)
        print('save jpg success!')


if __name__ == '__main__':
    plot()
