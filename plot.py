# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rc('font', family='Times New Roman')


def plot_line(x, y, ymin, ymax, ys_label, x_axis_label, y_axis_label, title, sci_no, filename):
    """
    Parameters:
    x - list, [1,2,3] or ['a','b','c']
    y - list, [y1,y2,...,yn]. yi:list, [1,2,3]
    width - float, the width of the bar
    ys - list, each element is a str, will be show in the legend
    x_axis_label, y_axis_label - str, will be added in axis
    title - str, title
    plotType - str, bar, line...
    """
    num_x = np.arange(len(x))
    # define subplots
    fig, ax = plt.subplots()
    # draw
    color = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange']
    color = color * len(y)
    lineType = ['-', '--', ':', '--']
    marker = ['.', '*', 'o', 'v']
    ax.tick_params(labelsize=20)
    for i in range(len(y)):
        ax.plot(num_x, y[i], marker=marker[i], linestyle=lineType[i], color=color[i], label=ys_label[i])

    # set label
    ax.set_xlabel(x_axis_label, fontsize=20)
    ax.set_ylabel(y_axis_label, fontsize=20)
    if ymin is not None:
        plt.ylim(ymin, ymax)
    # set x axis
    ax.set_xticks(num_x)
    if sci_no:
        ax.set_xticklabels([f"1e-{i}" if i else 0 for i in x])
    else:
        ax.set_xticklabels(x)
    ax.set_title(title, fontsize=20)
    # 网格线
    ax.grid()
    ax.legend(fontsize=20, loc='best')
    # fig.tight_layout()
    if filename is not None:
        fig = plt.gcf()
        fig.savefig(f'./{filename}.pdf')
    else:
        plt.show()


def plot_lines(x, y, y_range, ys_label, x_axis_label, y_axis_label, title, sci_no, filename):
    fig = plt.figure(figsize=(8, 4))
    num_x = np.arange(len(x))
    axes = fig.subplots(nrows=1, ncols=len(y))
    color = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange']
    color = color * len(y)
    lineType = ['-', '--', ':', '-.']
    marker = ['.', '*', 'v', 'o']
    j = 0
    for ax in fig.axes:
        ax.tick_params(labelsize=20)
        if ys_label is None:
            for i in range(len(y[j])):
                ax.plot(num_x, y[j][i], marker=marker[i], markersize=10, linestyle=lineType[i], color=color[i])
        else:
            for i in range(len(y[j])):
                ax.plot(num_x, y[j][i], marker=marker[i], markersize=10, linestyle=lineType[i],
                        color=color[i], label=ys_label[i])

        # set label
        # if j == 0:
        ax.set_ylabel(y_axis_label, fontsize=20)
        ax.set_xlabel(x_axis_label, fontsize=20)
        if y_range is not None:
            ax.set_ylim(y_range[j][0], y_range[j][1])
        # set x axis
        ax.set_xticks(num_x)
        if sci_no:
            ax.set_xticklabels([f"1e{i}" if i else 0 for i in x], rotation=0, fontsize=16)
        else:
            ax.set_xticklabels(x, rotation=0, fontsize=16)
        ax.set_title(title[j], fontsize=20)
        # 网格线
        ax.grid(linestyle='-.', linewidth=0.5)
        # '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
        j += 1
    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, loc='upper center', ncol=i + 1, edgecolor='black', fontsize=20)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    # fig.tight_layout()
    if filename is not None:
        fig = plt.gcf()
        fig.savefig(f'./{filename}.pdf')
        plt.show()
    else:
        plt.show()


def plot_subplot_three_grouped_bar(x, y, ymin, width, ys_label, x_axis_label, y_axis_label, title, filename):
    fig = plt.figure(figsize=(12, 5))
    num_x = np.arange(len(x))
    axes = fig.subplots(nrows=1, ncols=4)
    i = 0
    y1_label, y2_label, y3_label = ys_label[0], ys_label[1], ys_label[2]
    color_name = 'rainbow'

    for ax in fig.axes:
        y1, y2, y3 = y[i][0], y[i][1], y[i][2]
        ax.bar(num_x - width, y1, width, label=y1_label, bottom=ymin[i], color='#B97062')
        ax.bar(num_x, y2, width, label=y2_label, bottom=ymin[i], color='#F2DED0')
        ax.bar(num_x + width, y3, width, label=y3_label, bottom=ymin[i], color='#878787')
        # if i == 0:
        ax.set_ylabel(y_axis_label, fontsize=10.5)
        ax.tick_params(labelsize=10.5)
        ax.set_xticks(num_x)
        ax.set_xticklabels(x, rotation=0)
        ax.set_title(title[i], fontsize=10.5)
        i += 1
    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, loc='upper center', ncol=3, edgecolor='black', fontsize=10.5)
    fig.tight_layout(rect=(0.01, 0.2, 1, 0.88))
    if filename is not None:
        # fig = plt.gcf()
        fig.savefig(f'./{filename}.pdf')
        plt.show()
    else:
        plt.show()


def plot_subplot_multi_grouped_bar(title, data,group_labels):
    # 定义柱状图的颜色
    # colors = ['#B97062', '#F2DED0', '#878787', 'cornflowerblue']
    # labels = ['MF', 'ESCI_MF', 'LightGCN', 'ESCI_LightGCN']
    colors = ['#8E2D30', '#F2E8E3', '#5D74A2', '#33395B']
    labels = ["baesline", "no_Bi_Level", "no_GAN", "FGADA"]

    # 分组的标签
    # group_labels = [f'Group {i}' for i in range(1,5)]

    # 柱子的宽度
    bar_width = 0.2

    # 计算每个柱子的位置
    index = np.arange(len(group_labels))
    bar_positions = [index + i * bar_width for i in range(data.shape[1])]

    # 画图
    for i, bars in enumerate(bar_positions):
        plt.bar(bars, data[:, i], width=bar_width, label=labels[i], color=colors[i])

    # 设置图表标题和标签
    # plt.title('Target Items Grouped by Frequency')
    # plt.suptitle('Target Items Grouped by Frequency', y=0, fontsize=16)
    # plt.subplots_adjust(bottom=0.15)
    plt.ylabel('K@40', fontsize=16)
    # plt.xlabel('Target Users Grouped by Interaction Frequency', fontsize=16)
    # 设置x轴标签
    plt.xticks(index + bar_width * (data.shape[1] - 1) / 2, group_labels, fontsize=16)

    # 添加图例
    plt.legend(fontsize=16, loc='best')
    plt.tight_layout()
    # 显示图表
    plt.savefig(f'{title}.pdf')
    plt.show()


if __name__ == '__main__':

    # name = ['MF', 'LightGCN']
    #
    # ad_esci = [2.2677, 2.2463]
    # ad_esci_r = [2.9780, 2.3302]
    # ad_esci_n = [2.5572, 3.8738]
    #
    # ml_esci = [1.771, 1.6907]
    # ml_esci_r = [1.9207, 1.7030]
    # ml_esci_n = [1.7785, 1.7030]
    #
    # coat_esci = [0.0213, 0.0229]
    # coat_esci_r = [0.01983, 0.0201]
    # coat_esci_n = [0.01980, 0.0193]
    #
    # ya_esci = [0.0033, 0.0035]
    # ya_esci_r = [0.00316, 0.00348]
    # ya_esci_n = [0.00301, 0.00312]
    #
    # plot_subplot_three_grouped_bar(name, [[ad_esci_n, ad_esci_r, ad_esci], [ml_esci_n, ml_esci_r, ml_esci],
    #                                       [coat_esci_n, coat_esci_r, coat_esci],
    #                                       [ya_esci_n, ya_esci_r, ya_esci]],
    #                                [0, 0, 0, 0], 0.2, ['ESCI$_{ne}$', 'ESCI$_{re}$', 'ESCI'], None, 'Recall@5',
    #                                ['Adressa', 'MovieLens-10M', 'Coat', 'Yahoo! R3'], 'exposure')

    # name = ['MF', 'LightGCN']
    # ad_esci_u = [0.0337, 0.036529801]
    # ad_esci_i = [0.025385512, 0.03006405]
    # ml_esci_u = [0.042057176, 0.050127334]
    # ml_esci_i = [0.040847553, 0.038449477]
    #
    # coat_esci_u = [0.0208, 0.0216]
    # coat_esci_i = [0.0156, 0.0174]
    # ya_esci_u = [0.00312, 0.00324]
    # ya_esci_i = [0.00295, 0.00283]
    #
    # plot_subplot_three_grouped_bar(name, [[ad_esci_u, ad_esci_i, ad_esci], [ml_esci_u, ml_esci_i, ml_esci],
    #                                       [coat_esci_u, coat_esci_i, coat_esci],
    #                                       [ya_esci_u,ya_esci_i,ya_esci]],
    # [0, 0, 0,0], 0.2, ['ESCI-user', 'ESCI-item', 'ESCI'], None, 'Recall@5',
    # ['Adressa', 'MovieLens-10M', 'Coat', 'Yahoo! R3'], 'satisfaction')
    #
    # adv_lambda = [0.1, 1, 10, 100, 1000]
    # ad_esci = [0.03204956, 0.034398, 0.033941197, 0.03389941, 0.032346097]
    # ad_lgnesci = [0.034059435, 0.040853934, 0.037472487, 0.036401956, 0.038267168]
    # ml_esci = [0.034179057, 0.045626856, 0.048887702, 0.043364071, 0.044856567]
    # ml_lgnesci = [0.03605239, 0.050115287, 0.049955732, 0.052196783, 0.051105611]
    # plot_line(adv_lambda, [ad_esci, ad_lgnesci], 0.03, 0.05,
    #           ['ESCI_MF', 'ESCI_LightGCN'], '', '', None, False, 'lambda_ad')
    # plot_line(adv_lambda, [ml_esci, ml_lgnesci], 0.03, 0.07,
    #           ['ESCI_MF', 'ESCI_LightGCN'], '', '', None, False, 'lambda_ml')
    #
    # LAST_gan = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    # adv_epsilon =[0.1,0.5,10,100,1000]
    # bpr_ndcg = [0.182, 0.189, 0.1975, 0.1981, 0.181,0.1812]
    # bpr_eo = [0.63899, 0.5804, 0.579, 0.5841, 0.5863,0.5863]
    # ml_esci = [0.031562016, 0.039602701, 0.044193985, 0.048887702, 0.020684806]
    # ml_lgnesci = [0.036087894, 0.047799974, 0.045411975, 0.052196783, 0.035878103]
    # plot_line(LAST_gan, [bpr_ndcg, bpr_eo], 0.17, 0.65,
    #           ['NDCG@40', 'EO@40'], '', '', None, False, 'FGADAGD_BPR_LASTFM')
    # plot_line(adv_epsilon, [ml_esci, ml_lgnesci], 0.01, 0.07,
    #           ['ESCI_MF', 'ESCI_LightGCN'], '', '', None, False, 'epsilon_ml')
    #
    # x = [-1, -2, -3, -4, -5]
    # ad_esci_fix_alpha = [0.02621115, 0.028878589, 0.030066698, 0.030965992, 0.029992496]
    # ad_esci_fix_beta = [0.030083295, 0.032836254, 0.030066698, 0.032730429, 0.029021311]
    #
    # ad_esci_ada_alpha = [0.027836809, 0.034699888, 0.034398201, 0.033515664, 0.030808027]
    # ad_esci_ada_beta = [0.034890223, 0.033977275, 0.034398201, 0.034468326, 0.031904778]
    # plot_lines(x, [[ad_esci_fix_alpha, ad_esci_ada_alpha], [ad_esci_fix_beta, ad_esci_ada_beta]],
    #            [[0.01, 0.05], [0.01, 0.05]], ['normal learning', 'curriculum learning'], '', 'Recall',
    #            [r'$\alpha$', r'$\beta$'], True, 'Curriculum')
    #
    Bpr = np.array([
        [2.6003, 2.5772, 2.978, 2.2677],
        [2.5447, 3.8738, 2.3302, 2.2463],
            [2.0365,1.7785,1.9207,1.771],
        [1.9013,1.7030,1.7030,1.6907]
    ])
    groupbpr=["LFM_bpr","LFM_gccf" ,"ML_bpr","ML_gccf"]
    plot_subplot_multi_grouped_bar('AblationStudy', Bpr,groupbpr)

    #
    # ml_10m = np.array([
    #     [0.2869,0.1933,0.6126,0.6361,2.6003]
    #     [0.2914,0.1959,0.6107,0.6354,2.5772],
    #     [0.2739,0.1847,0.6842,0.6815,0.2978],
    #     [0.2936,0.1971,0.5456,0.5672,2.2677]
    # ])
    # plot_subplot_multi_grouped_bar('ml-10m_group', ml_10m)
    #
    # yahooR3 = np.array([
    #     [0.00151,0.003258, 0.002623, 0.003103],
    #     [0.001845,0.003512,0.003012,0.003546],
    #     [0.00241,0.004902,0.004013,0.005103],
    #     [0.0124,0.00741,0.00901,0.00361]
    # ])
    # plot_subplot_multi_grouped_bar('yahoo_group', yahooR3)