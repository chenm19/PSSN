import copy

import numpy as np
import pandas as pd 
import seaborn as sns
from matplotlib import pyplot as plt 
from scipy import stats
from statannot.StatResult import StatResult

import pickle


def FDR(p_vals):
    from scipy.stats import rankdata
    ranked_p_values = rankdata(p_vals)
    fdr = p_vals * len(p_vals) / ranked_p_values
    fdr[fdr > 1] = 1

    return fdr

###### median
def count_median(lis):
    lis = sorted(lis)
    if len(lis) % 2 == 0:
        mid = float((lis[len(lis) // 2] + lis[len(lis) // 2 - 1])) / 2
    else:
        mid = lis[len(lis) // 2]
    return mid

def draw_boxplt(input_filename, save_path, data_name, K, times_count):
    box_pair = [("#1", "#6"), ("#1", "#2"), ("#1", "#3"), ("#1", "#4"),
                ("#1", "#5"), ("#2", "#6"), ("#2", "#3"),
                ("#2", "#4"), ("#2", "#5"), ("#3", "#6"),
                ("#3", "#4"), ("#3", "#5"), ("#4", "#6"),
                ("#4", "#5"), ("#5", "#6")]

    CLINICAL_LABELS = ['EcogPtMem', 'EcogPtLang', 'EcogPtVisspat', 'EcogPtPlan', 'EcogPtOrgan',
                        'EcogPtDivatt', 'EcogPtTotal', 'EcogSPMem', 'EcogSPLang', 'EcogSPVisspat',
                        'EcogSPPlan', 'EcogSPOrgan', 'EcogSPDivatt', 'EcogSPTotal']

    CLINICAL_LABELS2 = ['Ecog Memory', 'Ecog Language', 'Ecog Visuospatial Abilities', 'Ecog Planning', 'Ecog Organization',
                    'Ecog Divided Attention', 'Ecog Total', 'Ecog Memory', 'Ecog Language', 'Ecog Visuospatial Abilities',
                    'Ecog Planning', 'Ecog Organization', 'Ecog Divided Attention', 'Ecog Total']

    with open(input_filename, "rb") as f:
        mydict = pickle.load(f)

    id,label,clinical_score = [],[],[]
    for i, (k, v) in enumerate(mydict.items()):
        id.append(k)
        label.append(v["label"])
        clinical_score.append(v["clinical"])

    # for i in range(len(label)):
    #     label[i] = "cluster"+ str(label[i])

    df_id, df_label,df_label2, df_score = [], [], [], []
    for i in range(len(label)*14):
        df_id.append(id[i//14])
        df_label.append(CLINICAL_LABELS[i%14])
        df_label2.append(CLINICAL_LABELS2[i%14])
        tmp = [None for j in range(6)]
        tmp[label[i//14]] = clinical_score[i//14][i%14]
        df_score.append(tmp)

    df = pd.DataFrame(df_score, columns=["#{}".format(i) for i in range(1, 7)])
    df["labels"] = df_label
    df["id"] = df_id
    # df = df[df.columns[::-1]]

    cat = [0 for i in range(len(df_label))]
    for i in range(len(df_label)):
        cat[i] = 'Specialist' if 'SP' in df_label[i] else "Self"

    df['E'] = cat
    df.to_csv('{}/dff.csv'.format(save_path), index=False)
    _data = pd.read_csv('{}/dff.csv'.format(save_path))

    for index in range(7, 14):
        # if index != 9:
        #     continue
        data = copy.deepcopy(_data)
        # data = df.copy(deep=True)
        # print(data.columns)
        columns = data.columns.tolist()
        columns.remove('id')
        columns.remove('labels')
        columns.remove('E')
        i_ = CLINICAL_LABELS[index]
        title = str(i_)
        ori_title = 'ORI CT CLASS ' + str(i_)
        df_0 = data.loc[data['labels'] == i_]

        cat = pd.DataFrame(cat)

        ##### outlier
        dic = {}
        for column in columns:
            df = df_0.loc[:,column]
            #df = df.reset_index(drop=True)
            df = pd.DataFrame({column: df})
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1    #IQR is interquartile range. 
            #print((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR)))
            idx = ~((df < (Q1 - 2 * IQR)) | (df > (Q3 + 2 * IQR))).any(axis=1)
            #   cat2 = pd.concat([cat.loc[idx]], axis=1)
            #   print(df)
            train_cleaned = pd.concat([df.loc[idx]], axis=1)

            #   print("------------------------------------",len(train_cleaned),len(cat))
            dic[column] = train_cleaned
            #   cat2 = pd.concat([df_0.loc[idx,'E']], axis=1)
            #   print(cat2)
        
        #####
        data0 = []
        for k in dic:
            #data0.append(dic[k])
            num = dic[k].shape[0]
            name = [k for _ in range(num)] 
            df = pd.DataFrame({'value':dic[k].loc[:,k].tolist(), 'name':name})
            data0.append(df)

        data = pd.concat(data0, axis=0)
        df2 = data.reset_index(drop=True) #sns.boxplot 

        order = ['#1', '#2', '#3', '#4', '#5', '#6']

        ##### 
        means_ = {}
        medians_ = {}
        for i in order:
            mean = df2[df2['name']==i].iloc[:,0].mean()
            median = df2[df2['name']==i].iloc[:,0].median()
            medians_[i] = median
            means_[i] = mean


        # print(means_)
        #   print(".",cat2)
        #   df2['Report by'] = cat2   hue='Report by',
        fig = plt.figure(dpi=400, figsize=(4, 3))

        # print(df2)
        #ax = fig.add_subplot(121)
        ax = sns.boxplot(x="name", y="value", data=df2, width=0.5, linewidth=1.0,  palette="Set3", medianprops=dict(color="black"))
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)

        #####t-test p-value
        dic_ = {}
        pairs = []
        p_vals = []
        for tup in box_pair:
            label_1 = tup[0]
            label_2 = tup[1]

            box_data1 = df2[df2['name'] == label_1].iloc[:, 0]
            box_data2 = df2[df2['name'] == label_2].iloc[:, 0]
            stat, pval = stats.ttest_ind(a=box_data1.dropna(), b=box_data2.dropna())
            result = StatResult(
                    't-test independent samples', 't-test_ind', 'stat', stat, pval
                )

            b = result.pval
            #b = format(b,'.3e')
            pairs.append(tup)
            p_vals.append(b)


        #####p-value fdr
        p_vals = np.array(p_vals)
        #print(p_vals)
        fdr = FDR(p_vals)
        #print(pd.DataFrame({'pval': p_vals, 'fdr':fdr}))

        #####adjust p-value
        for i in range(len(fdr)):
            value = fdr[i]
            tup_ = pairs[i]
            label_1 = tup_[0]
            label_2 = tup_[1]
            if value < 0.05:
                if means_[label_1] < means_[label_2]:
                    tup_ = (label_2, label_1)
                else:
                    tup_ = tup_
                dic = {tup_ : value}
                dic_.update(dic)
        #   print(dic_)

        dic_order = []
        d_order = sorted(dic_.items(),key=lambda x:x[1],reverse=False)
        for i in range(len(d_order)):
            dic_order.append(d_order[i][0])

        #####print
        s_total = ''
        for j in dic_order:
            s = j[0] + ' VS. ' + j[1] + ' : ' + str(format(dic_[j], '.3e'))
            if s_total == '':
                s_total = s
            else:
                s = '\n' + s
                s_total += s
        print(title)
        print(s_total)
        ylim = ax.get_ylim()
        # print(ylim)
        yrange = ylim[1] - ylim[0]
        xlim = ax.get_xlim()
        #   ax.set_ylim([0, 5])
        bbox = dict(boxstyle="round", fc="1")
        ax.text(x=1.06, y=0.03, fontsize=10, transform=ax.transAxes, bbox=bbox, s=s_total)
        #plt.figure(dpi=400, figsize=(4, 3))
        plt.title(title, fontsize=16, x=1.0)
        #   plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title='Report by')
        # plt.xticks(rotation=90)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(8)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(8)

        #ax.set(yticklabels=[])
        ax.set(ylabel=None)
        ax.set(xlabel=None)
        plt.subplots_adjust(right=0.7)
        
        out_path = save_path + "hist_{}_k={}_id={}_{}".format(data_name, K, times_count, i_)
        plt.savefig(out_path, dpi=400, bbox_inches="tight")
        plt.show()
        plt.clf()

    print("Drawing 7 dist plots on {} successfully".format(save_path))
    #   print(label)


if __name__ == "__main__":
    draw_boxplt('test/for_hist/box_data_delta1_k=6_3000_7.pkl', 'test/box_test/', "delta2", 6, 222)
