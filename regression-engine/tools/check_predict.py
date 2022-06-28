import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from collections import defaultdict


def main():
    df1 = pd.read_csv('../train.csv').to_dict()
    df2 = pd.read_csv('./output.csv').to_dict()

    target = {k: v for k, v in zip(
        df1['image_id'].values(), df1['label'].values())}
    pred = {k: v for k, v in zip(
        df2['image_id'].values(), df2['label'].values())}

    t = {}
    for k, v in pred.items():
        t[k] = target[k]

    target = t
    # check all
    target = {k: v for k, v in sorted(target.items(), key=lambda x: x[0])}
    pred = {k: v for k, v in sorted(pred.items(), key=lambda x: x[0])}
    for (k1, v1), (k2, v2) in zip(target.items(), pred.items()):
        assert k1 == k2, "doesn't match!! {} <--> {}".format(k1, k2)

    print('\naccuracy:', accuracy_score(
        list(target.values()), list(pred.values())))
    print(confusion_matrix(list(target.values()), list(pred.values())))
    print(classification_report(list(target.values()), list(pred.values())))


def pseudo_labelling():
    df = pd.read_csv('./pseudo_lb_for_extra.csv')
    print('df nums:', len(df))
    classes = defaultdict(int)
    psd_op = []
    for i in range(len(df)):
        x = np.fromstring(df.loc[i, 'label'].replace('[', '').replace(']', ''),
                          dtype=np.float,
                          sep=' ')
        if np.max(x) > 0.99:
            classes[np.argmax(x)] += 1
            psd_op.append([df.loc[i, 'image_id'], np.argmax(x)])
    print(classes)
    print('pseudo-lb nums:', sum(classes.values()))
    psd_df = pd.DataFrame(data=psd_op, columns=['image_id', 'label'])
    psd_df.to_csv('../extra.csv', index=False)
    print(psd_df.head(10))


if __name__ == "__main__":
    pseudo_labelling()
