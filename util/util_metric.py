from sklearn.metrics import f1_score


def show_metric(true, pred, f_avg='micro'):
    f1 = f1_score(true, pred, average=f_avg)
    print(f"\n F1 Score : {f1: .4f}")
