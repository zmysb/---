
from sklearn import svm
import numpy as np
import os
import pickle as pkl
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize,scale


def loadCSVfile(addr):
    tmp=np.loadtxt(addr,delimiter=",",skiprows=0)
    a=tmp[:,1:7].astype(float)
    b=normalize(a,norm="l2")
    data=scale(b)
    label=tmp[:,-1].astype(int)
    num=tmp[:,0].astype(int)
    return data,label,num

def logReg_classify(train_set,train_tag,test_set,test_tag):

    clf = svm.SVC(kernel='poly',degree=3)
    clf_res = clf.fit(train_set,train_tag)
    train_pred  = clf_res.predict(train_set)
    train_score=clf_res.score(train_set,train_tag)
    test_pred = clf_res.predict(test_set)
    test_score=clf_res.score(test_set,test_tag)

    print('=== 分类训练完毕，分类结果如下 ===')
    print('训练集 准确率:{c} f1:{e} 精确率:{a} 召回率:{b}'.format(c=train_score,e=f1_score(train_tag, train_pred),a=precision_score(train_tag,train_pred),b=recall_score(train_tag,train_pred)))
    print('检验集 准确率:{c} f1:{e} 精确率:{a} 召回率:{b}'.format(c=test_score,e=f1_score(test_tag, test_pred),a=precision_score(test_tag,test_pred),b=recall_score(test_tag,test_pred)))

    return clf_res


if __name__=='__main__':
    path_root="/Users/chenyilin/Desktop/svm"
    path_predictor=os.path.join(path_root,'predictor.pkl')
    path_finalPre=os.path.join(path_root,'predict.csv')
    n=10

    predictor=None
    # # ===================================================================训练
    if not os.path.exists(path_predictor):
        print('=== 未检测到判断器存在，开始进行分类过程 ===')
        data, label,num= loadCSVfile("/Users/chenyilin/Desktop/svm/data_train.csv")
        rarray=np.random.random(size=len(data))
        train_set = []
        train_tag = []
        test_set = []
        test_tag = []
        for i in range(len(data)):
            if rarray[i] < 0.8:
                train_set.append(data[i, :])
                train_tag.append(label[i])
            else:
                test_set.append(data[i, :])
                test_tag.append(label[i])
        # print(train_set, train_tag, test_set, test_tag)
        predictor = logReg_classify(train_set, train_tag, test_set, test_tag)
        x = open(path_predictor, 'wb')
        pkl.dump(predictor, x)
        x.close()
    else:
        print('=== 检测到分类器已经生成，跳过该阶段 ===')
    # # ===================================================================预测
    if not os.path.exists(path_finalPre):
        if not predictor:
            x = open(path_predictor, 'rb')
            predictor = pkl.load(x)
            x.close()
        data_demo, tmp, data_num = loadCSVfile('/Users/chenyilin/Desktop/svm/data_test.csv')
        data_predict = np.zeros((len(data_num), 2))
        x = predictor.predict(data_demo)
        data_predict[:, 0] = data_num
        data_predict[:, 1] = x
        final = data_predict.astype(int)
        np.savetxt(path_finalPre, final, delimiter=',')
        print(final)


