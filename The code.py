# functions


# to get inputs for X_test
def get_test_samples():
    Sample = input("\nenter the sample:    ")
    if re.search("^[0-1]{128}$", Sample):
        return Sample
    else:
        print("\nplease enter a valid sample")
        return get_test_samples()


# any 128 bit number
# input is a string of 128 char containing only 0 or 1
# output is an array of 32 arrays with float values between 0.0 and 1.5
def convolute_ip_test(a):
    Count = 0
    ConvolutedIp = []
    ConvolutedIpElement = ""
    for _ in a:
        Count += 1
        if Count == 4:
            ConvolutedIpElement += _
            ConvolutedIp.append(int(ConvolutedIpElement, 2) / 10)
            ConvolutedIpElement = ""
            Count = 0
        else:
            ConvolutedIpElement += _

    return ConvolutedIp


# any 128 bit number
# input is an array of 128 arrays with each sub array containing 0 or 1 for the ip
# output is an array of 32 arrays with float values between 0.0 and 1.5
def convolute_ip(a):
    Count = 0
    ConvolutedIp = []
    ConvolutedIpElement = ""
    for _ in a:
        _ = _.strip("[")
        _ = _.strip("]")
        Count += 1
        if Count == 4:
            ConvolutedIpElement += str(_[0])
            ConvolutedIp.append(int(ConvolutedIpElement, 2) / 10)
            ConvolutedIpElement = ""
            Count = 0
        else:
            ConvolutedIpElement += _

    return ConvolutedIp


# to predict if an ip is outlier
def pedict_outlier(Predictor, X):
    Prediction = Predictor.predict(X)
    if Prediction == 1:
        return "null"
    else:
        return X


# main Structure
if __name__ == "__main__":

    # ---Imoprting Libraries---

    import numpy as np
    import pandas as pd
    import re
    import csv
    from sklearn.ensemble import IsolationForest

    # ---Training Phase---

    Dataset = pd.read_csv(input("\n Enter the name of the dataset"))
    X_train = Dataset.iloc[:, :].values

    X_test = []
    Outliers = []
    X_train_convoluted = []
    CompareTest = []

    for _ in X_train:
        X_train_convoluted.append(convolute_ip(_))

    # contamination = 0 - 0.5 -- expected outlier to total ratio
    # np.random.RandomState(42) -- to get replicable results
    Predictor = IsolationForest(
        max_samples=100, contamination=0.01, random_state=np.random.RandomState(42)
    )

    Predictor.fit(X_train_convoluted)

    # ---Test Phase---

    Test = True
    while Test == True:
        TypeTest = input("\nDo you want to use a dataset or add samples? (d or s)")
        if TypeTest.lower()[0] == "s":
            Test = False
            while True:
                MoreSamples = input("\nDo you want to add samples? (yes or no)")
                if MoreSamples.lower()[0] == "y":
                    X_test.append(get_test_samples())
                elif MoreSamples.lower()[0] == "n":
                    break
                else:
                    print("\nenter a valid responce")

            for _ in X_test:
                ConvolutedIpElement = convolute_ip_test(_)
                Prediction = pedict_outlier(Predictor, [ConvolutedIpElement])
                if type(Prediction) == str:
                    Outliers.append(Prediction)
                else:
                    Outliers.append(_)

            for _ in range(len(X_test)):
                CompareTest.append((X_test[_], Outliers[_]))

            print(CompareTest)

        elif TypeTest.lower()[0] == "d":
            Test = False
            DatasetTest = pd.read_csv(input("\n Enter the name of the dataset"))
            X_test = DatasetTest.iloc[:, :].values
            X_test_convoluted = []

            for a in X_test:
                X_test_convoluted.append(convolute_ip(a))

            for _ in X_test_convoluted:
                ConvolutedIpElement = []
                for i in _:
                    ConvolutedIpElement.append(i)
                Prediction = pedict_outlier(Predictor, [ConvolutedIpElement])
                if type(Prediction) == str:
                    Outliers.append(Prediction)
                else:
                    Outliers.append("anomaly")

            for _ in range(len(X_test_convoluted)):
                CompareTest.append((_ + 1, Outliers[_]))

            print(CompareTest)
        else:
            print("\nenter a valid responce")
