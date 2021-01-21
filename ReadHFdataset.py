import os
import re


def LoadCases():
    subfolders = [x[0] for x in os.walk("./data")]
    before10 = "./data/0"
    after10 = "./data/"
    cases = []
    testcases = []
    countOfNumberOfFile = 1
    for i in subfolders:
        i = i.replace("\\", "/")
        if re.match("./data/[0-9]+$", i) != None:
            curr = str(countOfNumberOfFile)
            if countOfNumberOfFile < 10:
                curr = ('0' + curr)
            testcases.append("./data/" + (curr) + "/test.png")
            casesLocal = []
            casesLocal.append("./data/" + (curr) + "/1/1.png")
            casesLocal.append("./data/" + (curr) + "/1/2.png")
            casesLocal.append("./data/" + (curr) + "/2/1.png")
            casesLocal.append("./data/" + (curr) + "/2/2.png")
            casesLocal.append("./data/" + (curr) + "/3/1.png")
            casesLocal.append("./data/" + (curr) + "/3/2.png")
            cases.append(casesLocal)
            countOfNumberOfFile += 1
    return cases, testcases
