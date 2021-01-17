import  os
subfolders =[x[0] for x in os.walk("./data")]
cases=[]
countOfNumberOfFile=1
list=[]
before10="./data/0"
after10="./data/"
EditedSubFolder=[]
testcases=[]
count=0
subfolders2=[]
for i in subfolders:
    i=i.replace("\\","/")
    if i == "./data":
        continue
    elif countOfNumberOfFile<10:
        if i == (before10 + str(countOfNumberOfFile)):
            testcases.append(i+"/test.png")
            countOfNumberOfFile = countOfNumberOfFile + 1
        else:
            subfolders2.append(i)
            if count == 3:
                cases.append(list)
                list = []
                count = 0
            for j in range(1, 3, 1):
                list.append(i + '/' + str(j) + ".png")
            count += 1
    elif i == (after10 + str(countOfNumberOfFile)):
        continue
    else:
        subfolders2.append(i)
        if count == 3:
            cases.append(list)
            list = []
            count = 0
        for j in range(1, 3, 1):
            list.append(i + '/' + str(j) + ".png")
        count += 1

print(testcases)