import os


def readIMAdata():
    Y = []
    ImagesDir = []
    file_variable = open('forms.txt')
    all_lines_variable = file_variable.readlines()
    for line in all_lines_variable:
        line = line.split()
        ImagesDir.append(str(line[0] + ".png"))
        Y.append(line[1])
    return ImagesDir,Y
