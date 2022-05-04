import pandas as pd

# Parse CSV as dictionary of lists
def parse_dl(filename):
    data_frame = pd.read_csv(filename, encoding='utf-8-sig')
    outputDL = {}
    # ind = "ind"
    label1 = ""
    label2 = ""
    label3 = ""
    for index, row in data_frame.iterrows():
        if index > 0:
            r1, r2, r3 = (row[0], row[1], row[2])
            if str(r1) == "nan" and str(r2) == "nan" and str(r3) == "nan":
                return outputDL
            outputDL[label1].append(r1)
            outputDL[label2].append(r2)
            outputDL[label3].append(r3)
            continue
        label1 = row[0]
        label2 = row["Unnamed: 1"]
        label3 = row["Unnamed: 2"]
        outputDL[label1] = []
        outputDL[label2] = []
        outputDL[label3] = []

    return outputDL

# Parse csv as LoD and split up the clamshell information
def parse_csv(filename):
    # Parse as list of dictionaries instead
    outputDL = parse_dl(filename)
    outputLD = [dict(zip(outputDL, t)) for t in zip(*outputDL.values())]

    # Split clamshell info
    for currDict in outputLD:
        clamshellPos = currDict.pop('Position In Chip Package')

        # Convert clamshell letter to number (works for A-Z)
        clamshellLetter = "2" + str(chr(ord(clamshellPos[0]) - 37))
        if clamshellPos[0] < 'K':
            clamshellLetter = str(chr(ord(clamshellPos[0]) - 17))
        elif clamshellPos[0] < 'U':
            clamshellLetter = "1" + str(chr(ord(clamshellPos[0]) - 27))

        currDict['Chip Package Number'] = int(clamshellLetter)
        currDict['Position In Chip Package'] = int(clamshellPos[1])

        currDict['Position on Chip Traveler'] = int(currDict['Position on Chip Traveler'])
        currDict['Chip ID'] = int(currDict['Chip ID'])

    return outputLD


# Not super efficient but eh N=8
def find_chip_order(outputLD):
    outputDict = []
    for r in range(8):
        for c in reversed(range(8)):
            for currDict in outputLD:
                if int(currDict['Position on Chip Traveler']) == (r + c * 8):
                    outputDict.append(currDict)
    return outputDict


if __name__ == '__main__':
    output = parse_csv('Quicktest.csv')
    print(output)
    print(".........................")
    print(find_chip_order(output))
