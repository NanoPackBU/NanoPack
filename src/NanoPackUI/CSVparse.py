# import csv
import pandas as pd


def parseCSV(filename):
    # inputDict = csv.DictReader(open(filename, newline='',mode='r', encoding='utf-8-sig'))
    datafram = pd.read_csv(filename, encoding='utf-8-sig')
    csvDict = datafram.to_dict(orient='list')
    return csvDict

# if __name__ == '__main__':
#    parseCSV('first_prototype_config.csv')
