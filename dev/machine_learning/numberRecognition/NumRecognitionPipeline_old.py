import glob

from FindChipsAndNums import find_single_chip_nums as GetNums
from ExtractDigits import extract_single_digit as GetDigit
from DatasetBuilder import create_new_test_loader as BuildTestSet
from DigitRecognition import test

## def used to pipeline the chip pipeline 
def numberRecognition(folder_1_files, folder_2, folder_2_files, folder_3):
    chip_count = 1
    f1_locs = glob.glob(folder_1_files)
    for loc1 in f1_locs:
        GetNums(loc1, folder_2, chip_count)

        f2_locs = glob.glob(folder_2_files)
        for loc2 in f2_locs:
            GetDigit(loc2, folder_3, 1)
            # f3_locs = glob.glob(folder_3_files)
            # for loc3 in f3_locs:
            testloader = BuildTestSet(folder_3)
            accuracy = test(test_loader=testloader)
            print('the test accuracy over the whole test set is %d %%' % (accuracy))



if __name__ == "__main__":
    numberRecognition("pipeline_1/*.jpg", "pipeline_2", "pipeline_2/*.png", "pipeline_3")
