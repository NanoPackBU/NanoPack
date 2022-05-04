import glob
import matplotlib.pyplot as plt 
from skimage import io
import numpy as np
import csv
import os

def read_data():
    
    # get image filenames
    zero_locs = glob.glob('DigitsGreen/0/*.png')
    one_locs = glob.glob('DigitsGreen/1/*.png')
    two_locs = glob.glob('DigitsGreen/2/*.png')
    three_locs = glob.glob('DigitsGreen/3/*.png')
    four_locs = glob.glob('DigitsGreen/4/*.png')
    five_locs = glob.glob('DigitsGreen/5/*.png')
    six_locs = glob.glob('DigitsGreen/6/*.png')
    seven_locs = glob.glob('DigitsGreen/7/*.png')
    eight_locs = glob.glob('DigitsGreen/8/*.png')
    nine_locs = glob.glob('DigitsGreen/9/*.png')
    locs_array = [zero_locs, one_locs, two_locs, three_locs, four_locs, five_locs, six_locs, seven_locs, eight_locs, nine_locs]

    num_zeros = len(zero_locs)
    num_ones = len(one_locs)
    num_twos = len(two_locs)
    num_threes = len(three_locs)
    num_fours = len(four_locs)
    num_fives = len(five_locs)
    num_sixes = len(six_locs)
    num_sevens = len(seven_locs)
    num_eights = len(eight_locs)
    num_nines = len(nine_locs)
    nums_array = [num_zeros, num_ones, num_twos, num_threes, num_fours, num_fives, num_sixes, num_sevens, num_eights, num_nines]


    with open('green_digit_labels.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        for num in range(10):
            for i in range(nums_array[num]):
                img = locs_array[num][i]
                data = [os.path.basename(img), num]
                writer.writerow(data)

if __name__ == "__main__":
    read_data()
