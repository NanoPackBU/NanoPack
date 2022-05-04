import os
import sys

ROOT = "NanoView_G33"


def path_char():  # this method determines the path char
    is_wind = sys.platform.startswith('win')
    if is_wind:
        return "\\"
    return "/"


def get_root():  # this gets the root, by looking up the file tree fro the NanoView_G33 folder
    proj_folder = os.getcwd()
    rootToG33 = proj_folder[:proj_folder.find(ROOT)] + ROOT + path_char()
    return rootToG33


def make_path(folder_list=[],
              paths_string=""):  # list of folder from the first folder in the nano view g 33 dir or a regular path
    if len(folder_list) > 0:
        start = get_root()
        for i in folder_list:
            if len(i) > 0:
                start += i
                if "." in i[1:]:
                    return start
                if not (i[-1] == path_char()): start += path_char()
                continue
            start += path_char()
        return start
    if len(paths_string) > 0:
        is_wind = sys.platform.startswith('win')
        list_of_paths = []
        if is_wind:
            list_of_paths = paths_string.split("/")
        else:
            list_of_paths = paths_string.split("\\")
        accumilator = ""
        for i in list_of_paths:
            if len(i) > 0:
                accumilator += i
                if "." in i[1:]:
                    return accumilator

                if not (i[-1] == path_char()): accumilator += path_char()
                continue
            accumilator += path_char()
        return accumilator
    return "Error"
