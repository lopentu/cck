import os
from os import path
from CbetaFile import CbetaFile
import csv

# characters
characters_to_search = ['元', '首', '頭', '腦', '面', '顏', '額', '眉', '目', '眼', '耳', '唇', '口', '舌', '齒', '牙', '頰', '領', '項', '頸', '脰', '喉', '嚨', '咽', '嗌', '肩', '胸', '腰', '腹', '脊', '背', '心', '肺', '肝', '膽', '脾', '胃', '腎', '腸', '手', '臂', '肘', '足', '腳', '股']

def bigrams_for_character(character):
    bigrams = list()
    for friend in characters_to_search:
        if friend != character:
            bigrams.append(character+friend)
            bigrams.append((friend+character))

    return bigrams


# find xml files
cbeta_xmls_folder = "C:\\Users\\debor\\OneDrive\\文档\\Bookcase\\CBETA\\XML"

def find_all_files(folder_path):
    xmls_folder_path = path.abspath(folder_path)
    all_paths = list()
    for root, dirs, files in os.walk(xmls_folder_path):
        for name in files:
            all_paths.append(path.abspath(path.join(root, name)))
    return all_paths


def extract_onegram_contexts(character):
    list_of_contexts = list()
    for file in find_all_files(cbeta_xmls_folder):
        processed_file = CbetaFile(file)
        if processed_file.has_dynasty():
            onegram_contexts = processed_file.extract_context_of_manual(character)
            list_of_contexts.extend(onegram_contexts)
    return list_of_contexts


def extract_bigram_contexts(characters):  # does not differ from the onegram method
    list_of_contexts = list()
    for file in find_all_files(cbeta_xmls_folder):
        processed_file = CbetaFile(file)
        if processed_file.has_dynasty():
            onegram_contexts = processed_file.extract_context_of_manual(characters)
            list_of_contexts.extend(onegram_contexts)
    return list_of_contexts


if __name__ == '__main__':
    results_folder = path.abspath("C:\\Users\\debor\\OneDrive\\文档\\Bookcase\\CBETA\\results")
    fieldnames = ["context", "dynasty", "author", "file"]

    num_char = 0
    # print("Checking onegrams")
    # for character in characters_to_search:
    #     print("Looking at character: " + character + "; " + '{:.2f}%'.format(100*num_char/len(characters_to_search)))
    #     contexts = extract_onegram_contexts(character)
    #     if not path.exists(results_folder):
    #         os.makedirs(results_folder)
    #     if not path.exists(results_folder + "\\" + character):
    #         os.makedirs(results_folder + "\\" + character)
    #     with open(results_folder + "\\" + character + "\\" + character + "onegrams.tsv", 'w+') as f:
    #         writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
    #         for context in contexts:
    #             writer.writerow(context)
    #     num_char += 1

    num_char = 0
    print("Looking at bigrams")
    for char1 in characters_to_search:
        for char2 in characters_to_search:
            character = char1 + char2
            print("Looking at character: " + character + "; " + '{:.2f}%'.format(100*num_char / len(characters_to_search)**2))
            contexts = extract_onegram_contexts(character)
            if not path.exists(results_folder):
                os.makedirs(results_folder)
            if not path.exists(results_folder + "\\" + char1):
                os.makedirs(results_folder + "\\" + char1)
            if not path.exists(results_folder + "\\" + char2):
                os.makedirs(results_folder + "\\" + char2)
            with open(results_folder + "\\" + char1 + "\\" + char1 + "bigrams.tsv", 'a+') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
                for context in contexts:
                    writer.writerow(context)
            if char1 != char2:
                with open(results_folder + "\\" + char2 + "\\" + char2 + "bigrams.tsv", 'a+') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
                    for context in contexts:
                        writer.writerow(context)
            num_char += 1
    #with open()