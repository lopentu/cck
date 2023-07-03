import os
from os import path
from CbetaFile import CbetaFile
import csv
import pandas as pd

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

results_folder = "C:\\Users\\debor\\PycharmProjects\\extractContexts\\results"


def bigrams_other_ordering(c):
    bigrams = []
    for c1 in characters_to_search:
        if c1 == c:
            for c2 in characters_to_search:
                bigrams.append(c+c2)
        else:
            bigrams.append(c1+c)
    return bigrams


for j, c in enumerate(characters_to_search):
    print("Fixing", c, ',', j, "of", len(characters_to_search))
    bigrams = [c+char2 for char2 in characters_to_search]+[char2+c for char2 in characters_to_search]
    bigrams = bigrams_other_ordering(c)
    # bigrams_for_character(c) (wrong order)
    bigram_file = path.abspath(results_folder + "\\" + c + "\\" + c + "bigrams.tsv")
    print(os.path.isfile(bigram_file))
    df = pd.read_csv(bigram_file, delimiter="\t", names=['context', 'dynasty', 'author', 'file_name'])

    df['target'] = 'test'
    #add target column
    currentBigram = bigrams.pop(0)
    for i, row in df.iterrows():
        found = False
        while not found:
            char_pos = row['context'].find(currentBigram)
            while char_pos < 0:
                try:
                    currentBigram = bigrams.pop(0)
                except:
                    print(currentBigram)
                    print(row)
                char_pos = row['context'].find(currentBigram)
            while char_pos >= 0 and CbetaFile.extend_context(char_pos, row['context'], size=10, len_char=2) != row['context']:
                char_pos = row['context'].find(currentBigram, char_pos+1)
            if char_pos >= 0 and CbetaFile.extend_context(char_pos, row['context'], size=10, len_char=2) == row['context']:
                row['target'] = currentBigram
                found = True
            else:
                currentBigram = bigrams.pop(0)




    #move target column to the left
    columns = df.columns.tolist()
    columns = columns[-1:] + columns[:-1]
    df = df[columns]
    df.to_csv(results_folder + '\\' + c + '\\' + c + 'bigrams_with_target', sep="\t")



def find_all_bigram_files(folder_path):
    xmls_folder_path = path.abspath(folder_path)
    all_paths = list()
    for root, dirs, files in os.walk(xmls_folder_path):
        for name in files:
            all_paths.append(path.abspath(path.join(root, name)))
    return all_paths



#
# if __name__ == '__main__':
#     results_folder = path.abspath("C:\\Users\\debor\\OneDrive\\文档\\Bookcase\\CBETA\\results")
#     fieldnames = ["context", "dynasty", "author", "file"]
#
#     num_char = 0
#     # print("Checking onegrams")
#     # for character in characters_to_search:
#     #     print("Looking at character: " + character + "; " + '{:.2f}%'.format(100*num_char/len(characters_to_search)))
#     #     contexts = extract_onegram_contexts(character)
#     #     if not path.exists(results_folder):
#     #         os.makedirs(results_folder)
#     #     if not path.exists(results_folder + "\\" + character):
#     #         os.makedirs(results_folder + "\\" + character)
#     #     with open(results_folder + "\\" + character + "\\" + character + "onegrams.tsv", 'w+') as f:
#     #         writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
#     #         for context in contexts:
#     #             writer.writerow(context)
#     #     num_char += 1
#
#     num_char = 0
#     print("Looking at bigrams")
#     for char1 in characters_to_search:
#         for char2 in characters_to_search:
#             character = char1 + char2
#             print("Looking at character: " + character + "; " + '{:.2f}%'.format(100*num_char / len(characters_to_search)**2))
#             contexts = extract_onegram_contexts(character)
#             if not path.exists(results_folder):
#                 os.makedirs(results_folder)
#             if not path.exists(results_folder + "\\" + char1):
#                 os.makedirs(results_folder + "\\" + char1)
#             if not path.exists(results_folder + "\\" + char2):
#                 os.makedirs(results_folder + "\\" + char2)
#             with open(results_folder + "\\" + char1 + "\\" + char1 + "bigrams.tsv", 'a+') as f:
#                 writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
#                 for context in contexts:
#                     writer.writerow(context)
#             if char1 != char2:
#                 with open(results_folder + "\\" + char2 + "\\" + char2 + "bigrams.tsv", 'a+') as f:
#                     writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
#                     for context in contexts:
#                         writer.writerow(context)
#             num_char += 1
#     #with open()