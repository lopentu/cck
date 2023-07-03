import os
from os import path
from CbetaFile import CbetaFile
import csv



punctuation = ['！', '？', '｡', '。', '＂', '＃', '＄', '％', '＆', '＇', '（', '）', '＊', '＋', '，', '－', '／', '：', '；', '＜', '＝', '＞', '＠', '［', '＼', '］', '＾', '＿', '｀', '｛', '｜', '｝', '～', '｟', '｠', '｢', '｣', '､', '、', '〃', '》', '「', '」', '『', '』', '【', '】', '〔', '〕', '〖', '〗', '〘', '〙', '〚', '〛', '〜', '〝', '〞', '〟', '〰', '〾', '〿', '–', '—', '‘', '’', '‛', '“', '”', '„', '‟', '…', '‧', '﹏', '.']

# find xml files

def find_all_files(folder_path):
    xmls_folder_path = path.abspath(folder_path)
    all_paths = list()
    for root, dirs, files in os.walk(xmls_folder_path):
        for name in files:
            all_paths.append(path.abspath(path.join(root, name)))
    return all_paths

dyn_files = {}
def get_dynasty_map(dyn):
    if dyn not in dyn_files:
        dyn_files[dyn] = open(f'cbeta_text_{dyn}.txt', 'w')
    return dyn_files[dyn]

def close_all_dynasty_files():
    for file_x in dyn_files.values():
        file_x.close()

def preprocess(cbeta_xmls_folder):
    list_of_contexts = list()
    for (id, file) in enumerate(find_all_files(cbeta_xmls_folder)):
        cbeta_file = CbetaFile(file)
        dynasty = cbeta_file.dynasty
        if dynasty != "!!unknown!!":
            dyn_f = get_dynasty_map(dynasty)

            for headword in cbeta_file.glossary_headwords:
                dyn_f.write('\n' + clean_up(headword))
            for paragraph in cbeta_file.paragraphs:
                dyn_f.write('\n' + clean_up(paragraph))
        print(str(round(id/20190 * 100)) + '%')
    close_all_dynasty_files()

def clean_up(paragraph):
    paragraph_with_spaces = ' '.join(paragraph)
    paragraph_with_spaces_no_punct = ' '.join([character for character in paragraph_with_spaces.split() if character not in punctuation])
    return paragraph_with_spaces_no_punct



if __name__ == '__main__':
    cbeta_xmls_folder = "C:\\Users\\debor\\OneDrive\\文档\\Bookcase\\CBETA\\XML"

    preprocess(cbeta_xmls_folder)

