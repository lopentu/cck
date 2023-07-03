import pandas as pd
from os import path

characters_to_search = ['元', '首', '頭', '腦', '面', '顏', '額', '眉', '目', '眼', '耳', '唇', '口', '舌', '齒', '牙', '頰', '領', '項', '頸', '脰', '喉', '嚨', '咽', '嗌', '肩', '胸', '腰', '腹', '脊', '背', '心', '肺', '肝', '膽', '脾', '胃', '腎', '腸', '手', '臂', '肘', '足', '腳', '股']

results_folder = "C:\\Users\\debor\\PycharmProjects\\extractContexts\\results"

matrix_df = pd.DataFrame(0, columns=characters_to_search, index=characters_to_search, dtype=int)

total_df = pd.DataFrame(0, columns=['occurrences', 'co-occurrences with another body part', 'ratio'], index=characters_to_search, dtype=float)

for c in characters_to_search:

    with open(results_folder + "\\" + c + "\\" + c + "onegrams.tsv") as f:
        line_list = [line.strip() for line in f if line.strip()]
        number_of_lines = len(line_list)
        total_df.loc[c, 'occurrences'] += number_of_lines
        if total_df.loc[c, 'occurrences'] == 0:
            print('AAAAAHHHHHH!!')

    bigram_file = path.abspath(results_folder + "\\" + c + "\\" + c + "bigrams_with_target")
    bigrams_df = pd.read_csv(bigram_file, delimiter="\t")
    for i, row in bigrams_df.iterrows():
        target = row['target']
        c1 = target[0]
        c2 = target[1]
        # c1: column
        if c == c1:
            matrix_df.loc[c2, c1] += 1
            if c1 == c2:
                total_df.loc[c1, 'co-occurrences with another body part'] -= 1


for c in characters_to_search:
    total_df.loc[c, 'co-occurrences with another body part'] += matrix_df.loc[c, :].sum() + matrix_df.loc[:, c].sum()
    total_df.loc[c, 'ratio'] = total_df.loc[c, 'co-occurrences with another body part'] / total_df.loc[c, 'occurrences']

matrix_df.to_csv(results_folder + "\\cooccurrence.csv")
total_df.to_csv(results_folder + "\\total.csv")




