import os

read_path = '/home/omrijurim@staff.technion.ac.il/PycharmProjects/s2s/word_dataset_075_150/test_word.txt'
coupled = '/home/omrijurim@staff.technion.ac.il/PycharmProjects/s2s/word_dataset_075_150/test_word_coupled.txt'
with open(read_path, 'r') as read:
    with open(coupled, 'a') as out:
        for line in read:
            line_sing = line.split('_')
            line_sing[1] = 'sing'
            line_sing_str = ""
            for word in line_sing:
                line_sing_str += (word + '_')
            line_sing_str = line_sing_str[:-1]
            out.write(line)
            out.write(line_sing_str)




