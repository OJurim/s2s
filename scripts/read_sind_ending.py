import os
import sys
import shutil



in_file_path = sys.argv[1]
out_file_path = sys.argv[2]
out_dir = sys.argv[3]

out_fp = open(out_file_path, "a")

with open(in_file_path, "r") as in_f:
    while True:
        line_read = in_f.readline()

        if not line_read:
            break

        line_read = line_read[:-1]
        if "_read" in line_read:
            # src = "/home/ohadmochly@staff.technion.ac.il/NUS_word_dir/combined_words/" + line_sing
            dst = os.path.abspath(out_dir + line_read)
            line_read_new = line_read.replace("_read", "")
            line_read_new = line_read_new.replace(".wav", "_read.wav\n")
            if os.path.isfile(dst):
                # shutil.copyfile(src, dst)
                new_name = os.path.abspath(out_dir+line_read_new)
                os.rename(dst, new_name)
            elif os.path.isfile(os.path.abspath(out_dir + line_read_new)):
                no_n = os.path.abspath(out_dir + line_read_new.replace(".wav\n", ".wav"))
                os.rename(os.path.abspath(out_dir + line_read_new), no_n)

        line_sing = line_read.replace("_read", "_sing")
        if "_sing" in line_sing:
            dst = os.path.abspath(out_dir + line_sing)
            line_sing_new = line_sing.replace("_sing", "")
            line_sing_new = line_sing_new.replace(".wav", "_sing.wav\n")
            if os.path.isfile(dst):
                new_name = os.path.abspath(out_dir + line_sing_new)
                os.rename(dst, new_name)
            elif os.path.isfile(os.path.abspath(out_dir + line_sing_new)):
                no_n = os.path.abspath(out_dir + line_sing_new.replace(".wav\n", ".wav"))
                os.rename(os.path.abspath(out_dir + line_sing_new), no_n)

        out_fp.write(line_read_new)

out_fp.close()
