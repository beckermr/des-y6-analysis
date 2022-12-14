import os

with open("./badones.txt") as fp:
    for line in fp.readlines():
        fname = line.split(" ")[0]
        print("removing:", fname)
        os.system("rm -f %s" % fname)
