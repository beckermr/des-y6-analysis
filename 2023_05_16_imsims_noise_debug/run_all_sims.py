import os
import subprocess

run = "001"
mag = 23

seed_min = 45

tnames = []
with open("tiles.txt", "r") as fp:
    for line in fp.readlines():
        tnames.append(line.strip())

for seed_base, tname in enumerate(tnames):
    seed = seed_base + seed_min
    fname = f"run{run}/wq_{tname}_{seed}.yaml"
    bname = os.path.basename(fname)
    with open(fname, "w") as fp:
        fp.write(f"""\
N: 1
mode: by_node
priority: low
command: |
  source ~/.bashrc
  conda activate des-y6-imsims
  echo `which python`
  ../run_sims.sh {run} {tname} {seed} {mag} &> log_{tname}_{seed}.oe
""")
    subprocess.run(
        f"wq sub -b {bname}",
        shell=True,
        check=True,
        cwd=f"run{run}",
    )
