import sys
import subprocess

mag = int(sys.argv[1])

for seed in range(10, 50):
    fname = f"wq_{mag}_{seed}.yaml"
    with open(fname, "w") as fp:
        fp.write(f"""\
N: 1
mode: by_node
command: |
  source ~/.bashrc
  conda activate des-y6-imsims
  echo `which python`
  ./run_sims.sh {mag} {seed} &> log{mag}_{seed}.oe
""")
    subprocess.run(
        f"wq sub -b {fname}",
        shell=True,
        check=True,
    )
