import pathlib
import shlex
import subprocess

root = pathlib.Path('/home/john/Strata')
build = root / 'build-tensor-parralize-baseline'
commands = subprocess.check_output([str(root / '.venv/bin/ninja'), '-C', str(build), '-t', 'commands', 'split_moe_iq_test'], text=True).splitlines()
compile_args = shlex.split(commands[-2])
source = build / 'iq_ilp_scratch.cu'
obj = build / 'iq_ilp_scratch.o'
exe = build / 'iq_ilp_scratch'
for key, value in [('-MT', obj), ('-MF', str(obj) + '.d'), ('-o', obj)]:
    compile_args[compile_args.index(key) + 1] = str(value)
compile_args[-1] = str(source)
link_args = shlex.split(commands[-1])[2:-2]
old_obj = 'src/split/CMakeFiles/split_moe_iq_test.dir/tools/moe_iq_test.cu.o'
link_args[link_args.index(old_obj)] = str(obj)
link_args[link_args.index('-o') + 1] = str(exe)
with (build / 'iq-ilp-build.log').open('w') as log:
    subprocess.run(compile_args, cwd=build, stdout=log, stderr=subprocess.STDOUT, check=True)
    subprocess.run(link_args, cwd=build, stdout=log, stderr=subprocess.STDOUT, check=True)
print(exe)
