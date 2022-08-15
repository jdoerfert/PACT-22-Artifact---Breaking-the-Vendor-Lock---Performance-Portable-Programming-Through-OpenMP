import sys
from pathlib import Path
import json
import re
import os
import subprocess
import pandas as pd

def execute_command(cmd, bench_dir):
  p = subprocess.run( cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, cwd=bench_dir)
  out = str(p.stdout.decode('utf-8'))
  err = str(p.stderr.decode('utf-8'))
  return p.returncode, out,err

def analyze_errors(err, error_vals):
  error_codes = re.compile('error: use of undeclared identifier \'(.*?)\'')
  vals = re.findall('error: use of undeclared identifier \'(.*?)\'', err)
  for v in vals:
    if v not in error_vals:
      error_vals[v] = 1
    else:
      error_vals[v] += 1
  return error_vals

#module load clang/13.0.0;
commands={'cuda': 'module unload clang/13.0.0; source ../env.sh ; make clean; make -f Makefile.cuda',
          'cudaomp': 'module unload clang/13.0.0; source ../env.sh ; make clean; make -f Makefile.cudaomp'
         }
status = {}
error_vals = {}
directories =  []
dbStatusFile = Path(f'{sys.argv[1]}')
if dbStatusFile.is_file():
  with open(dbStatusFile,'r') as fd:
    status = json.load(fd)
  df = pd.DataFrame.from_dict(status, orient='index')
  roi = df[(df['cudaomp'] == 'Fail') & (df['cuda'] == 'Success')]
  for v in roi.index.tolist():
    directories.append(f'./{v}-cuda/')
else:
  with open('processed.txt', 'r') as fd:
    directories = fd.readlines()

for d in directories:
  bench_dir = d.strip('\n')
  name = bench_dir.replace('/', '').replace('.', '').replace('-cuda', '')
  status[name] = {}
  for c, v in commands.items():
    rc, out, err = execute_command(v, bench_dir)
    if rc == 0:
      status[name][c] = 'Success'
    else:
      if c == 'cudaomp':
        error_vals = analyze_errors(err, error_vals)
      status[name][c] = 'Fail'
  print(name, status[name])

df = pd.DataFrame.from_dict(status, orient='index')
all_data = {}
all_data['benches'] = status
all_data['errors'] = error_vals

with open(sys.argv[1],'w') as fd:
  json.dump(status, fd, indent=6, ensure_ascii=False)

with open("error_vals.json",'w') as fd:
  json.dump(error_vals, fd, indent=6, ensure_ascii=False)

print(error_vals)

