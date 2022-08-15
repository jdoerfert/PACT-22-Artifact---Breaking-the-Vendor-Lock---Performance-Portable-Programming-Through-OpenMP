# [PACT’22] Artifact: Breaking the Vendor Lock --- Performance Portable Programming Through OpenMP as Target Independent Runtime Layer

## Instructions:
The following contains information on how to repeat the measurements described in the evaluation section of our paper.
### Brief command-line instructions
```
./build-scripts/setup.sh
python3 benchmarks/run_eval.sh ./benchmarks <amd/nvidia>
vim eval_results_<amd/nvidia>.json
python3 benchmarks plot_results.sh <amd/nvidia/both>
open <lulesh/miniFE/rsbench/su3/triad/xsbench>.pdf
```

### Additional instructions:
The script `eval_results.py` features global evaluation parameters that can be modified at the top of the file such as the list of compilers, benchmarks, and number of measurements for the same configuration. Either the entire list of supported options is shown, or if not set by default, then added as an inline comment.
For a plot that contains execution time results for both Nvidia and AMD GPUs, one usually needs to perform the setup and evaluation steps on multiple machines and then gather the `eval_results_*.json` files so that the plot_results.py script can include both vendors’ GPUs in a single plot.

### Installation and Dependencies:
The ‘build-scripts’ directory contains a setup script that compiles and installs Clang/LLVM with our custom extensions. 
Dependencies include the common ones for LLVM. Please note that we use the build tool ‘ninja’ in our ‘build.sh’ script, therefore this is a dependency.

## Directory overview:
* `benchmarks`:
  * HeCBench with custom Makefiles (forked from the [HeCBench](https://github.com/zjin-lcf/HeCBench) GitHub repo, commit [7d046c4d2d2ab9a6897d5727507048c8ec17dda6](https://github.com/zjin-lcf/HeCBench/commit/7d046c4d2d2ab9a6897d5727507048c8ec17dda6))
  * a script `run_eval.py` that performs measurements on one machine and emits a JSON file with measured execution times
  * a script `plot_eval.py` that plots results based on the data produced by the `run_eval.py` script
* `build-scripts`: utility scripts for a convenient installation of LLVM
* `samples`: Contains exemplary measurement results and corresponding plots generated based on the above instructions
* `sources`: Source files of Clang/LLVM with our custom extensions for supporting the retargeting of GPU codes. 
(to be precise, this is a fork of the [llvm-project](https://github.com/llvm/llvm-project) GitHub repository, commit [ce87133120685cc61b60fe3f1545cbbfdaa59805](https://github.com/llvm/llvm-project/commit/ce87133120685cc61b60fe3f1545cbbfdaa59805), on top of which our extensions for retargeting have been implemented)

## Installation and Dependencies:
The `build-scripts` directory contains a setup script that compiles and installs Clang/LLVM with our custom extensions. Dependencies include the common ones for LLVM. Please note that we use the build tool `ninja` in our `build.sh` script, therefore this is a dependency.

The python scripts have their own dependencies as noticeable in the import lists at the top of each file. For the plotting script, `seaborn` and `matplotlib` are required for instance.

 
