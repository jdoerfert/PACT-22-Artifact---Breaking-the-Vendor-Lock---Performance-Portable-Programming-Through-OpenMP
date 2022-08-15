import os
import sys
import subprocess as sp
import re
import statistics
import json
import glob

from numpy import str_

# evaluation parameters
benchmarks =["xsbench", "miniFE", "su3", "xsbench", "rsbench", "lulesh", "triad"]
amd_compilers = ["hipcc", "clang-hip", "cudaomp-amd"]
nvidia_compilers = ["nvcc", "clang-cuda", "cudaomp"] #, "omp-offload"]
num_runs = 10 

# redirection of compilation output
compilation_output = sp.DEVNULL
compilation_err_output = None

#cudaomp compilation and linking flags
common_args = "--save-temps -x cu -O3 -fopenmp -foffload-lto -cudaomp -fopenmp-new-driver -fgpu-rdc -fopenmp-device-libm -lomptarget"
cudaomp_amd_args = "--offload=amdgcn-amd-amdhsa --offload-arch=gfx906 -nocudainc -nocudalib -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false" + " " + common_args
cudaomp_nvidia_args = "--offload-arch=sm_70" + " " + common_args

#global dictionary for evaluation results
results = {}

#internal filenames etc.
results_filename_local = 'run_results.txt'

# benchmark-specifics
sourcecode_filenames = {
    "xsbench" : "GridInit.cu  io.cu  Main.cu  Materials.cu  Simulation.cu  XSutils.cu",
    "rsbench" : "init.cu  io.cu  main.cu  material.cu  simulation.cu  utils.cu",
    "su3" : "su3_nn_bench.cu",
    "lulesh" : "lulesh-init.cu  lulesh-util.cu  lulesh-viz.cu  lulesh.cu",
    "miniFE" : "YAML_Doc.cpp  YAML_Element.cpp  main.cpp ../utils/BoxPartition.cpp ../utils/mytimer.cpp ../utils/param_utils.cpp ../utils/utils.cpp",
    "triad" : "triad.cu main.cpp  Option.cpp  OptionParser.cpp  Timer.cpp"
}

includes = {
    "xsbench" : "",
    "rsbench" : "",
    "su3" : "",
    "lulesh" : "",
    "miniFE" : "-I. -I../utils -I../fem ",
    "triad" : "-I. "
}

flags = {
    "xsbench" : "",
    "rsbench" : "",
    "su3" : "-DUSE_CUDA -DMILC_COMPLEX ",
    "lulesh" : "",
    "miniFE" : "-DMINIFE_CSR_MATRIX \
-DMINIFE_SCALAR=double   \
-DMINIFE_LOCAL_ORDINAL=int \
-DMINIFE_GLOBAL_ORDINAL=int \
-DMINIFE_RESTRICT=__restrict__ ",
    "triad" : ""
}

exec_name = {
    "rsbench" : "rsbench",
    "xsbench" : "xsbench",
    "su3" : "main",
    "lulesh" : "lulesh",
    "miniFE" : "miniFE.x",
    "triad" : "triad"
}

exec_cmd = {
    "rsbench" : "numactl --membind=0 " + exec_name["rsbench"] + " -m event -s large",
    "xsbench" : "numactl --membind=8 -C 8 " + exec_name["xsbench"] + " -m event -s large",
    "su3" : "numactl --membind=0 " + exec_name["su3"] + " -i 100 -l 32 -t 128 -v 3 -w 1",
    "lulesh" : "numactl --membind=8 -C 8 " + exec_name["lulesh"]  + " -i 100 -s 128 -r 11 -b 1 -c 1",
    "miniFE" : "numactl --membind=8 -C 8 " + exec_name["miniFE"]  + " -nx 128 -ny 128 -nz 128",
    "triad" : "numactl --membind=0 " + exec_name["triad"]  + " --passes 100"
}
dir_suffix = {"hipcc" : "hip", "clang-hip" : "hip", "cudaomp-amd" : "cuda", "cudaomp-amd-64" : "cuda", "nvcc" : "cuda", "clang-cuda" : "cuda", "cudaomp" : "cuda", "omp-offload" : "omp"}



def measure(benchmark_name : str, dataset_name : str):
    if benchmark_name == "rsbench" or benchmark_name == "xsbench":
        measure_rs_or_xs_bench(benchmark_name, dataset_name)
    elif benchmark_name == "su3":
        measure_su3(dataset_name)
    elif benchmark_name == "lulesh":
        measure_lulesh(dataset_name) 
    elif benchmark_name == "miniFE":
        measure_miniFE(dataset_name) 
    elif benchmark_name == "triad":
        measure_triad(dataset_name)    
    else:
        assert False, "unknown benchmark name"      

def measure_triad(dataset_name : str):
    total_times_own = []
    for i in range(1, (num_runs + 1)):
        print("Beginning measurement #", i, " of ", "triad", " using ", dataset_name)
        with open(results_filename_local, "w") as outf:
            print("CMD: ", exec_cmd["triad"])
            sp.run(exec_cmd["triad"].split(" "), stdout=outf)
        #extract exec time
        with open(results_filename_local, "r") as resf:
            for line in resf:
                if "Total execution time" in line:
                    time = (line.split(" "))[-2]
                    time = float(time) / 1000 # require seconds, input is milliseconds
                    total_times_own = total_times_own + [time]
                    break 
    #store data and median  
    results["triad"][dataset_name]["medians"] = {}
    results["triad"][dataset_name]["all runs"] = {}
    results["triad"][dataset_name]["medians"]["total own"] = statistics.median(total_times_own)
    results["triad"][dataset_name]["all runs"]["total own"] = total_times_own

def measure_miniFE(dataset_name : str):
    total_times_app = []
    for i in range(1, (num_runs + 1)):
        print("Beginning measurement #", i, " of ", "miniFE", " using ", dataset_name)
        with open(results_filename_local, "w") as outf:
            sp.run(exec_cmd["miniFE"].split(" "), stdout=outf)
        #extract exec time
        resf_name = glob.glob('*.yaml')[0] 
        with open(resf_name, "r") as resf:
            for line in resf:
                if "Total Program Time:" in line:
                    time = (line.split(" "))[-1]
                    total_times_app = total_times_app + [float(time)]
                    break 
    #store data and median  
    results["miniFE"][dataset_name]["medians"] = {}
    results["miniFE"][dataset_name]["all runs"] = {}
    results["miniFE"][dataset_name]["medians"]["total app"] = statistics.median(total_times_app)
    results["miniFE"][dataset_name]["all runs"]["total app"] = total_times_app

def measure_lulesh(dataset_name : str):
    total_times_app = []
    for i in range(1, (num_runs + 1)):
        print("Beginning measurement #", i, " of ", "lulesh", " using ", dataset_name)
        with open(results_filename_local, "w") as outf:
            lulesh_cmd = exec_cmd["lulesh"]
            if dataset_name in ["hipcc", "clang-hip", "cudaomp-amd"]:
                lulesh_cmd = "numactl --membind=0 --physcpubind 0 -C 0" + lulesh_cmd[19:]
            print(lulesh_cmd)
            sp.run(lulesh_cmd.split(" "), stdout=outf)
        #extract exec time
        with open(results_filename_local, "r") as resf:
            for line in resf:
                if "Elapsed time" in line:
                    time = (line.split(" "))[-2]
                    total_times_app = total_times_app + [float(time)]
                    break 

    #store data and median  
    results["lulesh"][dataset_name]["medians"] = {}
    results["lulesh"][dataset_name]["all runs"] = {}
    results["lulesh"][dataset_name]["medians"]["total app"] = statistics.median(total_times_app)
    results["lulesh"][dataset_name]["all runs"]["total app"] = total_times_app

def measure_su3(dataset_name : str):
    total_times_app = []
    for i in range(1, (num_runs + 1)):
        print("Beginning measurement #", i, " of ", "su3", " using ", dataset_name)
        with open(results_filename_local, "w") as outf:
            sp.run(exec_cmd["su3"].split(" "), stdout=outf)
        #extract exec time
        with open(results_filename_local, "r") as resf:
            for line in resf:
                if "Total execution time =" in line:
                    time = (line.split(" "))[-2]
                    total_times_app = total_times_app + [float(time)]
                    break 

    #store data and median  
    results["su3"][dataset_name]["medians"] = {}
    results["su3"][dataset_name]["all runs"] = {}
    results["su3"][dataset_name]["medians"]["total app"] = statistics.median(total_times_app)
    results["su3"][dataset_name]["all runs"]["total app"] = total_times_app

def measure_rs_or_xs_bench(benchmark_name: str, dataset_name : str):
    total_times_app = []
    if dataset_name != 'omp-offload': # no kernel time reporting in omp-offload version
        kernel_only_times_app = []
    for i in range(1, (num_runs + 1)):
        print("Beginning measurement #", i, " of ", benchmark_name, " using ", dataset_name)
        with open(results_filename_local, "w") as outf:
            print(os.getcwd())
            print(exec_cmd[benchmark_name])
            sp.run(exec_cmd[benchmark_name].split(" "), stdout=outf)
        #extract exec time
        with open(results_filename_local, "r") as resf:
            for line in resf:
                if dataset_name == 'omp-offload': # different exec. time reporting
                    if "Runtime:" in line:
                        total_times_app = total_times_app + [float(line.split(" ")[-2])]
                else:
                    if "Total Time Statistics" in line:
                        nline = next(resf)
                        total_times_app = total_times_app + [float(nline.split(" ")[-2])]
                    if "Kernel Only" in line:
                        nline = next(resf)
                        kernel_only_times_app = kernel_only_times_app + [float(nline.split(" ")[-2])]    

    #store data and medians    

    results[benchmark_name][dataset_name]["medians"] = {}
    results[benchmark_name][dataset_name]["all runs"] = {}
    results[benchmark_name][dataset_name]["medians"]["total app"] = statistics.median(total_times_app)
    if dataset_name != 'omp-offload': # no kernel time reporting in omp-offload version
        results[benchmark_name][dataset_name]["medians"]["kernel app"] = statistics.median(kernel_only_times_app)
    results[benchmark_name][dataset_name]["all runs"]["total app"] = total_times_app
    if dataset_name != 'omp-offload': # no kernel time reporting in omp-offload version
        results[benchmark_name][dataset_name]["all runs"]["kernel app"] = kernel_only_times_app


if __name__ == "__main__":

    original_dir = os.getcwd()

    #path to parent of benchmark directory
    eval_dir = sys.argv[1] 

    #nvidia or amd gpu
    mode = sys.argv[2]

    # init mode specifics
    if mode == "amd":
        compilers = amd_compilers
        cudaomp_args = cudaomp_amd_args
    elif mode == "nvidia":
        compilers = nvidia_compilers
        cudaomp_args = cudaomp_nvidia_args
    else:
        assert False, "unrecognized device vendor"


    top_dir = None
    # gather performance data
    for b in benchmarks:
        results[b] = {}
        for c in compilers:
            results[b][c] = {}
            print("########################################################")
            print("benchmark: ", b, "   ", "compiler:", c)
            print("########################################################")
            suffix = dir_suffix[c]
            prefix = "HeCBench/" + b + "-" + suffix + "/"
            if b == "miniFE":
                prefix = prefix + "src/"
            #go to directory  
            if top_dir:
                os.chdir(top_dir + '/' + prefix)
            else:
                os.chdir(eval_dir)
                top_dir = os.getcwd() 
                os.chdir(prefix)   
            #print(os.getcwd())
            if True: # recompile each time, currently shared directories between different compilers that are evaluated 
                print("Recompiling ", b, " with ", c)
                #make clean
                if b == "miniFE":
                    sp.run("make realclean".split(" "))  
                else: 
                    sp.run("make clean".split(" "))  
                #compile and link
                if c == "nvcc" or c == "hipcc" or c == "omp-offload":
                    sp.run("make -f Makefile".split(" "), stdout=compilation_output, stderr=compilation_err_output) 
                elif c == "clang-cuda":        
                    sp.run("make -f Makefile.cuda".split(" "), stdout=compilation_output, stderr=compilation_err_output)      
                elif c == "clang-hip":        
                    sp.run("make -f Makefile.clanghip".split(" "), stdout=compilation_output, stderr=compilation_err_output)
                elif c == "cudaomp" or c == "cudaomp-amd" or c == "cudaomp-amd-64":
                    cmd = "clang++ -o " + exec_name[b] + " " + cudaomp_args + " " + sourcecode_filenames[b] + " " + includes[b] + flags[b]
                    print (cmd)    
                    sp.run(cmd.split(), stdout=compilation_output, stderr=compilation_err_output) 
                else:
                    assert False, "Unrecognized compiler name"  
            #run and collect data
            measure(b, c)
            #go back to top dir
            os.chdir(top_dir) 

    

    os.chdir(original_dir)
    # output results on command line (optional)
    if False:
        print(results)

    with open(f"eval_results_{mode}.json", "w") as results_file:
        json.dump(results, results_file, indent=4, sort_keys=True)

    print(f"evaluation script finished. Results written to file \'eval_results_{mode}.json\'.")
