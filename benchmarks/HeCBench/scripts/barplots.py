import os
import sys

from sympy import plot
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pds
import json

def to_dataframe(nvidia_d, amd_d):
    benchmark_col = []
    compiler_col = []
    vendor_col = []
    exec_time_col = [] 
    for v in ["nvidia", "amd"]:
        if v == "nvidia":
            v_name = "NVIDIA"
            d = nvidia_d
        elif v == "amd":
            v_name = "AMD"
            d = amd_d
        for b in d:
            for c in d[b]:
                ts = d[b][c]["all runs"]["total app"]
                if c in ["nvcc", "hipcc"]:
                    c_name  = c
                elif c in ["clang-cuda","clang-hip"]:
                    c_name = c
                elif c in ["cudaomp", "cudaomp-amd"]:
                    c_name = "cuda-omp"
                # if c in ["nvcc", "hipcc"]:
                #     c_name  = "nvcc/hipcc"
                # elif c in ["clang-cuda","clang-hip"]:
                #     c_name = "clang-cuda/hip"
                # elif c in ["cudaomp", "cudaomp-amd"]:
                #     c_name = "clang-omp"
                # elif c  == "cudaomp-amd-32":
                #     c_name = "clang-omp-64threads"    
                for t in ts:
                    vendor_col = vendor_col + [v_name]
                    benchmark_col = benchmark_col + [b]
                    compiler_col = compiler_col + [c_name]
                    exec_time_col = exec_time_col + [t]
                    #print(v,b,c,t)
    data = {"vendor" : vendor_col, "benchmark" : benchmark_col, "compiler" : compiler_col, "exec. time (seconds)" : exec_time_col}
    df = pds.DataFrame.from_dict(data)                
    return df     


if __name__ == "__main__":

    with open('eval_results_amd.json', 'r') as f:
        amd_results_dict = json.load(f)
    with open('eval_results_nvidia.json', 'r') as f:
        nvidia_results_dict = json.load(f)    
    df = to_dataframe(nvidia_results_dict, amd_results_dict)
    
    sns.set(font_scale = 2)
    sns.set_style("whitegrid")

    benchmarks = set(df["benchmark"])
    vendors = set(df["vendor"])
    first_plot = True
    for b in benchmarks:
        df_b = df[df["benchmark"] == b] 
        for v in vendors:
            if v == "NVIDIA":
                c_order = ["nvcc","clang-cuda", "cuda-omp"]
            elif v == "AMD":
                c_order = ["hipcc","clang-hip", "cuda-omp"]
            df_b_v = df_b[df_b["vendor"] == v]      
            plot = sns.barplot(x = "compiler", order=c_order,
                    y = "exec. time (seconds)",
                    #hue="compiler", hue_order=["nvcc/hipcc","clang-cuda/hip", "clang-omp"],
                    data=df_b_v)#,
                    # capsize=.1)
            if not first_plot:
                plot.legend_.remove()
            plot.set(xlabel=None)
            plt.savefig(b +"_"+ v + "_barplot.pdf", bbox_inches='tight')
            plt.show()
    # plot = sns.catplot(x = "vendor", y = "time",
    #             hue="compiler", col="benchmark",
    #             data=df_b, kind="box",
    #             height=4, aspect=.7)

    print("plot generated")
