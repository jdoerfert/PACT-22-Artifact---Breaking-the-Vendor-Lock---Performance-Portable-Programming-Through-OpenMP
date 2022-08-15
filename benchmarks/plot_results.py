import os
import sys

#from sympy import plot
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pds
import json

NVIDIA_MACHINE = 'Power9 + V100'
AMD_MACHINE = 'AMD + MI50'


def to_dataframe(nvidia_d, amd_d, AMD=False, NVIDIA=False):
    benchmark_col = []
    compiler_col = []
    vendor_col = []
    exec_time_col = []
    time_omp = {}
    time_cudaomp = {}
    vendors = []
    if AMD:
        vendors = vendors + ["amd"]
    if NVIDIA:
        vendors = vendors + ["nvidia"]    
    for v in vendors:
        if v == "nvidia":
            v_name = "NVIDIA"
            d = nvidia_d
        elif v == "amd":
            v_name = "AMD"
            d = amd_d
        for b in d:
            time_omp[b] = {}
            time_cudaomp[b] = {}
            for c in d[b]:
                if b == "triad":
                    ts = d[b][c]["all runs"]["total own"]
                else:
                    ts = d[b][c]["all runs"]["total app"]

                if c in ["nvcc", "hipcc"]:
                    c_name  = c
                elif c in ["clang-cuda","clang-hip"]:
                    c_name = c
                elif c in ["cudaomp", "cudaomp-amd"]:
                    c_name = "cuda-omp"
                    #time_cudaomp[b] = d[b][c]["medians"]["total app"]
                elif c in ["omp-offload"]:
                    c_name = "omp-offload"
                    #time_omp[b] = d[b][c]["medians"]["total app"]    

                for t in ts:
                    vendor_col = vendor_col + [v_name]
                    benchmark_col = benchmark_col + [b]
                    compiler_col = compiler_col + [c_name]
                    exec_time_col = exec_time_col + [t]
                    #print(v,b,c,t)
            #print("SPEEDUP: ", b, round(time_omp[b] / time_cudaomp[b],3))
    data = {"vendor" : vendor_col, "benchmark" : benchmark_col, "compiler" : compiler_col, "exec. time (seconds)" : exec_time_col}
    df = pds.DataFrame.from_dict(data)                
    return df     


def main():
    gpu_vendors = sys.argv[1]
    if gpu_vendors == 'amd':
        AMD=True
        NVIDIA=False
    elif gpu_vendors == 'nvidia':
        AMD=False
        NVIDIA=True   
    elif gpu_vendors == 'both':
        AMD=True
        NVIDIA=True 
    else:
        print('unknown compiler vendor (command line argument)')
        exit(1)    
    if AMD:
        with open('eval_results_amd.json', 'r') as f:
            amd_results_dict = json.load(f)
    else:
        amd_results_dict = None     
    if NVIDIA:
        with open('eval_results_nvidia.json', 'r') as f:
            nvidia_results_dict = json.load(f)
    else:
        nvidia_results_dict = None        
    with open('eval_results_nvidia.json', 'r') as f:
        nvidia_results_dict = json.load(f)    
    df = to_dataframe(nvidia_results_dict, amd_results_dict, AMD=AMD, NVIDIA=NVIDIA)
    # print(df.columns)
    di = { 'NVIDIA' : NVIDIA_MACHINE , 'AMD' : AMD_MACHINE} 
    df['vendor'] = df['vendor'].map(di)
    if AMD and NVIDIA:
        systems=[di['NVIDIA'], di['AMD']]
    elif AMD: 
        systems=[di['AMD']]
    else: 
        systems=[di['NVIDIA']]
    sns.set(font_scale=1.25)
    sns.set_style("whitegrid")
    colors = sns.color_palette("tab10", 5)
    # print(colors)
    dc = {'clang-cuda' : 'clang-cc',  
          'cuda-omp' : 'cuda-omp-cc',
          'clang-hip' : 'clang-cc',
          'nvcc' : 'vendor-cc',
          'hipcc' : 'vendor-cc'}#,
          #'omp-offload' : 'omp-cc'}
    # print(df['compiler'].unique())
    df['compiler'] = df['compiler'].map(dc)
    # print(df['compiler'].unique())
    colors = sns.color_palette("tab10", len(df['compiler'].unique()))
    compilers = df['compiler'].unique()
    ci = {}
    for o, c in zip(colors, compilers):
        ci[c] = o

    # print(compilers)
    # print(ci)
    with sns.plotting_context(rc={"legend.fontsize":20, 'text.usetex' : True}):
        for bench in df['benchmark'].unique():
            d = df[df['benchmark'] == bench]
            g = sns.catplot(x = "compiler", 
                    col="vendor",
                    palette=ci,
                    col_order = systems,
                    order = ['vendor-cc', 'clang-cc', 'cuda-omp-cc'],#, 'omp-cc'],
                    linewidth=0.5,
                    sharey = True,
                    sharex = False,
                    facet_kws={'sharey': True, 'sharex': False},
                    y="exec. time (seconds)", 
                    data=d, 
                    kind="strip", 
                    alpha=0.7,
                    height=4, 
                    aspect=1);
            for c,s in zip(g.axes.flat,systems):
                c.set_title(s)
                c.set_xticklabels(c.get_xticklabels(),rotation = 30)

            g.set_axis_labels('', 'Execution Time (s)')
            plt.tight_layout()
            plt.savefig(f'{bench}.pdf')
            print('Generated ' + bench + '.pdf')
        plt.close()

if __name__ == "__main__":
    main()
