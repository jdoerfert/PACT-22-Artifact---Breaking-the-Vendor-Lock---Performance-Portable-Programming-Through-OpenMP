/*
   ECL-GC code: ECL-GC is a graph-coloring algorithm with shortcutting. The CUDA
   implementation thereof is quite fast. It operates on graphs stored in binary
   CSR format.

   Copyright (c) 2020, Texas State University. All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright
 notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
 notice, this list of conditions and the following disclaimer in the
 documentation and/or other materials provided with the distribution.
 * Neither the name of Texas State University nor the names of its
 contributors may be used to endorse or promote products derived from
 this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL TEXAS STATE UNIVERSITY BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Authors: Ghadeer Alabandi, Evan Powers, and Martin Burtscher

URL: The latest version of this code is available at
https://userweb.cs.txstate.edu/~burtscher/research/ECL-GC/.

Publication: This work is described in detail in the following paper.
Ghadeer Alabandi, Evan Powers, and Martin Burtscher. Increasing the Parallelism
of Graph Coloring via Shortcutting. Proceedings of the 2020 ACM SIGPLAN
Symposium on Principles and Practice of Parallel Programming, pp. 262-275.
February 2020.
 */


#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <chrono>
#include "common.h"
#include "graph.h"

static const int ThreadsPerBlock = 256;
static const int WS = 32;  // warp size and bits per int
static const int MSB = 1 << (WS - 1);
static const int Mask = (1 << (WS / 2)) - 1;

// https://github.com/intel/llvm/discussions/4441
inline unsigned int ballot(sycl::sub_group sg, int predicate) {
  size_t id = sg.get_local_linear_id();
  unsigned int local_val = (predicate ? 1u : 0u) << id;
  return sycl::reduce_over_group(sg, local_val, sycl::plus<>());
}

inline int ffs(int x) {
  return (x == 0) ? 0 : sycl::intel::ctz(x) + 1;
}

// https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
static unsigned int hash(unsigned int val)
{
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  return (val >> 16) ^ val;
}

void kernel_runLarge(
    nd_item<1> &item,
    const int nodes, 
    const int* const __restrict__ nidx,
    const int* const __restrict__ nlist,
    volatile int* const __restrict__ posscol,
    int* const __restrict__ posscol2,
    volatile int* const __restrict__ color,
    const int* const __restrict__ wl,
    const int* __restrict__ wlsize)
{
  const int stop = wlsize[0];
  if (stop != 0) {
    const int lane = item.get_local_id(0) % WS;
    const int thread = item.get_global_id(0);
    const int threads = item.get_group_range(0) * ThreadsPerBlock;
    bool again;
    auto sg = item.get_sub_group();
    do {
      again = false;
      for (int w = thread; ext::oneapi::any_of(item.get_group(), w < stop); w += threads) {
        bool shortcut, done, cond = false;
        int v, data, range, beg, pcol;
        if (w < stop) {
          v = wl[w];
          data = color[v];
          range = data >> (WS / 2);
          if (range > 0) {
            beg = nidx[v];
            pcol = posscol[v];
            cond = true;
          }
        }

        int bal = ballot(sg, cond);
        while (bal != 0) {
          const int who = ffs(bal) - 1;
          bal &= bal - 1;
          const int wdata = sg.shuffle(data, who);
          const int wrange = wdata >> (WS / 2);
          const int wbeg = sg.shuffle(beg, who);
          const int wmincol = wdata & Mask;
          const int wmaxcol = wmincol + wrange;
          const int wend = wbeg + wmaxcol;
          const int woffs = wbeg / WS;
          int wpcol = sg.shuffle(pcol, who);

          bool wshortcut = true;
          bool wdone = true;
          for (int i = wbeg + lane; ext::oneapi::any_of(item.get_group(), i < wend); i += WS) {
            int nei, neidata, neirange;
            if (i < wend) {
              nei = nlist[i];
              neidata = color[nei];
              neirange = neidata >> (WS / 2);
              const bool neidone = (neirange == 0);
              wdone &= neidone; //consolidated below
              if (neidone) {
                const int neicol = neidata;
                if (neicol < WS) {
                  wpcol &= ~((unsigned int)MSB >> neicol); //consolidated below
                } else {
                  if ((wmincol <= neicol) && (neicol < wmaxcol) && ((posscol2[woffs + neicol / WS] << (neicol % WS)) < 0)) {

                    //atomic_fetch_and(posscol2[woffs + neicol / WS], (int)(~((unsigned int)MSB >> (neicol % WS))));
                    auto ao = ext::oneapi::atomic_ref<int, 
                         ext::oneapi::memory_order::relaxed,
                         ext::oneapi::memory_scope::device,
                         access::address_space::global_space> (posscol2[woffs + neicol / WS]);
                    ao &= (int)(~((unsigned int)MSB >> (neicol % WS)));
                  }
                }
              } else {
                const int neimincol = neidata & Mask;
                const int neimaxcol = neimincol + neirange;
                if ((neimincol <= wmincol) && (neimaxcol >= wmincol)) wshortcut = false; //consolidated below
              }
            }
          }
          wshortcut = ext::oneapi::all_of(item.get_group(), wshortcut);
          wdone = ext::oneapi::all_of(item.get_group(), wdone);
          wpcol &= sg.shuffle_xor(wpcol, 1);
          wpcol &= sg.shuffle_xor(wpcol, 2);
          wpcol &= sg.shuffle_xor(wpcol, 4);
          wpcol &= sg.shuffle_xor(wpcol, 8);
          wpcol &= sg.shuffle_xor(wpcol, 16);
          if (who == lane) pcol = wpcol;
          if (who == lane) done = wdone;
          if (who == lane) shortcut = wshortcut;
        }

        if (w < stop) {
          if (range > 0) {
            const int mincol = data & Mask;
            int val = pcol, mc = 0;
            if (pcol == 0) {
              const int offs = beg / WS;
              mc = sycl::max(1, mincol / WS);
              while ((val = posscol2[offs + mc]) == 0) mc++;
            }
            int newmincol = mc * WS + sycl::clz(val);
            if (mincol != newmincol) shortcut = false;
            if (shortcut || done) {
              pcol = (newmincol < WS) ? ((unsigned int)MSB >> newmincol) : 0;
            } else {
              const int maxcol = mincol + range;
              const int range = maxcol - newmincol;
              newmincol = (range << (WS / 2)) | newmincol;
              again = true;
            }
            posscol[v] = pcol;
            color[v] = newmincol;
          }
        }
      }
    } while (ext::oneapi::any_of(item.get_group(), again));
  }
}

int main(int argc, char* argv[])
{
  printf("ECL-GC v1.2 (%s)\n", __FILE__);
  printf("Copyright 2020 Texas State University\n\n");

  if (argc != 2) {printf("USAGE: %s input_file_name\n\n", argv[0]);  exit(-1);}
  if (WS != 32) {printf("ERROR: warp size must be 32\n\n");  exit(-1);}
  if (WS != sizeof(int) * 8) {printf("ERROR: bits per word must match warp size\n\n");  exit(-1);}
  if ((ThreadsPerBlock < WS) || ((ThreadsPerBlock % WS) != 0)) {
    printf("ERROR: threads per block must be a multiple of the warp size\n\n");
    exit(-1);
  }
  if ((ThreadsPerBlock & (ThreadsPerBlock - 1)) != 0) {
    printf("ERROR: threads per block must be a power of two\n\n");
    exit(-1);
  }

  ECLgraph g = readECLgraph(argv[1]);
  printf("input: %s\n", argv[1]);

  const int nodes = g.nodes;
  const int edges = g.edges;

  printf("nodes: %d\n", nodes);
  printf("edges: %d\n", edges);
  printf("avg degree: %.2f\n", 1.0 * edges / nodes);

  int* const color = new int [nodes];

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<int, 1> nidx_d (g.nindex, nodes + 1);
  buffer<int, 1> nlist_d (g.nlist, edges);
  buffer<int, 1> nlist2_d (edges);
  buffer<int, 1> posscol_d (nodes);
  buffer<int, 1> posscol2_d (g.edges / WS + 1);
  buffer<int, 1> color_d (nodes);
  buffer<int, 1> wl_d (nodes);
  buffer<int, 1> wlsize_d (1);

  const int blocks = 24;

  q.wait();

  auto start = std::chrono::high_resolution_clock::now();


  range<1> gws (blocks * ThreadsPerBlock);
  range<1> lws (ThreadsPerBlock);

  for (int n = 0; n < 1; n++) {
    q.submit([&] (handler &cgh) {
      auto wlsize = wlsize_d.get_access<sycl_write>(cgh);
      cgh.fill(wlsize, 0);
    });
   
    q.submit([&] (handler &cgh) {
      stream out(1024, 256, cgh);
      auto nidx = nidx_d.get_access<sycl_read>(cgh);
      auto nlist = nlist_d.get_access<sycl_read>(cgh);
      auto nlist2 = nlist2_d.get_access<sycl_write>(cgh);
      auto posscol = posscol_d.get_access<sycl_write>(cgh);
      auto posscol2 = posscol2_d.get_access<sycl_write>(cgh);
      auto color = color_d.get_access<sycl_write>(cgh);
      auto wl = wl_d.get_access<sycl_write>(cgh);
      auto wlsize = wlsize_d.get_access<sycl_atomic>(cgh);
      cgh.parallel_for<class init>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {

        const int lane = item.get_local_id(0) % WS;
        const int thread = item.get_global_id(0);
        const int threads = item.get_group_range(0) * ThreadsPerBlock;
        auto sg = item.get_sub_group();

        int maxrange = -1;
        for (int v = thread; ext::oneapi::any_of(item.get_group(), v < nodes); v += threads) {
          bool cond = false;
          int beg, end, pos, degv, active;
          if (v < nodes) {
            beg = nidx[v];
            end = nidx[v + 1];
            degv = end - beg;
            cond = (degv >= WS);
            if (cond) {
              wl[atomic_fetch_add(wlsize[0], 1)] = v;
            } else {
              active = 0;
              pos = beg;
              for (int i = beg; i < end; i++) {
                const int nei = nlist[i];
                const int degn = nidx[nei + 1] - nidx[nei];
                if ((degv < degn) || ((degv == degn) && (hash(v) < hash(nei))) || ((degv == degn) && (hash(v) == hash(nei)) && (v < nei))) {
                  active |= (unsigned int)MSB >> (i - beg);
                  pos++;
                }
              }
            }
          }

          int bal = ballot(sg, cond);
          while (bal != 0) {
            const int who = ffs(bal) - 1;
            bal &= bal - 1;
            const int wv = sg.shuffle(v, who);
            const int wbeg = sg.shuffle(beg, who);
            const int wend = sg.shuffle(end, who);
            const int wdegv = wend - wbeg;
            int wpos = wbeg;
            for (int i = wbeg + lane; ext::oneapi::any_of(item.get_group(), i < wend); i += WS) {
              int wnei;
              bool prio = false;
              if (i < wend) {
                wnei = nlist[i];
                const int wdegn = nidx[wnei + 1] - nidx[wnei];
                prio = ((wdegv < wdegn) || ((wdegv == wdegn) && (hash(wv) < hash(wnei))) || ((wdegv == wdegn) && (hash(wv) == hash(wnei)) && (wv < wnei)));
              }
              const int b = ballot(sg, prio);
              const int offs = sycl::popcount(b & ((1 << lane) - 1));
              if (prio) nlist2[wpos + offs] = wnei;
              wpos += sycl::popcount(b);
            }
            if (who == lane) pos = wpos;
          }

          if (v < nodes) {
            const int range = pos - beg;
            maxrange = sycl::max(maxrange, range);
            color[v] = (cond || (range == 0)) ? (range << (WS / 2)) : active;
            posscol[v] = (range >= WS) ? -1 : (MSB >> range);
          }
        }
        if (maxrange >= Mask) out << "too many active neighbors\n";

        for (int i = thread; i < edges / WS + 1; i += threads) posscol2[i] = -1;
      });
    });

    q.submit([&] (handler &cgh) {
      auto nidx = nidx_d.get_access<sycl_read>(cgh);
      auto nlist = nlist2_d.get_access<sycl_read>(cgh);
      auto posscol = posscol_d.get_access<sycl_read_write>(cgh);
      auto posscol2 = posscol2_d.get_access<sycl_read_write>(cgh);
      auto color = color_d.get_access<sycl_read_write>(cgh);
      auto wl = wl_d.get_access<sycl_read>(cgh);
      auto wlsize = wlsize_d.get_access<sycl_read>(cgh);
      cgh.parallel_for<class runLarge>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        kernel_runLarge(item, nodes, 
                        nidx.get_pointer(),
                        nlist.get_pointer(),
                        posscol.get_pointer(), 
                        posscol2.get_pointer(), 
                        color.get_pointer(), 
                        wl.get_pointer(), 
                        wlsize.get_pointer());

      });
    });

    q.submit([&] (handler &cgh) {
      auto nidx = nidx_d.get_access<sycl_read>(cgh);
      auto nlist = nlist_d.get_access<sycl_read>(cgh);
      auto posscol = posscol_d.get_access<sycl_read_write>(cgh);
      auto color = color_d.get_access<sycl_read_write>(cgh);
      cgh.parallel_for<class runSmall>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        const int thread = item.get_global_id(0);
        const int threads = item.get_group_range(0) * ThreadsPerBlock;

        bool again;
        do {
          again = false;
          for (int v = thread; v < nodes; v += threads) {
            int pcol = posscol[v];
            if (sycl::popcount(pcol) > 1) {
              const int beg = nidx[v];
              int active = color[v];
              int allnei = 0;
              int keep = active;
              do {
                const int old = active;
                active &= active - 1;
                const int curr = old ^ active;
                const int i = beg + sycl::clz(curr);
                const int nei = nlist[i];
                const int neipcol = posscol[nei];
                allnei |= neipcol;
                if ((pcol & neipcol) == 0) {
                  pcol &= pcol - 1;
                  keep ^= curr;
                } else if (sycl::popcount(neipcol) == 1) {
                  pcol ^= neipcol;
                  keep ^= curr;
                }
              } while (active != 0);
              if (keep != 0) {
                const int best = (unsigned int)MSB >> sycl::clz(pcol);
                if ((best & ~allnei) != 0) {
                  pcol = best;
                  keep = 0;
                }
              }
              again |= keep;
              if (keep == 0) keep = sycl::clz(pcol);
              color[v] = keep;
              posscol[v] = pcol;
            }
          }
        } while (again);
      });
    });
  }
  q.wait();

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  double runtime = elapsed_seconds.count() / 100;

  printf("average runtime:    %.6f s\n", runtime);
  printf("throughput: %.6f Mnodes/s\n", nodes * 0.000001 / runtime);
  printf("throughput: %.6f Medges/s\n", edges * 0.000001 / runtime);

  q.submit([&] (handler &cgh) {
    auto acc = color_d.get_access<sycl_read>(cgh);
    cgh.copy(acc, color);
  }).wait();

  bool ok = true;
  for (int v = 0; v < g.nodes; v++) {
    if (color[v] < 0) {
      printf("ERROR: found unprocessed node in graph (node %d with deg %d)\n\n",
          v, g.nindex[v + 1] - g.nindex[v]);
      ok = false;
      break;
    }
    for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
      if (color[g.nlist[i]] == color[v]) {
        printf("ERROR: found adjacent nodes with same color %d (%d %d)\n\n",
            color[v], v, g.nlist[i]);
        ok = false;
        break;
      }
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

#ifdef DEBUG
  const int vals = 16;
  int c[vals];
  for (int i = 0; i < vals; i++) c[i] = 0;
  int cols = -1;
  for (int v = 0; v < nodes; v++) {
    cols = std::max(cols, color[v]);
    if (color[v] < vals) c[color[v]]++;
  }
  cols++;
  printf("colors used: %d\n", cols);

  int sum = 0;
  for (int i = 0; i < std::min(vals, cols); i++) {
    sum += c[i];
    printf("col %2d: %10d (%5.1f%%)\n", i, c[i], 100.0 * sum / nodes);
  }
#endif

  delete [] color;
  freeECLgraph(g);
  return 0;
}
