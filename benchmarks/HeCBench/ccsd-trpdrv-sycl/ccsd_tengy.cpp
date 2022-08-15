#include <stdio.h>
#include "common.h"

#define BLOCK_SIZE 16

inline double atomicAdd(double *addr, double operand)
{
  atomic<unsigned long long int, access::address_space::global_space> obj(
      (multi_ptr<unsigned long long int, access::address_space::global_space>(
                        reinterpret_cast<unsigned long long int *>(addr))));

  unsigned long long int old_value;
  double old_double_value;

  do {
    old_value = obj.load(memory_order::relaxed);
    old_double_value = *reinterpret_cast<const double *>(&old_value);
    const double new_double_value = old_double_value + operand;
    const unsigned long long int new_value =
      *reinterpret_cast<const unsigned long long int *>(&new_double_value);

    if (obj.compare_exchange_strong(old_value, new_value, memory_order::relaxed))
      break;
  } while (true);

  return old_double_value;
}

void ccsd_tengy_gpu(queue &q,
    const double * __restrict__ f1n,    const double * __restrict__ f1t,
    const double * __restrict__  f2n,    const double * __restrict__ f2t,
    const double * __restrict__  f3n,    const double * __restrict__ f3t,
    const double * __restrict__  f4n,    const double * __restrict__ f4t,
    const double * __restrict__  dintc1, const double * __restrict__ dintx1, const double * __restrict__ t1v1,
    const double * __restrict__  dintc2, const double * __restrict__ dintx2, const double * __restrict__ t1v2,
    const double * __restrict__  eorb,   const double eaijk,
    double * __restrict__ emp4i_, double * __restrict__ emp5i_,
    double * __restrict__ emp4k_, double * __restrict__ emp5k_,
    const int ncor, const int nocc, const int nvir)
{
  double emp5i = 0.0, emp4i = 0.0, emp5k = 0.0, emp4k = 0.0;

  {
    buffer<double, 1> d_f1n (f1n, nvir*nvir);
    buffer<double, 1> d_f2n (f2n, nvir*nvir);
    buffer<double, 1> d_f3n (f3n, nvir*nvir);
    buffer<double, 1> d_f4n (f4n, nvir*nvir);
    buffer<double, 1> d_f1t (f1t, nvir*nvir);
    buffer<double, 1> d_f2t (f2t, nvir*nvir);
    buffer<double, 1> d_f3t (f3t, nvir*nvir);
    buffer<double, 1> d_f4t (f4t, nvir*nvir);
    buffer<double, 1> d_dintc1 (dintc1, nvir);
    buffer<double, 1> d_dintc2 (dintc2, nvir);
    buffer<double, 1> d_dintx1 (dintx1, nvir);
    buffer<double, 1> d_dintx2 (dintx2, nvir);
    buffer<double, 1> d_t1v1 (t1v1, nvir);
    buffer<double, 1> d_t1v2 (t1v2, nvir);
    buffer<double, 1> d_eorb (eorb, (ncor+nocc+nvir));
    buffer<double, 1> d_emp5i (&emp5i, 1);
    buffer<double, 1> d_emp4i (&emp4i, 1);
    buffer<double, 1> d_emp5k (&emp5k, 1);
    buffer<double, 1> d_emp4k (&emp4k, 1);

    range<2> global_work_size((nvir+BLOCK_SIZE-1)/BLOCK_SIZE*BLOCK_SIZE, 
        (nvir+BLOCK_SIZE-1)/BLOCK_SIZE*BLOCK_SIZE);
    range<2> local_work_size(BLOCK_SIZE, BLOCK_SIZE);

    q.submit([&] (handler &cgh) {
      auto f1n = d_f1n.get_access<sycl_read>(cgh);
      auto f1t = d_f1t.get_access<sycl_read>(cgh);
      auto f2n = d_f2n.get_access<sycl_read>(cgh);
      auto f2t = d_f2t.get_access<sycl_read>(cgh);
      auto f3n = d_f3n.get_access<sycl_read>(cgh);
      auto f3t = d_f3t.get_access<sycl_read>(cgh);
      auto f4n = d_f4n.get_access<sycl_read>(cgh);
      auto f4t = d_f4t.get_access<sycl_read>(cgh);
      auto dintc1 = d_dintc1.get_access<sycl_read>(cgh);
      auto dintx1 = d_dintx1.get_access<sycl_read>(cgh);
      auto t1v1 = d_t1v1.get_access<sycl_read>(cgh);
      auto dintc2 = d_dintc2.get_access<sycl_read>(cgh);
      auto dintx2 = d_dintx2.get_access<sycl_read>(cgh);
      auto t1v2 = d_t1v2.get_access<sycl_read>(cgh);
      auto eorb = d_eorb.get_access<sycl_read>(cgh);
      auto emp4i = d_emp4i.get_access<sycl_read_write>(cgh);
      auto emp5i = d_emp5i.get_access<sycl_read_write>(cgh);
      auto emp4k = d_emp4k.get_access<sycl_read_write>(cgh);
      auto emp5k = d_emp5k.get_access<sycl_read_write>(cgh);

      cgh.parallel_for<class tengy>(nd_range<2>(global_work_size, local_work_size), [=] (nd_item<2> item) {
        const int b = item.get_global_id(1);
        const int c = item.get_global_id(0); 

        if (b < nvir && c < nvir) {
          const double denom = -1.0 / (eorb[ncor+nocc+b] + eorb[ncor+nocc+c] + eaijk);

          // nvir < 10000 so this should never overflow
          const int bc = b+c*nvir;
          const int cb = c+b*nvir;

          const double f1nbc = f1n[bc];
          const double f1tbc = f1t[bc];
          const double f1ncb = f1n[cb];
          const double f1tcb = f1t[cb];

          const double f2nbc = f2n[bc];
          const double f2tbc = f2t[bc];
          const double f2ncb = f2n[cb];
          const double f2tcb = f2t[cb];

          const double f3nbc = f3n[bc];
          const double f3tbc = f3t[bc];
          const double f3ncb = f3n[cb];
          const double f3tcb = f3t[cb];

          const double f4nbc = f4n[bc];
          const double f4tbc = f4t[bc];
          const double f4ncb = f4n[cb];
          const double f4tcb = f4t[cb];

          atomicAdd(emp4i.get_pointer() , denom * (f1tbc+f1ncb+f2tcb+f3nbc+f4ncb) * (f1tbc-f2tbc*2-f3tbc*2+f4tbc)
                  - denom * (f1nbc+f1tcb+f2ncb+f3ncb) * (f1tbc*2-f2tbc-f3tbc+f4tbc*2)
                  + denom * 3 * (f1nbc*(f1nbc+f3ncb+f4tcb*2) +f2nbc*f2tcb+f3nbc*f4tbc));

          atomicAdd(emp4k.get_pointer() , denom * (f1nbc+f1tcb+f2ncb+f3tbc+f4tcb) * (f1nbc-f2nbc*2-f3nbc*2+f4nbc)
                  - denom * (f1tbc+f1ncb+f2tcb+f3tcb) * (f1nbc*2-f2nbc-f3nbc+f4nbc*2)
                  + denom * 3 * (f1tbc*(f1tbc+f3tcb+f4ncb*2) +f2tbc*f2ncb+f3tbc*f4nbc));

          const double t1v1b = t1v1[b];
          const double t1v2b = t1v2[b];

          const double dintx1c = dintx1[c];
          const double dintx2c = dintx2[c];
          const double dintc1c = dintc1[c];
          const double dintc2c = dintc2[c];

          atomicAdd(emp5i.get_pointer(), denom * t1v1b * dintx1c * (f1tbc+f2nbc+f4ncb-(f3tbc+f4nbc+f2ncb+f1nbc+f2tbc+f3ncb)*2
                    +(f3nbc+f4tbc+f1ncb)*4) + denom * t1v1b * dintc1c * (f1nbc+f4nbc+f1tcb -(f2nbc+f3nbc+f2tcb)*2));
          atomicAdd(emp5k.get_pointer(), denom * t1v2b * dintx2c * (f1nbc+f2tbc+f4tcb -(f3nbc+f4tbc+f2tcb +f1tbc+f2nbc+f3tcb)*2
                    +(f3tbc+f4nbc+f1tcb)*4) + denom * t1v2b * dintc2c * (f1tbc+f4tbc+f1ncb -(f2tbc+f3tbc+f2ncb)*2));
        }
      });
    });
  }

  *emp4i_ = emp4i;
  *emp4k_ = emp4k;
  *emp5i_ = emp5i;
  *emp5k_ = emp5k;
}

