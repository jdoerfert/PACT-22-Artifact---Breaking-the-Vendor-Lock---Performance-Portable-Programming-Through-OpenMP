//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
//
// =============================================================

#include "mandel.hpp"
#include "util.hpp"


void Execute() {
  // Demonstrate the Mandelbrot calculation serial and parallel
  MandelParallel m_par(row_size, col_size, max_iterations);
  MandelSerial m_ser(row_size, col_size, max_iterations);

  // Run the code once to trigger JIT
  m_par.Evaluate();

  // Run the parallel version
  common::MyTimer t_par;
  // time the parallel computation
  for (int i = 0; i < repetitions; ++i) 
    m_par.Evaluate();
  common::Duration parallel_time = t_par.elapsed();

  // Print the results
  m_par.Print();

  // Run the serial version
  common::MyTimer t_ser;
  m_ser.Evaluate();
  common::Duration serial_time = t_ser.elapsed();

  // Report the results
  std::cout << std::setw(20) << "serial time: " << serial_time.count() << "s\n";
  std::cout << std::setw(20) << "parallel time: " << (parallel_time / repetitions).count() << "s\n";

  // Validating
  m_par.Verify(m_ser);
}

void Usage(std::string program_name) {
  // Utility function to display argument usage
  std::cout << " Incorrect parameters\n";
  std::cout << " Usage: ";
  std::cout << program_name << "\n\n";
  exit(-1);
}

int main(int argc, char *argv[]) {
  if (argc != 1) {
    Usage(argv[0]);
  }

  try {
    Execute();
  } catch (...) {
    std::cout << "Failure\n";
    std::terminate();
  }
  std::cout << "Success\n";
  return 0;
}
