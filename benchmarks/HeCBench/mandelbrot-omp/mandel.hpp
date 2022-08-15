//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#pragma once

#include <exception>
#include <iomanip>
#include <iostream>

constexpr int row_size = 100;
constexpr int col_size = 100;
constexpr int max_iterations = 100;
constexpr int repetitions = 100;
#define THREADS_PER_BLOCK_X 16
#define THREADS_PER_BLOCK_Y 16

// define a type of floating-point complex number
typedef struct {
	float real;
	float imag;
} ComplexF;



struct MandelParameters {
  int row_count_;
  int col_count_;
  int max_iterations_;

  MandelParameters(int row_count, int col_count, int max_iterations)
      : row_count_(row_count),
        col_count_(col_count),
        max_iterations_(max_iterations) { }

  int row_count() const { return row_count_; }
  int col_count() const { return col_count_; }
  int max_iterations() const { return max_iterations_; } 

  // scale from 0..row_count to -1.5..0.5
  float ScaleRow(int i) const { return -1.5f + (i * (2.0f / row_count_)); }

  // scale from 0..col_count to -1..1
  float ScaleCol(int i) const { return -1.0f + (i * (2.0f / col_count_)); }

  // mandelbrot set are points that do not diverge within max_iterations
  int Point(const ComplexF& c) const {
    int count = 0;
    ComplexF z = {0, 0};
    for (int i = 0; i < max_iterations_; ++i) {
      auto r = z.real;
      auto im = z.imag;
      // leave loop if diverging
      if (((r * r) + (im * im)) >= 4.0f) {
        break;
      }
      //z = z * z + c;
      z.real = r*r-im*im + c.real;
      z.imag = 2*r*im + c.imag;

      count++;
    }
    return count;
  }
};

class Mandel {
 private:
  MandelParameters p_;
  int *data_;  // [p_.row_count_][p_.col_count_];

 public:

  Mandel(int row_count, int col_count, int max_iterations)
      : p_(row_count, col_count, max_iterations) {
    data_ = new int[ p_.row_count() * p_.col_count() ];
  }

  ~Mandel() { delete[] data_; }

  MandelParameters GetParameters() const { return p_; }

  // use only for debugging with small dimensions
  void Print() {
    if (p_.row_count() > 128 || p_.col_count() > 128) {
      std::cout << "No Print() output due to size too large" << std::endl;
      return;
    }
    for (int i = 0; i < p_.row_count(); ++i) {
      for (int j = 0; j < p_.col_count_; ++j) {
        std::cout << std::setw(1) << ((GetValue(i,j) >= p_.max_iterations()) ? "x" : " ");
      }
      std::cout << std::endl;
    }
  }

  // accessors for data and count values
  int *data() const { return data_; }

  // accessors to read a value into the mandelbrot data matrix
  void SetValue(int i, int j, float v) { data_[i * p_.col_count_ + j] = v; }

  // accessors to store a value into the mandelbrot data matrix
  int GetValue(int i, int j) const { return data_[i * p_.col_count_ + j]; }

  // validate the results match
  void Verify(Mandel &m) {
    if ((m.p_.row_count() != p_.row_count_) || (m.p_.col_count() != p_.col_count_)) {
      std::cout << "Fail verification - matrix size is different" << std::endl;
      throw std::runtime_error("Verification failure");
    }

    int diff = 0;
    for (int i = 0; i < p_.row_count(); ++i) {
      for (int j = 0; j < p_.col_count(); ++j) {
        if (m.GetValue(i,j) != GetValue(i,j)) 
          diff++;
      }
    }

    double tolerance = 0.05;
    double ratio = (double)diff / (double)(p_.row_count() * p_.col_count());

#if _DEBUG
    std::cout << "diff: " << diff << std::endl;
    std::cout << "total count: " << p_.row_count() * p_.col_count() << std::endl;
#endif

    if (ratio > tolerance) {
      std::cout << "Fail verification - diff larger than tolerance"<< std::endl;
      throw std::runtime_error("Verification failure");
    }
#if _DEBUG
    std::cout << "Pass verification" << std::endl;
#endif
  }
};


class MandelSerial : public Mandel {
public:
  MandelSerial(int row_count, int col_count, int max_iterations)
    : Mandel(row_count, col_count, max_iterations) { }

  void Evaluate() {
    // iterate over image and compute mandel for each point
    MandelParameters p = GetParameters();

    for (int i = 0; i < p.row_count(); ++i) {
      for (int j = 0; j < p.col_count(); ++j) {
        //auto c = MandelParameters::ComplexF(p.ScaleRow(i), p.ScaleCol(j));
	ComplexF c= {p.ScaleRow(i), p.ScaleCol(j)};
        SetValue(i, j, p.Point(c));
      }
    }
  }
};


class MandelParallel : public Mandel {
public:
  MandelParallel(int row_count, int col_count, int max_iterations)
    : Mandel(row_count, col_count, max_iterations) { }

  void Evaluate() {
    // iterate over image and check if each point is in mandelbrot set
    MandelParameters p = GetParameters();

    const int rows = p.row_count();
    const int cols = p.col_count();

    //mandel <<< dim3(block_x, block_y), dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y) >>> (data_buf, p_buf, rows, cols);

    size_t image_size = rows * cols;
    auto b = data();
    #pragma omp target map(from: b[0:image_size])
    {
      #pragma omp teams distribute parallel for simd collapse(2) thread_limit(THREADS_PER_BLOCK_X*THREADS_PER_BLOCK_Y) 
      for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
          b[i * cols + j] = p.Point({p.ScaleRow(i), p.ScaleCol(j)});
    }
  }
};
