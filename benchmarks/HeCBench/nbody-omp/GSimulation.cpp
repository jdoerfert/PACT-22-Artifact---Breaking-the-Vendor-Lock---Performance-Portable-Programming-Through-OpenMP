//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include "GSimulation.hpp"

/* Default Constructor for the GSimulation class which sets up the default
 * values for number of particles, number of integration steps, time steo and
 * sample frequency */
GSimulation::GSimulation() {
  std::cout << "==============================="
    << "\n";
  std::cout << " Initialize Gravity Simulation"
    << "\n";
  set_npart(16000);
  set_nsteps(10);
  set_tstep(0.1);
  set_sfreq(1);
}

/* Set the number of particles */
void GSimulation::SetNumberOfParticles(int N) { set_npart(N); }

/* Set the number of integration steps */
void GSimulation::SetNumberOfSteps(int N) { set_nsteps(N); }

/* Initialize the position of all the particles using random number generator
 * between 0 and 1.0 */
void GSimulation::InitPos() {
  std::random_device rd;  // random number generator
  std::mt19937 gen(42);
  std::uniform_real_distribution<RealType> unif_d(0, 1.0);

  for (int i = 0; i < get_npart(); ++i) {
    particles_[i].pos[0] = unif_d(gen);
    particles_[i].pos[1] = unif_d(gen);
    particles_[i].pos[2] = unif_d(gen);
  }
}

/* Initialize the velocity of all the particles using random number generator
 * between -1.0 and 1.0 */
void GSimulation::InitVel() {
  std::random_device rd;  // random number generator
  std::mt19937 gen(42);
  std::uniform_real_distribution<RealType> unif_d(-1.0, 1.0);

  for (int i = 0; i < get_npart(); ++i) {
    particles_[i].vel[0] = unif_d(gen) * 1.0e-3f;
    particles_[i].vel[1] = unif_d(gen) * 1.0e-3f;
    particles_[i].vel[2] = unif_d(gen) * 1.0e-3f;
  }
}

/* Initialize the acceleration of all the particles to 0 */
void GSimulation::InitAcc() {
  for (int i = 0; i < get_npart(); ++i) {
    particles_[i].acc[0] = 0.f;
    particles_[i].acc[1] = 0.f;
    particles_[i].acc[2] = 0.f;
  }
}

/* Initialize the mass of all the particles using a random number generator
 * between 0 and 1 */
void GSimulation::InitMass() {
  RealType n = static_cast<RealType>(get_npart());
  std::random_device rd;  // random number generator
  std::mt19937 gen(42);
  std::uniform_real_distribution<RealType> unif_d(0.0, 1.0);

  for (int i = 0; i < get_npart(); ++i) {
    particles_[i].mass = n * unif_d(gen);
  }
}

/* This function does the simulation logic for Nbody */
void GSimulation::Start() {
  RealType dt = get_tstep();
  int n = get_npart();
  // RealType* energy = new RealType[n];
  std::vector<RealType> energy(n, 0.f);
  // allocate particles
  // particles_ = new Particle[n];
  particles_.resize(n);

  InitPos();
  InitVel();
  InitAcc();
  InitMass();

  PrintHeader();

  total_time_ = 0.;

  constexpr float kSofteningSquared = 1e-3f;
  // prevents explosion in the case the particles are really close to each other
  constexpr float kG = 6.67259e-11f;
  double gflops = 1e-9 * ((11. + 18.) * n * n + n * 19.);
  int nf = 0;
  double av = 0.0, dev = 0.0;

  Particle *p = particles_.data();
  RealType *e = energy.data();

  TimeInterval t0;
  int nsteps = get_nsteps();

#pragma omp target data map (to: p[0:n]) map(alloc: e[0:n])
  {
    // Looping across integration steps
    for (int s = 1; s <= nsteps; ++s) {
      TimeInterval ts0;
      // computes acceleration of all particles
#pragma omp target teams distribute parallel for thread_limit(256)
      for (int i = 0; i < n; i++) {
        RealType acc0 = p[i].acc[0];
        RealType acc1 = p[i].acc[1];
        RealType acc2 = p[i].acc[2];
        for (int j = 0; j < n; j++) {
          RealType dx, dy, dz;
          RealType distance_sqr = 0.0f;
          RealType distance_inv = 0.0f;

          dx = p[j].pos[0] - p[i].pos[0];  // 1flop
          dy = p[j].pos[1] - p[i].pos[1];  // 1flop
          dz = p[j].pos[2] - p[i].pos[2];  // 1flop

          distance_sqr =
            dx * dx + dy * dy + dz * dz + kSofteningSquared;  // 6flops
          distance_inv = 1.0f / sqrtf(distance_sqr);       // 1div+1sqrt

          acc0 += dx * kG * p[j].mass * distance_inv * distance_inv *
            distance_inv;  // 6flops
          acc1 += dy * kG * p[j].mass * distance_inv * distance_inv *
            distance_inv;  // 6flops
          acc2 += dz * kG * p[j].mass * distance_inv * distance_inv *
            distance_inv;  // 6flops
        }
        p[i].acc[0] = acc0;
        p[i].acc[1] = acc1;
        p[i].acc[2] = acc2;
      }

      // Second kernel updates the velocity and position for all particles
#pragma omp target teams distribute parallel for thread_limit(256)
      for (int i = 0; i < n; i++) {
        p[i].vel[0] += p[i].acc[0] * dt;  // 2flops
        p[i].vel[1] += p[i].acc[1] * dt;  // 2flops
        p[i].vel[2] += p[i].acc[2] * dt;  // 2flops

        p[i].pos[0] += p[i].vel[0] * dt;  // 2flops
        p[i].pos[1] += p[i].vel[1] * dt;  // 2flops
        p[i].pos[2] += p[i].vel[2] * dt;  // 2flops

        p[i].acc[0] = 0.f;
        p[i].acc[1] = 0.f;
        p[i].acc[2] = 0.f;

        e[i] = p[i].mass *
          (p[i].vel[0] * p[i].vel[0] + p[i].vel[1] * p[i].vel[1] +
           p[i].vel[2] * p[i].vel[2]);  // 7flops
      }
      // Third kernel accumulates the energy of this Nbody system
#pragma omp target 
      for (int i = 1; i < n; i++) e[0] += e[i];

#pragma omp target update from (e[0:1])
      kenergy_ = 0.5 * e[0];
      e[0] = 0;
      double elapsed_seconds = ts0.Elapsed();
      if ((s % get_sfreq()) == 0) {
        nf += 1;
        std::cout << " " << std::left << std::setw(8) << s << std::left
          << std::setprecision(5) << std::setw(8) << s * get_tstep()
          << std::left << std::setprecision(5) << std::setw(12)
          << kenergy_ << std::left << std::setprecision(5)
          << std::setw(12) << elapsed_seconds << std::left
          << std::setprecision(5) << std::setw(12)
          << gflops * get_sfreq() / elapsed_seconds << "\n";
        if (nf > 2) {
          av += gflops * get_sfreq() / elapsed_seconds;
          dev += gflops * get_sfreq() * gflops * get_sfreq() /
            (elapsed_seconds * elapsed_seconds);
        }
      }
    }  // end of the time step loop
  }
  total_time_ = t0.Elapsed();
  total_flops_ = gflops * get_nsteps();
  av /= (double)(nf - 2);
  dev = sqrt(dev / (double)(nf - 2) - av * av);

  std::cout << "\n";
  std::cout << "# Total Time (s)     : " << total_time_ << "\n";
  std::cout << "# Average Performance : " << av << " +- " << dev << "\n";
  std::cout << "==============================="
    << "\n";
}

/* Print the headers for the output */
void GSimulation::PrintHeader() {
  std::cout << " nPart = " << get_npart() << "; "
    << "nSteps = " << get_nsteps() << "; "
    << "dt = " << get_tstep() << "\n";

  std::cout << "------------------------------------------------"
    << "\n";
  std::cout << " " << std::left << std::setw(8) << "s" << std::left
    << std::setw(8) << "dt" << std::left << std::setw(12) << "kenergy"
    << std::left << std::setw(12) << "time (s)" << std::left
    << std::setw(12) << "GFLOPS"
    << "\n";
  std::cout << "------------------------------------------------"
    << "\n";
}
