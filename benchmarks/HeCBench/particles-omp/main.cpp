/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#define MAX_EPSILON_ERROR 5.00f
#define THRESHOLD         0.30f
#define GRID_SIZE         64
#define NUM_PARTICLES     16384

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "particles.h"

// Simulation parameters
const float timestep = 0.5f;                // Time slice for re-computation iteration
//const float gravity = 0.0005f;            // Strength of gravity
//const float damping = 1.0f;
const float fParticleRadius = 0.023f;       // Radius of individual particles
const float fColliderRadius = 0.17f;        // Radius of collider for interacting with particles in 'm' mode
//const float collideSpring = 0.4f;         // Elastic spring constant for impact between particles
//const float collideDamping = 0.025f;      // Inelastic loss component for impact between particles
//const float collideShear = 0.12f;         // Friction constant for particles in contact
//const float collideAttraction = 0.0012f;  // Attraction between particles (~static or Van der Waals) 


inline float frand(void)
{
  return (float)rand() / (float)RAND_MAX;
}

void initGrid(float *hPos, float *hVel, float particleRadius, float spacing, 
              unsigned int numParticles)
{
  float jitter = particleRadius * 0.01f;
  unsigned int s = (int) ceilf(powf((float) numParticles, 1.0f / 3.0f));
  unsigned int gridSize[3];
  gridSize[0] = gridSize[1] = gridSize[2] = s;

  srand(1973);
  for(unsigned int z=0; z<gridSize[2]; z++) 
  {
    for(unsigned int y=0; y<gridSize[1]; y++) 
    {
      for(unsigned int x=0; x<gridSize[0]; x++) 
      {
        unsigned int i = (z * gridSize[0] * gridSize[1]) + (y * gridSize[1]) + x;
        if (i < numParticles) 
        {
          hPos[i * 4] = (spacing * x) + particleRadius - 1.0f + (frand() * 2.0f - 1.0f) * jitter;
          hPos[i * 4 + 1] = (spacing * y) + particleRadius - 1.0f + (frand() * 2.0f - 1.0f) * jitter;
          hPos[i * 4 + 2] = (spacing * z) + particleRadius - 1.0f + (frand() * 2.0f - 1.0f) * jitter;
          hPos[i * 4 + 3] = 1.0f;
          hVel[i * 4] = 0.0f;
          hVel[i * 4 + 1] = 0.0f;
          hVel[i * 4 + 2] = 0.0f;
          hVel[i * 4 + 3] = 0.0f;
        }
      }
    }
  }
}


int main(int argc, char** argv) 
{
  const int iterations = atoi(argv[1]);               // Number of iterations
  unsigned int numParticles = NUM_PARTICLES;
  unsigned int gridDim = GRID_SIZE;

  // Set and log grid size and particle count, after checking optional command-line inputs
  uint3 gridSize;
  gridSize.x = gridSize.y = gridSize.z = gridDim;

  unsigned int numGridCells = gridSize.x * gridSize.y * gridSize.z;
  //float3 worldSize = {2.0f, 2.0f, 2.0f};

  // set simulation parameters
  simParams_t params;
  params.gridSize = gridSize;
  params.numCells = numGridCells;
  params.numBodies = numParticles;
  params.particleRadius = fParticleRadius; 
  params.colliderPos = {1.2f, -0.8f, 0.8f};
  params.colliderRadius = fColliderRadius;

  params.worldOrigin = {1.0f, -1.0f, -1.0f};
  float cellSize = params.particleRadius * 2.0f;  // cell size equal to particle diameter
  params.cellSize = {cellSize, cellSize, cellSize};

  params.spring = 0.5f;
  params.damping = 0.02f;
  params.shear = 0.1f;
  params.attraction = 0.0f;
  params.boundaryDamping = -0.5f;

  params.gravity = {0.0f, -0.0003f, 0.0f};
  params.globalDamping = 1.0f;

  printf(" grid: %d x %d x %d = %d cells\n", gridSize.x, gridSize.y, gridSize.z, numGridCells);
  printf(" particles: %d\n\n", numParticles);

  float* hPos          = (float*)malloc(numParticles * 4 * sizeof(float));
  float* hVel          = (float*)malloc(numParticles * 4 * sizeof(float));
  float* hReorderedPos = (float*)malloc(numParticles * 4 * sizeof(float));
  float* hReorderedVel = (float*)malloc(numParticles * 4 * sizeof(float));
  unsigned int* hHash      = (unsigned int*)malloc(numParticles * sizeof(unsigned int));
  unsigned int* hIndex     = (unsigned int*)malloc(numParticles * sizeof(unsigned int));
  unsigned int* hCellStart = (unsigned int*)malloc(numGridCells * sizeof(unsigned int));
  unsigned int* hCellEnd   = (unsigned int*)malloc(numGridCells * sizeof(unsigned int));

  // configure grid 
  initGrid(hPos, hVel, params.particleRadius, params.particleRadius * 2.0f, numParticles);

  float4* dPos = (float4*)hPos;
  float4* dVel = (float4*)hVel;
  float4* dReorderedPos = (float4*)hReorderedPos; 
  float4* dReorderedVel = (float4*)hReorderedVel;
  unsigned int* dHash = hHash;
  unsigned int* dIndex = hIndex;
  unsigned int* dCellStart = hCellStart;
  unsigned int* dCellEnd = hCellEnd;

#pragma omp target data map(to: dPos[0:numParticles], \
                                dVel[0:numParticles], \
                                dReorderedPos[0:numParticles], \
                                dReorderedVel[0:numParticles]) \
                        map(alloc: dHash[0:numParticles], \
                                dIndex[0:numParticles], \
                                dCellStart[0:numGridCells], \
                                dCellEnd[0:numGridCells])
{
  for (int i = 0; i < iterations; i++)
  {
    integrateSystem(
        dPos,
        dVel,
        params,
        timestep,
        numParticles);

    calcHash(
        dHash,
        dIndex,
        dPos,
        params,
        numParticles);

    bitonicSort(dHash, dIndex, dHash, dIndex, 1, numParticles, 0);

    //Find start and end of each cell and
    //Reorder particle data for better cache coherency
    findCellBoundsAndReorder(
        dCellStart,
        dCellEnd,
        dReorderedPos,
        dReorderedVel,
        dHash,
        dIndex,
        dPos,
        dVel,
        numParticles,
        numGridCells);

    collide(
        dVel,
        dReorderedPos,
        dReorderedVel,
        dIndex,
        dCellStart,
        dCellEnd,
        params,
        numParticles,
        numGridCells);
  }

#ifdef DEBUG
  // results should not differ much from those in CUDA SDK 5_Simulation/particles
  // Note the values of certain simulation parameters must be the same
#pragma omp target update from (dVel[0:numParticles])
#pragma omp target update from (dPos[0:numParticles])

  for (unsigned int i = 0; i < numParticles; i++) {
    printf("%d: ", i);
    printf("pos: (%.4f, %.4f, %.4f, %.4f)\n",
        dPos[i].x, dPos[i].y, dPos[i].z, dPos[i].w);
    printf("vel: (%.4f, %.4f, %.4f, %.4f)\n",
        dVel[i].x, dVel[i].y, dVel[i].z, dVel[i].w);
  }
#endif

}

  free(hPos         );
  free(hVel         );
  free(hReorderedPos);
  free(hReorderedVel);
  free(hHash        );
  free(hIndex       );
  free(hCellStart   );
  free(hCellEnd     );
  return EXIT_SUCCESS;
}
