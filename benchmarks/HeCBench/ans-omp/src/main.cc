/*****************************************************************************
 *
 * MULTIANS - Massively parallel ANS decoding on GPUs
 *
 * released under LGPL-3.0
 *
 * 2017-2019 André Weißenberger
 *
 *****************************************************************************/

#include <cassert>
#include <random>
#include <algorithm>
#include <iostream>
#include <iomanip>

#include "multians.h"

// encoder configuration //
#define NUM_SYMBOLS 256
#define NUM_STATES 1024

// seed for PRNG to generate random test data
#define SEED 5

// decoder configuration //

// SUBSEQUENCE_SIZE must be a multiple of 4
#define SUBSEQUENCE_SIZE 4

// number of GPU threads per thread block //
#define THREADS_PER_BLOCK 128

void run(long int input_size) {

  // print column headers
  std::cout << "\u03BB | compressed size (bytes) | ";
  std::cout << std::endl << std::endl;

  for(float lambda = 0.1f; lambda < 2.5f; lambda += 0.16) {

    // vectors to record timings
    std::cout << std::left << std::setw(5) << lambda << std::setfill(' ');

    // generate random, exponentially distributed data
    auto dist = ANSTableGenerator::generate_distribution(
        SEED, NUM_SYMBOLS, NUM_STATES,
        [&](double x) {return lambda * exp(-lambda * x);});

    auto random_data = ANSTableGenerator::generate_test_data(
          dist.dist, input_size, NUM_STATES, SEED);

    // create an ANS table, based on the distribution
    auto table = ANSTableGenerator::generate_table(
        dist.prob, dist.dist, nullptr, NUM_SYMBOLS,
        NUM_STATES);

    // derive an encoder table from the ANS table
    auto encoder_table = ANSTableGenerator::generate_encoder_table(table);

    // derive a decoder table from the ANS table
    auto decoder_table = ANSTableGenerator::get_decoder_table(encoder_table);

    // tANS-encode the generated data using the encoder table
    auto input_buffer = ANSEncoder::encode(
        random_data->data(), input_size, encoder_table);

    // allocate buffer for the decoded output
    auto output_buffer = std::make_shared<CUHDOutputBuffer>(input_size);

    // allocate device buffer for compressed input
    size_t compressed_size = input_buffer->get_compressed_size();
    size_t input_buffer_size = (compressed_size + 4);
    UNIT_TYPE *d_input_buffer = input_buffer->get_compressed_data(); 

    // allocate device buffer for coding table
    size_t decoder_table_size = decoder_table->get_size();
    std::uint32_t *d_decoder_table = reinterpret_cast<std::uint32_t*>(decoder_table->get()) ;

    // allocate device buffer for decompressed output
    // copy decompressed output from the GPU to the host system
    size_t output_buffer_size = output_buffer->get_uncompressed_size();
    SYMBOL_TYPE *d_output_buffer = output_buffer->get_decompressed_data().get();

    size_t num_subseq = SDIV(compressed_size, SUBSEQUENCE_SIZE);
    size_t num_blocks = SDIV(num_subseq, THREADS_PER_BLOCK);

    // allocate device buffer for subsequence synchronization
    // Note the original type is cuhd::CUHDSubsequenceSyncPoint (uint4 is equivalent)
    uint4 *d_sync_info = (uint4*) calloc (num_subseq, sizeof(uint4));

    // allocate device buffer for size of output for each subsequence
    std::uint32_t *d_output_sizes = (std::uint32_t*) malloc (num_subseq * sizeof(std::uint32_t));

    // allocate device buffer for indicating inter-sequence synchronisation
    std::uint8_t *d_sequence_synced = (std::uint8_t*) calloc (num_blocks, sizeof(std::uint8_t));

#pragma omp target data map(to: d_input_buffer[0:input_buffer_size],\
                                d_decoder_table[0:decoder_table_size],\
                                d_sync_info[0:num_subseq],\
                                d_sequence_synced[0:num_blocks]),\
                        map(alloc: d_output_sizes[0:num_subseq]),\
                        map(from: d_output_buffer[0:output_buffer_size])
{
    // decode the compressed data on a GPU
    cuhd::CUHDGPUDecoder::decode(
        d_input_buffer, input_buffer->get_compressed_size(),
        d_output_buffer, output_buffer->get_uncompressed_size(),
        d_decoder_table,
        d_sync_info,
        d_output_sizes,
        d_sequence_synced,
        input_buffer->get_first_state(),
        input_buffer->get_first_bit(), 
        decoder_table->get_num_entries(),
        11, 
        SUBSEQUENCE_SIZE, 
        THREADS_PER_BLOCK);
}

    // reverse all bytes
    output_buffer->reverse();

    // check for errors in decompressed data
    if(cuhd::CUHDUtil::equals(random_data->data(),
          output_buffer->get_decompressed_data().get(), input_size));
    else std::cout << "********* MISMATCH ************" << std::endl;

    // print compressed size (bytes)
    std::cout << std::left << std::setw(10)
      << input_buffer->get_compressed_size() * sizeof(UNIT_TYPE)
      << std::setfill(' ') << std::endl;

    free(d_sync_info);
    free(d_output_sizes);
    free(d_sequence_synced);
  }
}

int main(int argc, char **argv) {

  // name of the binary file
  const char* bin = argv[0];

  auto print_help = [&]() {
    std::cout << "USAGE: " << bin << "<size of input in megabytes> " << std::endl;
  };

  if(argc < 2) {print_help(); return 1;}

  // input size in MB
  const long int size = atoi(argv[1]) * 1024 * 1024;

  if(size < 1) {
    print_help();
    return 1;
  }

  // SUBSEQUENCE_SIZE must be a multiple of 4
  assert(SUBSEQUENCE_SIZE % 4 == 0);

  // run the test
  run(size);

  return 0;
}

