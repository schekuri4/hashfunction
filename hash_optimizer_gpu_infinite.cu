#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iomanip>
#include <thread>
#include <atomic>
#include <signal.h>

using namespace std;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            cout << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error) << endl; \
            exit(1); \
        } \
    } while(0)

// Global variables for graceful shutdown
atomic<bool> keep_running(true);
double global_best_score = 1000.0;
unsigned int global_best_h = 0;
unsigned int global_best_k = 0;

// Signal handler for Ctrl+C
void signal_handler(int signal) {
    cout << "\n\nReceived interrupt signal. Saving results and shutting down gracefully..." << endl;
    keep_running = false;
}

// GPU hash function (matches your current implementation)
__device__ int hash_function_gpu(const char* text, int len, unsigned int h_seed, unsigned int k_seed) {
    unsigned int h = h_seed;
    unsigned int k = k_seed;
    
    for (int i = 0; i < len; i++) {
        h ^= static_cast<unsigned int>(text[i]);
        h *= k;
        h ^= h >> 16;
        k ^= h;  // This line from your current hash.cpp
    }
    
    // Final hash processing
    h *= 0x846ca68b;
    h ^= h >> 16;
    
    return h % 100;
}

// GPU kernel for calculating standard deviation for multiple datasets
__global__ void calculate_multi_dataset_std_dev_kernel(
    char** datasets, int** string_lengths, int** string_offsets, int* num_strings_per_dataset,
    int num_datasets, unsigned int* h_seeds, unsigned int* k_seeds, 
    double* results, int num_tests) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tests) return;
    
    unsigned int h_seed = h_seeds[idx];
    unsigned int k_seed = k_seeds[idx];
    
    double total_std_dev = 0.0;
    
    // Test on all datasets
    for (int dataset_idx = 0; dataset_idx < num_datasets; dataset_idx++) {
        int buckets[100] = {0};
        int num_strings = num_strings_per_dataset[dataset_idx];
        
        // Hash all strings in this dataset
        for (int i = 0; i < num_strings; i++) {
            char* str_start = datasets[dataset_idx] + string_offsets[dataset_idx][i];
            int str_len = string_lengths[dataset_idx][i];
            int bucket = hash_function_gpu(str_start, str_len, h_seed, k_seed);
            buckets[bucket]++;
        }
        
        // Calculate standard deviation for this dataset
        double mean = static_cast<double>(num_strings) / 100.0;
        double variance = 0.0;
        
        for (int i = 0; i < 100; i++) {
            double diff = buckets[i] - mean;
            variance += diff * diff;
        }
        variance /= 100.0;
        total_std_dev += sqrt(variance);
    }
    
    // Store average standard deviation across all datasets
    results[idx] = total_std_dev / num_datasets;
}

// GPU kernel for random seed generation
__global__ void generate_seeds_kernel(unsigned int* h_seeds, unsigned int* k_seeds, 
                                     int num_tests, unsigned long long seed_offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tests) return;
    
    curandState state;
    curand_init(seed_offset + idx, 0, 0, &state);
    
    h_seeds[idx] = curand(&state);
    k_seeds[idx] = curand(&state);
}

// Structure for GPU dataset
struct GPUDataset {
    char* d_data;
    int* d_lengths;
    int* d_offsets;
    int num_strings;
    int total_chars;
};

GPUDataset prepare_dataset_for_gpu(const vector<string>& dataset) {
    GPUDataset gpu_dataset;
    gpu_dataset.num_strings = dataset.size();
    
    // Calculate total characters needed
    gpu_dataset.total_chars = 0;
    for (const string& str : dataset) {
        gpu_dataset.total_chars += str.length();
    }
    
    // Prepare host data
    vector<char> all_chars;
    vector<int> lengths;
    vector<int> offsets;
    
    int current_offset = 0;
    for (const string& str : dataset) {
        offsets.push_back(current_offset);
        lengths.push_back(str.length());
        
        for (char c : str) {
            all_chars.push_back(c);
        }
        current_offset += str.length();
    }
    
    // Allocate GPU memory
    CUDA_CHECK(cudaMalloc(&gpu_dataset.d_data, gpu_dataset.total_chars * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&gpu_dataset.d_lengths, gpu_dataset.num_strings * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gpu_dataset.d_offsets, gpu_dataset.num_strings * sizeof(int)));
    
    // Copy to GPU
    CUDA_CHECK(cudaMemcpy(gpu_dataset.d_data, all_chars.data(), 
                         gpu_dataset.total_chars * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_dataset.d_lengths, lengths.data(), 
                         gpu_dataset.num_strings * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_dataset.d_offsets, offsets.data(), 
                         gpu_dataset.num_strings * sizeof(int), cudaMemcpyHostToDevice));
    
    return gpu_dataset;
}

void free_gpu_dataset(GPUDataset& dataset) {
    cudaFree(dataset.d_data);
    cudaFree(dataset.d_lengths);
    cudaFree(dataset.d_offsets);
}

vector<string> load_dataset(const string& filename) {
    vector<string> dataset;
    ifstream file(filename);
    string line;
    
    while (getline(file, line)) {
        if (!line.empty()) {
            dataset.push_back(line);
        }
    }
    
    return dataset;
}

void save_best_result(double best_score, unsigned int best_h, unsigned int best_k, 
                     long long total_tests, int runtime_seconds) {
    ofstream file("gpu_infinite_best_result.txt");
    file << "=== INFINITE GPU OPTIMIZATION BEST RESULT ===" << endl;
    file << "Best h_seed: " << best_h << " (0x" << hex << best_h << dec << ")" << endl;
    file << "Best k_seed: " << best_k << " (0x" << hex << best_k << dec << ")" << endl;
    file << "Average Standard Deviation: " << fixed << setprecision(6) << best_score << endl;
    file << "Total tests completed: " << total_tests << endl;
    file << "Runtime: " << runtime_seconds << " seconds" << endl;
    file << "Tests per second: " << (total_tests / max(1, runtime_seconds)) << endl;
    file.close();
}

int main() {
    // Set up signal handler for graceful shutdown
    signal(SIGINT, signal_handler);
    
    cout << "=== INFINITE GPU HASH FUNCTION OPTIMIZER ===" << endl;
    cout << "Press Ctrl+C to stop and save the best result found" << endl;
    
    // Check CUDA availability
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        cout << "No CUDA devices found!" << endl;
        return 1;
    }
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    cout << "\nUsing GPU: " << prop.name << endl;
    cout << "Compute capability: " << prop.major << "." << prop.minor << endl;
    cout << "Max threads per block: " << prop.maxThreadsPerBlock << endl;
    cout << "Global memory: " << (prop.totalGlobalMem / 1024 / 1024) << " MB" << endl;
    
    // Load datasets
    vector<string> dataset_names = {
        "inputs/sample_input.txt",
        "inputs/atoz.txt", 
        "inputs/common500.txt",
        "inputs/bertuncased.txt",
        "inputs/mit_a.txt"
    };
    
    vector<vector<string>> datasets;
    vector<GPUDataset> gpu_datasets;
    
    cout << "\nLoading datasets..." << endl;
    for (const string& name : dataset_names) {
        vector<string> dataset = load_dataset(name);
        if (dataset.empty()) {
            cout << "Warning: Could not load " << name << endl;
            continue;
        }
        datasets.push_back(dataset);
        gpu_datasets.push_back(prepare_dataset_for_gpu(dataset));
        cout << "Loaded " << dataset.size() << " entries from " << name << endl;
    }
    
    if (datasets.empty()) {
        cout << "Error: No datasets loaded!" << endl;
        return 1;
    }
    
    // Configuration for maximum GPU utilization
    int tests_per_batch = 1000000;  // 1M tests per batch
    int threads_per_block = 256;
    int blocks = (tests_per_batch + threads_per_block - 1) / threads_per_block;
    
    cout << "\nGPU Configuration:" << endl;
    cout << "- Tests per batch: " << tests_per_batch << endl;
    cout << "- Threads per block: " << threads_per_block << endl;
    cout << "- Number of blocks: " << blocks << endl;
    cout << "- Datasets: " << datasets.size() << endl;
    
    // Prepare GPU memory for multi-dataset processing
    vector<char*> h_dataset_ptrs(gpu_datasets.size());
    vector<int*> h_lengths_ptrs(gpu_datasets.size());
    vector<int*> h_offsets_ptrs(gpu_datasets.size());
    vector<int> h_num_strings(gpu_datasets.size());
    
    for (size_t i = 0; i < gpu_datasets.size(); i++) {
        h_dataset_ptrs[i] = gpu_datasets[i].d_data;
        h_lengths_ptrs[i] = gpu_datasets[i].d_lengths;
        h_offsets_ptrs[i] = gpu_datasets[i].d_offsets;
        h_num_strings[i] = gpu_datasets[i].num_strings;
    }
    
    // Allocate GPU memory for dataset pointers
    char** d_dataset_ptrs;
    int** d_lengths_ptrs;
    int** d_offsets_ptrs;
    int* d_num_strings;
    
    CUDA_CHECK(cudaMalloc(&d_dataset_ptrs, gpu_datasets.size() * sizeof(char*)));
    CUDA_CHECK(cudaMalloc(&d_lengths_ptrs, gpu_datasets.size() * sizeof(int*)));
    CUDA_CHECK(cudaMalloc(&d_offsets_ptrs, gpu_datasets.size() * sizeof(int*)));
    CUDA_CHECK(cudaMalloc(&d_num_strings, gpu_datasets.size() * sizeof(int)));
    
    CUDA_CHECK(cudaMemcpy(d_dataset_ptrs, h_dataset_ptrs.data(), 
                         gpu_datasets.size() * sizeof(char*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lengths_ptrs, h_lengths_ptrs.data(), 
                         gpu_datasets.size() * sizeof(int*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offsets_ptrs, h_offsets_ptrs.data(), 
                         gpu_datasets.size() * sizeof(int*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_num_strings, h_num_strings.data(), 
                         gpu_datasets.size() * sizeof(int), cudaMemcpyHostToDevice));
    
    // Allocate GPU memory for seeds and results
    unsigned int *d_h_seeds, *d_k_seeds;
    double *d_results;
    
    CUDA_CHECK(cudaMalloc(&d_h_seeds, tests_per_batch * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_k_seeds, tests_per_batch * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_results, tests_per_batch * sizeof(double)));
    
    // Host memory for results
    vector<double> h_results(tests_per_batch);
    vector<unsigned int> h_h_seeds(tests_per_batch);
    vector<unsigned int> h_k_seeds(tests_per_batch);
    
    auto start_time = chrono::high_resolution_clock::now();
    long long total_tests = 0;
    int batch_count = 0;
    unsigned long long seed_offset = 0;
    
    cout << "\n🚀 Starting infinite optimization..." << endl;
    cout << "Current best: " << fixed << setprecision(6) << global_best_score << endl;
    
    while (keep_running) {
        batch_count++;
        
        // Generate random seeds
        generate_seeds_kernel<<<blocks, threads_per_block>>>(
            d_h_seeds, d_k_seeds, tests_per_batch, seed_offset);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Run optimization kernel
        calculate_multi_dataset_std_dev_kernel<<<blocks, threads_per_block>>>(
            d_dataset_ptrs, d_lengths_ptrs, d_offsets_ptrs, d_num_strings,
            gpu_datasets.size(), d_h_seeds, d_k_seeds, d_results, tests_per_batch);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Copy results back to host
        CUDA_CHECK(cudaMemcpy(h_results.data(), d_results, 
                             tests_per_batch * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_h_seeds.data(), d_h_seeds, 
                             tests_per_batch * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_k_seeds.data(), d_k_seeds, 
                             tests_per_batch * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        
        // Find best result in this batch
        auto min_it = min_element(h_results.begin(), h_results.end());
        int best_idx = distance(h_results.begin(), min_it);
        double batch_best = *min_it;
        
        total_tests += tests_per_batch;
        
        // Update global best if we found something better
        if (batch_best < global_best_score) {
            global_best_score = batch_best;
            global_best_h = h_h_seeds[best_idx];
            global_best_k = h_k_seeds[best_idx];
            
            auto current_time = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::seconds>(current_time - start_time);
            
            cout << "\n🎉 NEW BEST FOUND!" << endl;
            cout << "Batch: " << batch_count << " | Tests: " << total_tests << endl;
            cout << "Score: " << fixed << setprecision(6) << global_best_score << endl;
            cout << "h_seed: " << global_best_h << " (0x" << hex << global_best_h << dec << ")" << endl;
            cout << "k_seed: " << global_best_k << " (0x" << hex << global_best_k << dec << ")" << endl;
            cout << "Runtime: " << duration.count() << "s | Rate: " 
                 << (total_tests / max(1, (int)duration.count())) << " tests/sec" << endl;
            
            // Save immediately when we find a better result
            save_best_result(global_best_score, global_best_h, global_best_k, 
                           total_tests, duration.count());
        }
        
        // Progress update every 10 batches
        if (batch_count % 10 == 0) {
            auto current_time = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::seconds>(current_time - start_time);
            
            cout << "Batch " << batch_count << " | " << total_tests << " tests | " 
                 << duration.count() << "s | Best: " << fixed << setprecision(6) 
                 << global_best_score << " | Rate: " 
                 << (total_tests / max(1, (int)duration.count())) << " tests/sec" << endl;
        }
        
        seed_offset += tests_per_batch;
    }
    
    // Final save and cleanup
    auto end_time = chrono::high_resolution_clock::now();
    auto total_duration = chrono::duration_cast<chrono::seconds>(end_time - start_time);
    
    cout << "\n=== INFINITE OPTIMIZATION STOPPED ===" << endl;
    cout << "Total runtime: " << total_duration.count() << " seconds" << endl;
    cout << "Total tests: " << total_tests << endl;
    cout << "Average tests per second: " << (total_tests / max(1, (int)total_duration.count())) << endl;
    
    cout << "\n🏆 FINAL BEST RESULT:" << endl;
    cout << "h_seed: " << global_best_h << " (0x" << hex << global_best_h << dec << ")" << endl;
    cout << "k_seed: " << global_best_k << " (0x" << hex << global_best_k << dec << ")" << endl;
    cout << "Average Standard Deviation: " << fixed << setprecision(6) << global_best_score << endl;
    
    save_best_result(global_best_score, global_best_h, global_best_k, 
                    total_tests, total_duration.count());
    
    cout << "\nResults saved to gpu_infinite_best_result.txt" << endl;
    
    // Cleanup
    for (auto& gpu_dataset : gpu_datasets) {
        free_gpu_dataset(gpu_dataset);
    }
    
    cudaFree(d_dataset_ptrs);
    cudaFree(d_lengths_ptrs);
    cudaFree(d_offsets_ptrs);
    cudaFree(d_num_strings);
    cudaFree(d_h_seeds);
    cudaFree(d_k_seeds);
    cudaFree(d_results);
    
    return 0;
}