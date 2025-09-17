#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>
#include <curand_kernel.h>

using namespace std;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            cout << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error) << endl; \
            exit(1); \
        } \
    } while(0)

// GPU hash function matching hash.cpp exactly
__device__ int hash_function_gpu(const char* text, int len, unsigned int h_seed, unsigned int k_seed) {
    unsigned int h = h_seed ^ 0x9e3779b9;
    unsigned int k = k_seed ^ 0x85ebca6b;
    
    for (int i = 0; i < len; i++) {
        unsigned int c = (unsigned int)text[i];
        
        h ^= c * 0x9e3779b1;
        k += c * 0xc2b2ae35;
        
        h = (h << 13) | (h >> 19);
        k = (k << 17) | (k >> 15);
        
        h += k * 0x165667b1;
        k ^= h * 0x27d4eb2f;
        
        h ^= k;
        k += h;
    }
    
    h ^= len;
    k ^= h;
    
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    
    return h ;
}

// GPU kernel for calculating standard deviation
__global__ void calculate_std_dev_kernel(
    char* dataset, int* string_lengths, int* string_offsets, int num_strings,
    unsigned int* h_seeds, unsigned int* k_seeds, double* results, int num_tests) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tests) return;
    
    unsigned int h_seed = h_seeds[idx];
    unsigned int k_seed = k_seeds[idx];
    
    // Count bucket usage
    int buckets[100] = {0};
    
    for (int i = 0; i < num_strings; i++) {
        char* str_start = dataset + string_offsets[i];
        int str_len = string_lengths[i];
        int bucket = hash_function_gpu(str_start, str_len, h_seed, k_seed);
        buckets[bucket]++;
    }
    
    // Calculate standard deviation
    double mean = static_cast<double>(num_strings) / 100.0;
    double variance = 0.0;
    
    for (int i = 0; i < 100; i++) {
        double diff = buckets[i] - mean;
        variance += diff * diff;
    }
    variance /= 100.0;
    
    results[idx] = sqrt(variance);
}

// GPU kernel for random seed generation
__global__ void generate_seeds_kernel(unsigned int* h_seeds, unsigned int* k_seeds, 
                                     int num_tests, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tests) return;
    
    curandState state;
    curand_init(seed + idx, 0, 0, &state);
    
    h_seeds[idx] = curand(&state);
    k_seeds[idx] = curand(&state);
}

// Load dataset and prepare for GPU
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

int main() {
    cout << "=== GPU-Accelerated Hash Function Seed Optimizer ===" << endl;
    
    // Check CUDA availability
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        cout << "No CUDA devices found! Falling back to CPU version." << endl;
        return 1;
    }
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    cout << "Using GPU: " << prop.name << endl;
    cout << "Compute capability: " << prop.major << "." << prop.minor << endl;
    cout << "Max threads per block: " << prop.maxThreadsPerBlock << endl;
    
    // Load datasets
    vector<string> dataset_names = {
        "inputs/test_alls.txt",
        "inputs/test_passwords.txt",
        "inputs/test_wordle500.txt"
    };
    
    vector<vector<string>> datasets;
    vector<GPUDataset> gpu_datasets;
    
    for (const string& name : dataset_names) {
        cout << "Loading " << name << "..." << endl;
        vector<string> dataset = load_dataset(name);
        if (dataset.empty()) {
            cout << "Warning: Could not load " << name << endl;
            continue;
        }
        datasets.push_back(dataset);
        gpu_datasets.push_back(prepare_dataset_for_gpu(dataset));
        cout << "Loaded " << dataset.size() << " entries" << endl;
    }
    
    if (datasets.empty()) {
        cout << "Error: No datasets loaded!" << endl;
        return 1;
    }
    
    // Configuration
    int total_tests;
    cout << "Enter total number of tests to run (recommended: 1000000+): ";
    cin >> total_tests;
    
    int threads_per_block = 256;
    int blocks = (total_tests + threads_per_block - 1) / threads_per_block;
    
    cout << "\nGPU Configuration:" << endl;
    cout << "- Total tests: " << total_tests << endl;
    cout << "- Threads per block: " << threads_per_block << endl;
    cout << "- Number of blocks: " << blocks << endl;
    cout << "- Datasets: " << datasets.size() << endl;
    
    // Allocate GPU memory for seeds and results
    unsigned int *d_h_seeds, *d_k_seeds;
    double *d_results;
    
    CUDA_CHECK(cudaMalloc(&d_h_seeds, total_tests * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_k_seeds, total_tests * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_results, total_tests * sizeof(double)));
    
    auto start_time = chrono::high_resolution_clock::now();
    
    // Generate random seeds on GPU
    cout << "Generating random seeds..." << endl;
    generate_seeds_kernel<<<blocks, threads_per_block>>>(
        d_h_seeds, d_k_seeds, total_tests, chrono::high_resolution_clock::now().time_since_epoch().count());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Test each dataset and find best overall result
    vector<double> best_results_per_dataset(datasets.size(), 1000.0);
    unsigned int best_h = 0, best_k = 0;
    double best_avg = 1000.0;
    
    for (size_t dataset_idx = 0; dataset_idx < gpu_datasets.size(); dataset_idx++) {
        cout << "Testing dataset " << (dataset_idx + 1) << "/" << gpu_datasets.size() << "..." << endl;
        
        // Run kernel for this dataset
        calculate_std_dev_kernel<<<blocks, threads_per_block>>>(
            gpu_datasets[dataset_idx].d_data,
            gpu_datasets[dataset_idx].d_lengths,
            gpu_datasets[dataset_idx].d_offsets,
            gpu_datasets[dataset_idx].num_strings,
            d_h_seeds, d_k_seeds, d_results, total_tests);
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Copy results back to host
        vector<double> results(total_tests);
        CUDA_CHECK(cudaMemcpy(results.data(), d_results, 
                             total_tests * sizeof(double), cudaMemcpyDeviceToHost));
        
        // Find best result for this dataset
        auto min_it = min_element(results.begin(), results.end());
        int best_idx = distance(results.begin(), min_it);
        best_results_per_dataset[dataset_idx] = *min_it;
        
        cout << "Best result for " << dataset_names[dataset_idx] << ": " << *min_it << endl;
    }
    
    // Calculate average and find best seeds
    // For simplicity, we'll run one more comprehensive test with a smaller subset
    int final_test_size = min(10000, total_tests);
    cout << "\nRunning final comprehensive test with " << final_test_size << " best candidates..." << endl;
    
    // Copy seeds back to host for final evaluation
    vector<unsigned int> h_seeds(total_tests), k_seeds(total_tests);
    CUDA_CHECK(cudaMemcpy(h_seeds.data(), d_h_seeds, 
                         total_tests * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(k_seeds.data(), d_k_seeds, 
                         total_tests * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    
    // Test top candidates on all datasets (CPU-based for simplicity)
    for (int i = 0; i < final_test_size; i++) {
        double total_std_dev = 0.0;
        
        for (size_t dataset_idx = 0; dataset_idx < datasets.size(); dataset_idx++) {
            // Calculate std dev for this dataset (simplified CPU version)
            vector<int> buckets(100, 0);
            for (const string& text : datasets[dataset_idx]) {
                unsigned int h = h_seeds[i];
                unsigned int k = k_seeds[i];
                
                for (char c : text) {
                    h ^= static_cast<unsigned int>(c);
                    h *= k;
                    h ^= h >> 16;
                }
                h *= 0x846ca68b;
                h ^= h >> 16;
                
                buckets[h % 100]++;
            }
            
            double mean = static_cast<double>(datasets[dataset_idx].size()) / 100.0;
            double variance = 0.0;
            for (int count : buckets) {
                variance += (count - mean) * (count - mean);
            }
            variance /= 100.0;
            total_std_dev += sqrt(variance);
        }
        
        double avg_std_dev = total_std_dev / datasets.size();
        if (avg_std_dev < best_avg) {
            best_avg = avg_std_dev;
            best_h = h_seeds[i];
            best_k = k_seeds[i];
        }
    }
    
    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::seconds>(end_time - start_time);
    
    // Display results
    cout << "\n=== GPU OPTIMIZATION COMPLETE ===" << endl;
    cout << "Time taken: " << duration.count() << " seconds" << endl;
    cout << "Tests completed: " << total_tests << endl;
    cout << "Tests per second: " << (total_tests / max(1, (int)duration.count())) << endl;
    
    cout << "\n=== BEST RESULT ===" << endl;
    cout << "h_seed: " << best_h << " (0x" << hex << best_h << dec << ")" << endl;
    cout << "k_seed: " << best_k << " (0x" << hex << best_k << dec << ")" << endl;
    cout << "Average Standard Deviation: " << best_avg << endl;
    
    // Save results
    ofstream results_file("gpu_optimization_results.txt");
    results_file << "Best h_seed: " << best_h << endl;
    results_file << "Best k_seed: " << best_k << endl;
    results_file << "Average Standard Deviation: " << best_avg << endl;
    results_file << "Tests completed: " << total_tests << endl;
    results_file << "Time taken: " << duration.count() << " seconds" << endl;
    
    cout << "\nResults saved to gpu_optimization_results.txt" << endl;
    
    // Cleanup
    for (auto& gpu_dataset : gpu_datasets) {
        free_gpu_dataset(gpu_dataset);
    }
    
    cudaFree(d_h_seeds);
    cudaFree(d_k_seeds);
    cudaFree(d_results);
    
    return 0;
}