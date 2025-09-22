#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
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
string output_filename;
chrono::system_clock::time_point start_timestamp;

// Signal handler for Ctrl+C
void signal_handler(int signal) {
    cout << "\n\nReceived interrupt signal. Saving results and shutting down gracefully..." << endl;
    keep_running = false;
}

// New optimized GPU hash function
__device__ unsigned int hash_function_gpu(const char* text, int len, unsigned int h_seed, unsigned int k_seed) {
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
    
    return h;
}

// GPU kernel for calculating standard deviation for multiple datasets
__global__ void calculate_multi_dataset_std_dev_kernel(
    char** datasets, int** string_lengths, int** string_offsets, int* num_strings_per_dataset,
    int* table_sizes, int num_datasets, unsigned int* h_seeds, unsigned int* k_seeds, 
    double* results, int num_tests) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tests) return;
    
    unsigned int h_seed = h_seeds[idx];
    unsigned int k_seed = k_seeds[idx];
    
    double total_std_dev = 0.0;
    
    // Test on all datasets
    for (int dataset_idx = 0; dataset_idx < num_datasets; dataset_idx++) {
        // Use fixed-size array with safe maximum for GPU stack memory
        const int MAX_BUCKETS = 5000;  // Reduced to prevent stack overflow
        int buckets[MAX_BUCKETS];
        
        // Use the actual table size for this dataset (bounded by MAX_BUCKETS)
        int table_size = min(table_sizes[dataset_idx], MAX_BUCKETS);
        for (int i = 0; i < table_size; i++) buckets[i] = 0;
        
        int num_strings = num_strings_per_dataset[dataset_idx];
        
        // Hash all strings in this dataset
        for (int i = 0; i < num_strings; i++) {
            char* str_start = datasets[dataset_idx] + string_offsets[dataset_idx][i];
            int str_len = string_lengths[dataset_idx][i];
            unsigned int hash_val = hash_function_gpu(str_start, str_len, h_seed, k_seed);
            // Use unsigned arithmetic to avoid negative modulo issues
            unsigned int bucket = hash_val % table_size;
            buckets[bucket]++;
        }
        
        // Calculate standard deviation for this dataset
        double mean = static_cast<double>(num_strings) / static_cast<double>(table_size);
        double variance = 0.0;
        
        for (int i = 0; i < table_size; i++) {
            double diff = static_cast<double>(buckets[i]) - mean;
            variance += diff * diff;
        }
        variance /= static_cast<double>(table_size);
        total_std_dev += sqrt(variance);
    }
    
    // Store average standard deviation across all datasets
    results[idx] = total_std_dev / num_datasets;
}

// Improved GPU kernel for high-quality random seed generation
__global__ void generate_seeds_kernel(unsigned int* h_seeds, unsigned int* k_seeds, 
                                     int num_tests, unsigned long long seed_offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tests) return;
    
    curandState state;
    // Use more diverse seeding: combine global offset, block ID, thread ID, and time-based component
    unsigned long long diverse_seed = seed_offset + 
                                     (blockIdx.x * 1000000ULL) + 
                                     (threadIdx.x * 1000ULL) + 
                                     ((seed_offset >> 16) * idx);
    curand_init(diverse_seed, idx, 0, &state);
    
    // Generate multiple random numbers and use XOR mixing for better quality
    unsigned int r1 = curand(&state);
    unsigned int r2 = curand(&state);
    unsigned int r3 = curand(&state);
    unsigned int r4 = curand(&state);
    
    // Mix the random numbers for better distribution
    h_seeds[idx] = r1 ^ (r2 << 16) ^ (r3 >> 8);
    k_seeds[idx] = r2 ^ (r4 << 12) ^ (r1 >> 4);
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

pair<vector<string>, int> load_dataset(const string& filename) {
    vector<string> dataset;
    int table_size = 0;
    ifstream file(filename);
    
    if (!file.is_open()) {
        return {dataset, table_size};
    }
    
    // Read table size from first line
    file >> table_size;
    file.ignore(); // ignore newline after number
    
    string line;
    while (getline(file, line)) {
        if (!line.empty()) {
            dataset.push_back(line);
        }
    }
    
    return {dataset, table_size};
}

string extract_hash_function() {
    ifstream source_file(__FILE__);
    string line;
    string hash_function = "";
    bool in_function = false;
    int brace_count = 0;
    
    while (getline(source_file, line)) {
        // Look for the start of the hash function
        if (line.find("__device__ unsigned int hash_function_gpu") != string::npos) {
            in_function = true;
            hash_function += line + "\n";
            if (line.find("{") != string::npos) {
                brace_count = 1;
            }
            continue;
        }
        
        if (in_function) {
            hash_function += line + "\n";
            
            // Count braces to find the end of the function
            for (char c : line) {
                if (c == '{') brace_count++;
                else if (c == '}') brace_count--;
            }
            
            // If we've closed all braces, we're done
            if (brace_count == 0) {
                break;
            }
        }
    }
    
    source_file.close();
    return hash_function;
}

void save_best_result(double best_score, unsigned int best_h, unsigned int best_k, 
                     long long total_tests, int runtime_seconds) {
    // Get current time for this result
    auto now = chrono::system_clock::now();
    auto time_t = chrono::system_clock::to_time_t(now);
    
    // Append to the single output file
    ofstream file(output_filename, ios::app);
    file << "[" << put_time(localtime(&time_t), "%Y-%m-%d %H:%M:%S") << "] "
         << "GPU NEW BEST: Score=" << fixed << setprecision(6) << best_score 
         << " | Hash: h=" << best_h << "(0x" << hex << best_h << dec 
         << "), k=" << best_k << "(0x" << hex << best_k << dec 
         << ") | Tests=" << total_tests << " | Runtime=" << runtime_seconds 
         << "s | Rate=" << (total_tests / max(1, runtime_seconds)) << "/s" << endl;
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
    
    // Load datasets - include ALL input files
    vector<string> dataset_names = {
        "inputs/1.txt",
        "inputs/2.txt",
        "inputs/3.txt",
        "inputs/4.txt",
        "inputs/5.txt",
        "inputs/atoz.txt",
        "inputs/bertuncased.txt",
        "inputs/common500.txt",
        "inputs/mit_a.txt",
        "inputs/sample_input.txt",
        "inputs/test_alls.txt",
        "inputs/test_passwords.txt",
        "inputs/test_wordle500.txt"
    };
    
    vector<vector<string>> datasets;
    vector<int> table_sizes;
    vector<GPUDataset> gpu_datasets;
    
    cout << "\nLoading datasets..." << endl;
    for (const string& name : dataset_names) {
        pair<vector<string>, int> result = load_dataset(name);
        vector<string> dataset = result.first;
        int table_size = result.second;
        if (dataset.empty()) {
            cout << "Warning: Could not load " << name << endl;
            continue;
        }
        datasets.push_back(dataset);
        table_sizes.push_back(table_size);
        gpu_datasets.push_back(prepare_dataset_for_gpu(dataset));
        cout << "Loaded " << dataset.size() << " entries from " << name << " (table size: " << table_size << ")" << endl;
    }
    
    // Initialize output file with start timestamp
    start_timestamp = chrono::system_clock::now();
    auto start_time_t = chrono::system_clock::to_time_t(start_timestamp);
    
    stringstream filename;
    filename << "hash_optimization_results_" 
             << put_time(localtime(&start_time_t), "%Y%m%d_%H%M%S") << ".txt";
    output_filename = filename.str();
    
    ofstream file(output_filename);
    file << "=== HASH FUNCTION OPTIMIZATION RESULTS ===" << endl;
    file << "Started: " << put_time(localtime(&start_time_t), "%Y-%m-%d %H:%M:%S") << endl;
    file << "Hash Function Implementation:" << endl;
    file << extract_hash_function();
    file << "Datasets: " << datasets.size() << " loaded" << endl;
    file << "=== OPTIMIZATION PROGRESS ===" << endl;
    file.close();
    
    cout << "\nResults will be saved to: " << output_filename << endl;
    
    if (datasets.empty()) {
        cout << "Error: No datasets loaded!" << endl;
        return 1;
    }
    
    cout << "\nUsing individual table sizes for each dataset:" << endl;
    int total_strings = 0;
    for (size_t i = 0; i < datasets.size(); i++) {
        total_strings += datasets[i].size();
        cout << "- Dataset " << (i+1) << ": " << datasets[i].size() << " strings, table size: " << table_sizes[i] << endl;
    }
    cout << "- Total strings across all datasets: " << total_strings << endl;
    
    // Adaptive configuration for better exploration
    int base_tests_per_batch = 250000;  // Start with smaller batches for faster feedback
    int current_tests_per_batch = base_tests_per_batch;
    int max_tests_per_batch = 1000000;
    int threads_per_block = 256;
    int blocks = (current_tests_per_batch + threads_per_block - 1) / threads_per_block;
    
    cout << "\nGPU Configuration:" << endl;
    cout << "- Initial tests per batch: " << current_tests_per_batch << " (adaptive)" << endl;
    cout << "- Max tests per batch: " << max_tests_per_batch << endl;
    cout << "- Threads per block: " << threads_per_block << endl;
    cout << "- Initial number of blocks: " << blocks << endl;
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
    int* d_table_sizes;
    
    CUDA_CHECK(cudaMalloc(&d_dataset_ptrs, gpu_datasets.size() * sizeof(char*)));
    CUDA_CHECK(cudaMalloc(&d_lengths_ptrs, gpu_datasets.size() * sizeof(int*)));
    CUDA_CHECK(cudaMalloc(&d_offsets_ptrs, gpu_datasets.size() * sizeof(int*)));
    CUDA_CHECK(cudaMalloc(&d_num_strings, gpu_datasets.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_table_sizes, gpu_datasets.size() * sizeof(int)));
    
    CUDA_CHECK(cudaMemcpy(d_dataset_ptrs, h_dataset_ptrs.data(), 
                         gpu_datasets.size() * sizeof(char*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lengths_ptrs, h_lengths_ptrs.data(), 
                         gpu_datasets.size() * sizeof(int*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offsets_ptrs, h_offsets_ptrs.data(), 
                         gpu_datasets.size() * sizeof(int*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_num_strings, h_num_strings.data(), 
                         gpu_datasets.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_table_sizes, table_sizes.data(), 
                         gpu_datasets.size() * sizeof(int), cudaMemcpyHostToDevice));
    
    // Allocate GPU memory for seeds and results (use max size for buffers)
    unsigned int *d_h_seeds, *d_k_seeds;
    double *d_results;
    
    CUDA_CHECK(cudaMalloc(&d_h_seeds, max_tests_per_batch * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_k_seeds, max_tests_per_batch * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_results, max_tests_per_batch * sizeof(double)));
    
    // Host memory for results (use max size for buffers)
    vector<double> h_results(max_tests_per_batch);
    vector<unsigned int> h_h_seeds(max_tests_per_batch);
    vector<unsigned int> h_k_seeds(max_tests_per_batch);
    
    auto start_time = chrono::high_resolution_clock::now();
    long long total_tests = 0;
    int batch_count = 0;
    unsigned long long seed_offset = 0;
    int batches_since_improvement = 0;
    double last_best_score = global_best_score;
    
    cout << "\n🚀 Starting adaptive infinite optimization..." << endl;
    cout << "Current best: " << fixed << setprecision(6) << global_best_score << endl;
    
    while (keep_running) {
        batch_count++;
        
        // Adaptive batch sizing: increase batch size if no improvement for a while
        if (batches_since_improvement > 20 && current_tests_per_batch < max_tests_per_batch) {
            current_tests_per_batch = min(max_tests_per_batch, current_tests_per_batch * 2);
            blocks = (current_tests_per_batch + threads_per_block - 1) / threads_per_block;
            batches_since_improvement = 0;
            cout << "📈 Increasing batch size to " << current_tests_per_batch << " tests" << endl;
        }
        
        // Generate random seeds with time-based diversity
        generate_seeds_kernel<<<blocks, threads_per_block>>>(
            d_h_seeds, d_k_seeds, current_tests_per_batch, seed_offset);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Run optimization kernel
        calculate_multi_dataset_std_dev_kernel<<<blocks, threads_per_block>>>(
            d_dataset_ptrs, d_lengths_ptrs, d_offsets_ptrs, d_num_strings,
            d_table_sizes, gpu_datasets.size(), d_h_seeds, d_k_seeds, d_results, current_tests_per_batch);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Copy only results first to find the best score
        CUDA_CHECK(cudaMemcpy(h_results.data(), d_results, 
                             current_tests_per_batch * sizeof(double), cudaMemcpyDeviceToHost));
        
        // Find best result in this batch
        auto min_it = min_element(h_results.begin(), h_results.begin() + current_tests_per_batch);
        int best_idx = distance(h_results.begin(), min_it);
        double batch_best = *min_it;
        
        // Only copy seed data if we found an improvement (saves bandwidth)
        bool need_seeds = (batch_best < global_best_score);
        if (need_seeds) {
            CUDA_CHECK(cudaMemcpy(h_h_seeds.data(), d_h_seeds, 
                                 current_tests_per_batch * sizeof(unsigned int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_k_seeds.data(), d_k_seeds, 
                                 current_tests_per_batch * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        }
        
        total_tests += current_tests_per_batch;
        
        // Update global best if we found something better
        bool found_improvement = false;
        if (batch_best < global_best_score) {
            global_best_score = batch_best;
            global_best_h = h_h_seeds[best_idx];
            global_best_k = h_k_seeds[best_idx];
            found_improvement = true;
            batches_since_improvement = 0;
            
            auto current_time = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::seconds>(current_time - start_time);
            
            // Get current timestamp
            auto now = chrono::system_clock::now();
            auto time_t = chrono::system_clock::to_time_t(now);
            
            cout << "\n🎉 [" << put_time(localtime(&time_t), "%Y-%m-%d %H:%M:%S") 
                 << "] NEW BEST: Score=" << fixed << setprecision(6) << global_best_score 
                 << " | Hash: h=" << global_best_h << "(0x" << hex << global_best_h << dec 
                 << "), k=" << global_best_k << "(0x" << hex << global_best_k << dec 
                 << ") | Function: h=h_seed^0x9e3779b9, k=k_seed^0x85ebca6b | Tests=" << total_tests 
                 << " | Runtime=" << duration.count() << "s | Rate=" 
                 << (total_tests / max(1, (int)duration.count())) << "/s" << endl;
            
            // Save immediately when we find a better result
            save_best_result(global_best_score, global_best_h, global_best_k, 
                           total_tests, duration.count());
        } else {
            batches_since_improvement++;
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
        
        // Update seed offset for next batch (use current batch size)
        seed_offset += current_tests_per_batch;
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
    
    // Cleanup
    for (auto& gpu_dataset : gpu_datasets) {
        free_gpu_dataset(gpu_dataset);
    }
    
    cudaFree(d_dataset_ptrs);
    cudaFree(d_lengths_ptrs);
    cudaFree(d_offsets_ptrs);
    cudaFree(d_num_strings);
    cudaFree(d_table_sizes);
    cudaFree(d_h_seeds);
    cudaFree(d_k_seeds);
    cudaFree(d_results);
    
    return 0;
}