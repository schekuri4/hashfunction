#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <random>
#include <thread>
#include <atomic>
#include <iomanip>
#include <signal.h>
#include <mutex>

using namespace std;

// Global variables for graceful shutdown and thread safety
atomic<bool> keep_running(true);
atomic<double> global_best_score(1000.0);
atomic<unsigned int> global_best_h(0);
atomic<unsigned int> global_best_k(0);
atomic<long long> total_tests_completed(0);
mutex output_mutex;

// Signal handler for Ctrl+C
void signal_handler(int signal) {
    cout << "\n\nReceived interrupt signal. Saving results and shutting down gracefully..." << endl;
    keep_running = false;
}

// Ultra-advanced hash function with prime-based mixing and multi-round processing
int hash_function(const string& text, unsigned int h_seed, unsigned int k_seed) {
    unsigned int h = h_seed;
    unsigned int k = k_seed;
    unsigned int len = text.length();
    
    // Length-dependent seed adjustment for better distribution
    h ^= len * 0x9e3779b9;  // Golden ratio
    k ^= len * 0x6a09e667;  // Square root of 2
    
    // Prime constants for superior mixing
    const unsigned int prime1 = 0x9e3779b1;  // Large prime near golden ratio
    const unsigned int prime2 = 0x85ebca77;  // Large prime
    const unsigned int prime3 = 0xc2b2ae3d;  // Large prime
    const unsigned int prime4 = 0x27d4eb2f;  // Large prime
    
    // Multi-round processing with different strategies per round
    for (size_t i = 0; i < len; i++) {
        unsigned int c = (unsigned int)text[i];
        
        // Round 1: Prime-based character mixing
        h ^= c * prime1;
        k ^= c * prime2;
        
        // Advanced bit rotations (using Fibonacci numbers for optimal distribution)
        h = (h << 13) | (h >> 19);  // 13-bit rotation (Fibonacci)
        k = (k >> 8) | (k << 24);   // 8-bit rotation (Fibonacci)
        
        // Cross-pollination between h and k
        h += k * prime3;
        k ^= h * prime4;
        
        // Round 2: Position-dependent mixing
        unsigned int pos_factor = (i + 1) * 0x9e3779b9;
        h ^= (c << (i % 16)) * pos_factor;
        k ^= (c >> (i % 16)) * pos_factor;
        
        // Additional avalanche per character
        h ^= h >> 16;
        k ^= k >> 16;
        h *= 0x85ebca6b;
        k *= 0xc2b2ae35;
        
        // Round 3: Length-position interaction
        if (i % 3 == 0) {
            h ^= (len * c) >> 11;
            k ^= (len * c) << 7;
        }
    }
    
    // Ultra-enhanced final avalanche (6-stage mixing)
    h ^= k;
    h ^= h >> 16;
    h *= prime1;
    h ^= h >> 13;
    h *= prime2;
    h ^= h >> 16;
    h *= prime3;
    h ^= h >> 15;
    h *= prime4;
    h ^= h >> 14;
    h *= 0x9e3779b9;
    h ^= h >> 13;
    
    return h % 100;
}

// Calculate standard deviation for a dataset with given seeds
double calculate_std_dev(const vector<string>& dataset, unsigned int h_seed, unsigned int k_seed) {
    vector<int> buckets(100, 0);
    
    for (const string& str : dataset) {
        int bucket = hash_function(str, h_seed, k_seed);
        buckets[bucket]++;
    }
    
    double mean = static_cast<double>(dataset.size()) / 100.0;
    double variance = 0.0;
    
    for (int count : buckets) {
        double diff = count - mean;
        variance += diff * diff;
    }
    variance /= 100.0;
    
    return sqrt(variance);
}

// Calculate average standard deviation across all datasets
double calculate_average_std_dev(const vector<vector<string>>& datasets, 
                                unsigned int h_seed, unsigned int k_seed) {
    double total_std_dev = 0.0;
    
    for (const auto& dataset : datasets) {
        total_std_dev += calculate_std_dev(dataset, h_seed, k_seed);
    }
    
    return total_std_dev / datasets.size();
}

// Load dataset from file
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

// Save best result to file
void save_best_result(double best_score, unsigned int best_h, unsigned int best_k, 
                     long long total_tests, int runtime_seconds, int thread_count) {
    ofstream file("cpu_infinite_best_result.txt");
    file << "=== INFINITE CPU OPTIMIZATION BEST RESULT ===" << endl;
    file << "Best h_seed: " << best_h << " (0x" << hex << best_h << dec << ")" << endl;
    file << "Best k_seed: " << best_k << " (0x" << hex << best_k << dec << ")" << endl;
    file << "Average Standard Deviation: " << fixed << setprecision(6) << best_score << endl;
    file << "Total tests completed: " << total_tests << endl;
    file << "Runtime: " << runtime_seconds << " seconds" << endl;
    file << "Threads used: " << thread_count << endl;
    file << "Tests per second: " << (total_tests / max(1, runtime_seconds)) << endl;
    file.close();
}

// Worker thread function
void worker_thread(int thread_id, const vector<vector<string>>& datasets, 
                  int tests_per_batch, chrono::high_resolution_clock::time_point start_time) {
    random_device rd;
    mt19937 gen(rd() + thread_id);  // Different seed per thread
    uniform_int_distribution<unsigned int> dis;
    
    int batch_count = 0;
    
    while (keep_running) {
        batch_count++;
        
        // Test a batch of random seeds
        for (int i = 0; i < tests_per_batch && keep_running; i++) {
            unsigned int h_seed = dis(gen);
            unsigned int k_seed = dis(gen);
            
            double score = calculate_average_std_dev(datasets, h_seed, k_seed);
            total_tests_completed++;
            
            // Check if this is a new best
            double current_best = global_best_score.load();
            if (score < current_best) {
                // Try to update the global best (atomic compare-and-swap)
                if (global_best_score.compare_exchange_weak(current_best, score)) {
                    global_best_h = h_seed;
                    global_best_k = k_seed;
                    
                    auto current_time = chrono::high_resolution_clock::now();
                    auto duration = chrono::duration_cast<chrono::seconds>(current_time - start_time);
                    
                    lock_guard<mutex> lock(output_mutex);
                    cout << "\n🎉 NEW BEST FOUND by Thread " << thread_id << "!" << endl;
                    cout << "Tests: " << total_tests_completed.load() << endl;
                    cout << "Score: " << fixed << setprecision(6) << score << endl;
                    cout << "h_seed: " << h_seed << " (0x" << hex << h_seed << dec << ")" << endl;
                    cout << "k_seed: " << k_seed << " (0x" << hex << k_seed << dec << ")" << endl;
                    cout << "Runtime: " << duration.count() << "s | Rate: " 
                         << (total_tests_completed.load() / max(1, (int)duration.count())) << " tests/sec" << endl;
                    
                    // Save immediately when we find a better result
                    save_best_result(score, h_seed, k_seed, total_tests_completed.load(), 
                                   duration.count(), thread::hardware_concurrency());
                }
            }
        }
        
        // Progress update every 100 batches per thread
        if (batch_count % 100 == 0) {
            auto current_time = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::seconds>(current_time - start_time);
            
            lock_guard<mutex> lock(output_mutex);
            cout << "Thread " << thread_id << " | Batch " << batch_count 
                 << " | Total tests: " << total_tests_completed.load() 
                 << " | Best: " << fixed << setprecision(6) << global_best_score.load()
                 << " | Rate: " << (total_tests_completed.load() / max(1, (int)duration.count())) << " tests/sec" << endl;
        }
    }
}

int main() {
    // Set up signal handler for graceful shutdown
    signal(SIGINT, signal_handler);
    
    cout << "=== INFINITE CPU HASH FUNCTION OPTIMIZER ===" << endl;
    cout << "Press Ctrl+C to stop and save the best result found" << endl;
    
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
    
    cout << "\nLoading datasets..." << endl;
    for (const string& name : dataset_names) {
        vector<string> dataset = load_dataset(name);
        if (dataset.empty()) {
            cout << "Warning: Could not load " << name << endl;
            continue;
        }
        datasets.push_back(dataset);
        cout << "Loaded " << dataset.size() << " entries from " << name << endl;
    }
    
    if (datasets.empty()) {
        cout << "Error: No datasets loaded!" << endl;
        return 1;
    }
    
    // Configuration
    int num_threads = thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;  // Fallback
    int tests_per_batch = 1000;  // Tests per batch per thread
    
    cout << "\nCPU Configuration:" << endl;
    cout << "- Number of threads: " << num_threads << endl;
    cout << "- Tests per batch per thread: " << tests_per_batch << endl;
    cout << "- Datasets: " << datasets.size() << endl;
    cout << "- Total estimated tests per second: " << (num_threads * tests_per_batch) << "+" << endl;
    
    auto start_time = chrono::high_resolution_clock::now();
    
    cout << "\n🚀 Starting infinite optimization with " << num_threads << " threads..." << endl;
    cout << "Current best: " << fixed << setprecision(6) << global_best_score.load() << endl;
    
    // Start worker threads
    vector<thread> threads;
    for (int i = 0; i < num_threads; i++) {
        threads.emplace_back(worker_thread, i, ref(datasets), tests_per_batch, start_time);
    }
    
    // Main monitoring loop
    while (keep_running) {
        this_thread::sleep_for(chrono::seconds(30));  // Update every 30 seconds
        
        if (!keep_running) break;
        
        auto current_time = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::seconds>(current_time - start_time);
        
        lock_guard<mutex> lock(output_mutex);
        cout << "\n📊 STATUS UPDATE:" << endl;
        cout << "Runtime: " << duration.count() << " seconds" << endl;
        cout << "Total tests: " << total_tests_completed.load() << endl;
        cout << "Current best score: " << fixed << setprecision(6) << global_best_score.load() << endl;
        cout << "Tests per second: " << (total_tests_completed.load() / max(1, (int)duration.count())) << endl;
        cout << "Active threads: " << num_threads << endl;
    }
    
    // Wait for all threads to finish
    for (auto& t : threads) {
        t.join();
    }
    
    // Final save and cleanup
    auto end_time = chrono::high_resolution_clock::now();
    auto total_duration = chrono::duration_cast<chrono::seconds>(end_time - start_time);
    
    cout << "\n=== INFINITE OPTIMIZATION STOPPED ===" << endl;
    cout << "Total runtime: " << total_duration.count() << " seconds" << endl;
    cout << "Total tests: " << total_tests_completed.load() << endl;
    cout << "Average tests per second: " << (total_tests_completed.load() / max(1, (int)total_duration.count())) << endl;
    
    cout << "\n🏆 FINAL BEST RESULT:" << endl;
    cout << "h_seed: " << global_best_h.load() << " (0x" << hex << global_best_h.load() << dec << ")" << endl;
    cout << "k_seed: " << global_best_k.load() << " (0x" << hex << global_best_k.load() << dec << ")" << endl;
    cout << "Average Standard Deviation: " << fixed << setprecision(6) << global_best_score.load() << endl;
    
    save_best_result(global_best_score.load(), global_best_h.load(), global_best_k.load(), 
                    total_tests_completed.load(), total_duration.count(), num_threads);
    
    cout << "\nResults saved to cpu_infinite_best_result.txt" << endl;
    
    return 0;
}