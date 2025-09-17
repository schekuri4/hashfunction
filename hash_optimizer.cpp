#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <random>
#include <fstream>
#include <string>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <atomic>

using namespace std;

// Hash function implementation (same as in hash.cpp)
int hash_function_test(const string &text, unsigned int h_seed, unsigned int k_seed) {
    unsigned int h = h_seed;
    unsigned int k = k_seed;
    
    for (char c : text) {
        h ^= static_cast<unsigned int>(c);
        h *= k;
        h ^= h >> 16;
    }
    
    // Final hash processing
    h *= 0x846ca68b;
    h ^= h >> 16;
    
    return h % 100;  // Same table size as original
}

// Calculate standard deviation for a dataset
double calculate_std_dev(const vector<string>& dataset, unsigned int h_seed, unsigned int k_seed) {
    vector<int> buckets(100, 0);
    
    // Hash all strings and count bucket usage
    for (const string& text : dataset) {
        int bucket = hash_function_test(text, h_seed, k_seed);
        buckets[bucket]++;
    }
    
    // Calculate mean
    double mean = static_cast<double>(dataset.size()) / 100.0;
    
    // Calculate standard deviation
    double variance = 0.0;
    for (int count : buckets) {
        variance += (count - mean) * (count - mean);
    }
    variance /= 100.0;
    
    return sqrt(variance);
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

// Structure to hold test results
struct TestResult {
    unsigned int h_seed;
    unsigned int k_seed;
    double avg_std_dev;
    vector<double> individual_std_devs;
};

// Global variables for thread coordination
mutex result_mutex;
vector<TestResult> all_results;
atomic<int> tests_completed(0);
atomic<bool> should_stop(false);

// Worker thread function
void worker_thread(int thread_id, int num_tests, const vector<vector<string>>& datasets, 
                  const vector<string>& dataset_names) {
    
    random_device rd;
    mt19937 gen(rd() + thread_id);  // Different seed per thread
    uniform_int_distribution<unsigned int> dis(0, UINT32_MAX);
    
    TestResult best_local;
    best_local.avg_std_dev = 1000.0;  // Initialize to high value
    
    for (int test = 0; test < num_tests && !should_stop; test++) {
        // Generate random seeds
        unsigned int h_seed = dis(gen);
        unsigned int k_seed = dis(gen);
        
        // Test on all datasets
        vector<double> std_devs;
        double total_std_dev = 0.0;
        
        for (size_t i = 0; i < datasets.size(); i++) {
            double std_dev = calculate_std_dev(datasets[i], h_seed, k_seed);
            std_devs.push_back(std_dev);
            total_std_dev += std_dev;
        }
        
        double avg_std_dev = total_std_dev / datasets.size();
        
        // Check if this is the best result for this thread
        if (avg_std_dev < best_local.avg_std_dev) {
            best_local.h_seed = h_seed;
            best_local.k_seed = k_seed;
            best_local.avg_std_dev = avg_std_dev;
            best_local.individual_std_devs = std_devs;
        }
        
        tests_completed++;
        
        // Print progress every 1000 tests
        if (test % 1000 == 0) {
            cout << "Thread " << thread_id << ": " << test << "/" << num_tests 
                 << " tests completed. Best so far: " << best_local.avg_std_dev << endl;
        }
    }
    
    // Add best result to global results
    lock_guard<mutex> lock(result_mutex);
    all_results.push_back(best_local);
    
    cout << "Thread " << thread_id << " completed. Best result: " 
         << best_local.avg_std_dev << " (h=" << best_local.h_seed 
         << ", k=" << best_local.k_seed << ")" << endl;
}

int main() {
    cout << "=== Hash Function Seed Optimizer ===" << endl;
    cout << "Using CPU multi-threading for parallel optimization" << endl;
    
    // Load all datasets
    vector<string> dataset_names = {
        "inputs/sample_input.txt",
        "inputs/atoz.txt", 
        "inputs/common500.txt",
        "inputs/bertuncased.txt",
        "inputs/mit_a.txt"
    };
    
    vector<vector<string>> datasets;
    for (const string& name : dataset_names) {
        cout << "Loading " << name << "..." << endl;
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
    
    int total_tests;
    cout << "Enter total number of tests to run (recommended: 100000+): ";
    cin >> total_tests;
    
    int tests_per_thread = total_tests / num_threads;
    
    cout << "\nStarting optimization with:" << endl;
    cout << "- Threads: " << num_threads << endl;
    cout << "- Total tests: " << total_tests << endl;
    cout << "- Tests per thread: " << tests_per_thread << endl;
    cout << "- Datasets: " << datasets.size() << endl;
    
    // Start timing
    auto start_time = chrono::high_resolution_clock::now();
    
    // Launch worker threads
    vector<thread> threads;
    for (int i = 0; i < num_threads; i++) {
        threads.emplace_back(worker_thread, i, tests_per_thread, 
                           ref(datasets), ref(dataset_names));
    }
    
    // Progress monitoring thread
    thread progress_thread([&]() {
        while (!should_stop && tests_completed < total_tests) {
            this_thread::sleep_for(chrono::seconds(5));
            cout << "Progress: " << tests_completed << "/" << total_tests 
                 << " (" << (100.0 * tests_completed / total_tests) << "%)" << endl;
        }
    });
    
    // Wait for all threads to complete
    for (auto& t : threads) {
        t.join();
    }
    
    should_stop = true;
    progress_thread.join();
    
    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::seconds>(end_time - start_time);
    
    // Find the best overall result
    TestResult best_overall;
    best_overall.avg_std_dev = 1000.0;
    
    for (const auto& result : all_results) {
        if (result.avg_std_dev < best_overall.avg_std_dev) {
            best_overall = result;
        }
    }
    
    // Display results
    cout << "\n=== OPTIMIZATION COMPLETE ===" << endl;
    cout << "Time taken: " << duration.count() << " seconds" << endl;
    cout << "Tests completed: " << tests_completed << endl;
    cout << "Tests per second: " << (tests_completed.load() / max(1, (int)duration.count())) << endl;
    
    cout << "\n=== BEST RESULT ===" << endl;
    cout << "h_seed: " << best_overall.h_seed << " (0x" << hex << best_overall.h_seed << dec << ")" << endl;
    cout << "k_seed: " << best_overall.k_seed << " (0x" << hex << best_overall.k_seed << dec << ")" << endl;
    cout << "Average Standard Deviation: " << best_overall.avg_std_dev << endl;
    
    cout << "\nIndividual dataset results:" << endl;
    for (size_t i = 0; i < dataset_names.size() && i < best_overall.individual_std_devs.size(); i++) {
        cout << "  " << dataset_names[i] << ": " << best_overall.individual_std_devs[i] << endl;
    }
    
    // Save results to file
    ofstream results_file("optimization_results.txt");
    results_file << "Best h_seed: " << best_overall.h_seed << endl;
    results_file << "Best k_seed: " << best_overall.k_seed << endl;
    results_file << "Average Standard Deviation: " << best_overall.avg_std_dev << endl;
    results_file << "Tests completed: " << tests_completed << endl;
    results_file << "Time taken: " << duration.count() << " seconds" << endl;
    
    cout << "\nResults saved to optimization_results.txt" << endl;
    
    return 0;
}