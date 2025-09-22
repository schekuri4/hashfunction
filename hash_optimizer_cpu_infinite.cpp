#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
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
string output_filename;
chrono::system_clock::time_point start_timestamp;

// Signal handler for Ctrl+C
void signal_handler(int signal) {
    cout << "\n\nReceived interrupt signal. Saving results and shutting down gracefully..." << endl;
    keep_running = false;
}

// New optimized hash function
unsigned int hash_function(const string& text, unsigned int h_seed, unsigned int k_seed) {
    unsigned int h = h_seed ^ 0x9e3779b9;
    unsigned int k = k_seed ^ 0x85ebca6b;
    unsigned int len = text.length();
    
    for (size_t i = 0; i < len; i++) {
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

// Calculate standard deviation for a dataset with given seeds and bucket count
double calculate_std_dev(const vector<string>& dataset, unsigned int h_seed, unsigned int k_seed, int table_size) {
    vector<int> buckets(table_size, 0);
    
    for (const string& str : dataset) {
        unsigned int hash_val = hash_function(str, h_seed, k_seed);
        int bucket = hash_val % table_size;
        buckets[bucket]++;
    }
    
    double mean = static_cast<double>(dataset.size()) / static_cast<double>(table_size);
    double variance = 0.0;
    
    for (int count : buckets) {
        variance += (count - mean) * (count - mean);
    }
    variance /= static_cast<double>(table_size);
    
    return sqrt(variance);
}

// Calculate average standard deviation across all datasets
double calculate_average_std_dev(const vector<vector<string>>& datasets, 
                                const vector<int>& table_sizes,
                                unsigned int h_seed, unsigned int k_seed) {
    double total_std_dev = 0.0;
    
    for (size_t i = 0; i < datasets.size(); i++) {
        total_std_dev += calculate_std_dev(datasets[i], h_seed, k_seed, table_sizes[i]);
    }
    
    return total_std_dev / datasets.size();
}

// Load dataset from file
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

// Save best result to file
string extract_hash_function() {
    ifstream source_file(__FILE__);
    string line;
    string hash_function = "";
    bool in_function = false;
    int brace_count = 0;
    
    while (getline(source_file, line)) {
        // Look for the start of the hash function
        if (line.find("int hash_function(const string& text") != string::npos) {
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
                     long long total_tests, int runtime_seconds, int thread_count) {
    // Get current time for this result
    auto now = chrono::system_clock::now();
    auto time_t = chrono::system_clock::to_time_t(now);
    
    // Append to the single output file
    ofstream file(output_filename, ios::app);
    file << "[" << put_time(localtime(&time_t), "%Y-%m-%d %H:%M:%S") << "] "
         << "CPU NEW BEST (T" << thread_count << "): Score=" << fixed << setprecision(6) << best_score 
         << " | Hash: h=" << best_h << "(0x" << hex << best_h << dec 
         << "), k=" << best_k << "(0x" << hex << best_k << dec 
         << ") | Tests=" << total_tests << " | Runtime=" << runtime_seconds 
         << "s | Rate=" << (total_tests / max(1, runtime_seconds)) << "/s" << endl;
    file.close();
}

// Worker thread function
void worker_thread(int thread_id, const vector<vector<string>>& datasets, 
                  const vector<int>& table_sizes,
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
            
            double score = calculate_average_std_dev(datasets, table_sizes, h_seed, k_seed);
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
                    
                    // Get current timestamp
                    auto now = chrono::system_clock::now();
                    auto time_t = chrono::system_clock::to_time_t(now);
                    
                    lock_guard<mutex> lock(output_mutex);
                    cout << "\n🎉 [" << put_time(localtime(&time_t), "%Y-%m-%d %H:%M:%S") 
                         << "] NEW BEST (T" << thread_id << "): Score=" << fixed << setprecision(6) << score 
                         << " | Hash: h=" << h_seed << "(0x" << hex << h_seed << dec 
                         << "), k=" << k_seed << "(0x" << hex << k_seed << dec 
                         << ") | Function: h=h_seed^0x9e3779b9, k=k_seed^0x85ebca6b | Tests=" << total_tests_completed.load() 
                         << " | Runtime=" << duration.count() << "s | Rate=" 
                         << (total_tests_completed.load() / max(1, (int)duration.count())) << "/s" << endl;
                    
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
    vector<int> table_sizes;
    
    cout << "\nLoading datasets..." << endl;
    for (const string& name : dataset_names) {
        auto [dataset, table_size] = load_dataset(name);
        if (dataset.empty()) {
            cout << "Warning: Could not load " << name << endl;
            continue;
        }
        datasets.push_back(dataset);
        table_sizes.push_back(table_size);
        cout << "Loaded " << dataset.size() << " entries from " << name << " (table size: " << table_size << ")" << endl;
    }
    
    if (datasets.empty()) {
        cout << "Error: No datasets loaded!" << endl;
        return 1;
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
        threads.emplace_back(worker_thread, i, ref(datasets), ref(table_sizes), tests_per_batch, start_time);
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
    
    return 0;
}