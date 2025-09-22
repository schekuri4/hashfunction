#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <iomanip>
#include "hash.h"

using namespace std;

struct TestResult {
    string filename;
    int table_size;
    int num_strings;
    double mean;
    double std_deviation;
    int min_count;
    int max_count;
    vector<int> bucket_counts;
};

TestResult analyze_hash_distribution(const string& filename) {
    TestResult result;
    result.filename = filename;
    
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        return result;
    }
    
    // Read table size from first line
    file >> result.table_size;
    file.ignore(); // ignore newline after table size
    
    // Initialize bucket counts
    result.bucket_counts.resize(result.table_size, 0);
    
    string line;
    result.num_strings = 0;
    
    // Read each string and hash it
    while (getline(file, line)) {
        if (!line.empty()) {
            unsigned int hash_value = hash_function(line);
            int bucket = hash_value % result.table_size;
            result.bucket_counts[bucket]++;
            result.num_strings++;
        }
    }
    
    file.close();
    
    // Calculate statistics
    result.mean = (double)result.num_strings / result.table_size;
    
    // Find min and max
    result.min_count = result.bucket_counts[0];
    result.max_count = result.bucket_counts[0];
    for (int count : result.bucket_counts) {
        if (count < result.min_count) result.min_count = count;
        if (count > result.max_count) result.max_count = count;
    }
    
    // Calculate standard deviation
    double sum_squared_diff = 0.0;
    for (int count : result.bucket_counts) {
        double diff = count - result.mean;
        sum_squared_diff += diff * diff;
    }
    result.std_deviation = sqrt(sum_squared_diff / result.table_size);
    
    return result;
}

void print_detailed_results(const TestResult& result) {
    cout << "\n" << string(60, '=') << endl;
    cout << "HASH DISTRIBUTION ANALYSIS: " << result.filename << endl;
    cout << string(60, '=') << endl;
    
    cout << "Table Size: " << result.table_size << endl;
    cout << "Number of Strings: " << result.num_strings << endl;
    cout << "Expected Items per Bucket (Mean): " << fixed << setprecision(4) << result.mean << endl;
    cout << "Standard Deviation: " << fixed << setprecision(4) << result.std_deviation << endl;
    cout << "Min Bucket Count: " << result.min_count << endl;
    cout << "Max Bucket Count: " << result.max_count << endl;
    cout << "Load Factor: " << fixed << setprecision(4) << (double)result.num_strings / result.table_size << endl;
    
    // Calculate and display distribution quality metrics
    double coefficient_of_variation = result.std_deviation / result.mean;
    cout << "Coefficient of Variation: " << fixed << setprecision(4) << coefficient_of_variation << endl;
    
    // Show bucket distribution histogram
    cout << "\nBucket Distribution:" << endl;
    cout << "Bucket | Count | Visualization" << endl;
    cout << string(40, '-') << endl;
    
    for (int i = 0; i < result.table_size; i++) {
        cout << setw(6) << i << " | " << setw(5) << result.bucket_counts[i] << " | ";
        
        // Simple visualization with asterisks
        int bar_length = (result.bucket_counts[i] * 20) / (result.max_count > 0 ? result.max_count : 1);
        for (int j = 0; j < bar_length; j++) {
            cout << "*";
        }
        cout << endl;
    }
    
    // Calculate chi-square goodness of fit test
    double chi_square = 0.0;
    for (int count : result.bucket_counts) {
        double expected = result.mean;
        if (expected > 0) {
            chi_square += ((count - expected) * (count - expected)) / expected;
        }
    }
    cout << "\nChi-Square Statistic: " << fixed << setprecision(4) << chi_square << endl;
    cout << "Degrees of Freedom: " << (result.table_size - 1) << endl;
}

void print_summary_table(const vector<TestResult>& results) {
    cout << "\n" << string(80, '=') << endl;
    cout << "SUMMARY TABLE - HASH FUNCTION PERFORMANCE" << endl;
    cout << string(80, '=') << endl;
    
    cout << left << setw(20) << "File" 
         << setw(12) << "Table Size" 
         << setw(12) << "Strings" 
         << setw(12) << "Mean" 
         << setw(15) << "Std Deviation" 
         << setw(10) << "Min/Max" << endl;
    cout << string(80, '-') << endl;
    
    for (const auto& result : results) {
        cout << left << setw(20) << result.filename.substr(result.filename.find_last_of("/\\") + 1)
             << setw(12) << result.table_size
             << setw(12) << result.num_strings
             << setw(12) << fixed << setprecision(2) << result.mean
             << setw(15) << fixed << setprecision(4) << result.std_deviation
             << setw(10) << (to_string(result.min_count) + "/" + to_string(result.max_count)) << endl;
    }
    
    cout << "\nInterpretation:" << endl;
    cout << "- Lower standard deviation indicates better hash distribution" << endl;
    cout << "- Ideal standard deviation for uniform distribution ≈ sqrt(mean)" << endl;
    cout << "- Coefficient of variation < 0.5 is generally considered good" << endl;
}

int main() {
    vector<string> test_files = {
        "inputs/test_alls.txt",
        "inputs/test_passwords.txt", 
        "inputs/test_wordle500.txt"
    };
    
    vector<TestResult> results;
    
    cout << "HASH FUNCTION DISTRIBUTION ANALYSIS" << endl;
    cout << "Testing hash function with " << test_files.size() << " input files..." << endl;
    
    for (const string& filename : test_files) {
        cout << "\nProcessing: " << filename << "..." << endl;
        TestResult result = analyze_hash_distribution(filename);
        
        if (result.num_strings > 0) {
            results.push_back(result);
            print_detailed_results(result);
        } else {
            cout << "Error: No data processed for " << filename << endl;
        }
    }
    
    if (!results.empty()) {
        print_summary_table(results);
    }
    
    return 0;
}