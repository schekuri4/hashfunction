/*
CSE 310 Hash Function DIY Contest
Instructor: Yiran "Lawrence" Luo
Your name(s):
Your team alias: 
*/

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cmath>
#include "hash.h"
using namespace std;

int main() {

    int k = 0;
    int n = 0;
    string texts[500];

    // WARNING: Start of the tokenizer that loads the input from std::cin, DO NOT change this part!
    cin >> k;
    string line;
    getline(cin, line);

    while (getline(cin, line)) {
        texts[n] = line;
        n++;
    }
    // WARNING: End of the tokenizer, DO NOT change this part!

    // By this point, k is the # of slots, and n is the # of tokens to fit in
    // texts[] stores the input sequence of tokens/keys to be inserted into your hash table

    // The template is able to be compiled by running 
    //   make
    //   ./encoder < inputs/sample_input.txt
    // which puts out the placeholders only.

    // Build hash table
    vector<vector<string>> table(k);
    for (int i = 0; i < n; i++) {
        int hash_val = hash_function(texts[i]);
        int idx = abs(hash_val) % k;
        table[idx].push_back(texts[i]);
    }

    // Print first 5 slots
    cout << "==== Printing the contents of the first 5 slots ====" << endl;
    for (int i = 0; i < 5 && i < k; i++) {
        cout << "Slot " << i << ": ";
        for (auto &s : table[i]) cout << s << " ";
        cout << endl;
    }

    // Print slot lengths
    cout << "==== Printing the slot lengths ====" << endl;
    for (int i = 0; i < k; i++) {
        cout << "Slot " << i << " length = " << table[i].size() << endl;
    }

    // Print standard deviation
    cout << "==== Printing the standard deviation =====" << endl;
    double mean = (double)n / k;
    double sum = 0;
    for (int i = 0; i < k; i++) {
        double diff = table[i].size() - mean;
        sum += diff * diff;
    }
    cout << sqrt(sum / k) << endl;

    return 0;
}