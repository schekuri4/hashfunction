#include <iostream>
#include <string>
#include "hash.h"

using namespace std;

int hash_function(const string& text, unsigned int h_seed, unsigned int k_seed) {
    unsigned int h = h_seed;
    unsigned int k = k_seed;
    unsigned int len = text.length();
    
    
    h ^= len * 0x9e3779b9;  // Golden ratio
    k ^= len * 0x6a09e667; 

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
    
    return h 
}