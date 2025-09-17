#include <iostream>
#include <string>
#include "hash.h"

using namespace std;

unsigned int hash_function(const string& text) {
    unsigned int h = 0xa74e5167 ^ 0x9e3779b9;
    unsigned int k = 0x616740fd ^ 0x85ebca6b;
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