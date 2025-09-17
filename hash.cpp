#include <iostream>
#include <string>
#include "hash.h"

using namespace std;

int hash_function(const string &text) { 
    unsigned int h = 17;            // Small prime
    unsigned int k = 31;            // Another small prime 


    for (char c : text) { 
        h ^= (unsigned int)c * 0x85ebca6b; 
        k ^= (unsigned int)c * 0xc2b2ae35; 


        h = (h << 13) | (h >> 19); 
        k = (k >> 11) | (k << 21);   


        h += k; 
        k ^= h; 
    } 


  
    h ^= (h >> 16); 
    h *= 0x7feb352d; 
    h ^= (h >> 15); 
    h *= 0x846ca68b; 
    h ^= (h >> 16); 


    return (int)h; 
}