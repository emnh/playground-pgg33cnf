# Fast 6x12 Connected Components using bit-optimized BFS

This is pretty standard BFS, just bit-optimized and with a lookup table for neighbours.
Performance varies a lot depending on how densely populated the bitboards are,
and that in turn depends a lot on the random seed they are initialized with.
But you can expect 2-5 millions of iterations per 100ms, suitable for Smash the Code.
Enough talk, here's to the code:

```C++ runnable
#pragma GCC optimize("Ofast","unroll-loops","omit-frame-pointer","inline")
#pragma GCC option("arch=native","tune=native","no-zeroupper") //Enable AVX
// #pragma GCC target("avx2")
// #pragma GCC target("popcnt") //Enable popcount
// #pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,avx2") //Enable AVX
#include <x86intrin.h> //SSE Extensions
#include <bits/stdc++.h> //All main STD libraries
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>
#include <bitset>

#define USE_AVX 1

typedef __uint128_t BOARD_T;

using namespace std;

using time_interval_t = std::chrono::microseconds;
using myClock = std::chrono::high_resolution_clock;

const unsigned int rowCount = 12;
const unsigned int colCount = 8;
const unsigned int realColCount = colCount;
const unsigned int twoRowsCellCount = 2 * colCount;
const unsigned int colorCount = 6;
const unsigned int cellCount = rowCount * colCount;
const unsigned int backgroundCount = 1;
const unsigned int maxComponents = backgroundCount + cellCount / 2;
const unsigned int maxComponentsPerRow = 3;
const unsigned int lookupTableSizeFor1Row = 1 << (1 * realColCount);
const unsigned int lookupTableSizeFor2Rows = 1 << (2 * realColCount);

union converter8
{
  uint8_t num;
  struct
  {
      uint8_t num8[1];
  } bytes;
};

inline converter8 initializeConverter(uint8_t num) {
  converter8 converter;
  converter.num = num;
  return converter;
}

union converter16
{
  uint16_t num;
  struct
  {
      uint8_t num8[2];
  } bytes;
};

inline converter16 initializeConverter(uint16_t num) {
  converter16 converter;
  converter.num = num;
  return converter;
}

union converter32
{
    uint32_t num;
    struct
    {
        uint8_t num8[4];
    } bytes;
};

inline converter32 initializeConverter(uint32_t num) {
  converter32 converter;
  converter.num = num;
  return converter;
}

union converter64
{
    uint64_t num;
    struct
    {
        uint8_t num8[8];
    } bytes;
};

inline converter64 initializeConverter(uint64_t num) {
  converter64 converter;
  converter.num = num;
  return converter;
}

union converter128
{
  BOARD_T num;
  struct
  {
      uint8_t num8[16];
  } bytes;
  struct
  {
      uint64_t num64[2];
  } halves;
};

inline converter128 initializeConverter(__uint128_t num) {
  converter128 converter;
  converter.num = num;
  return converter;
}

inline int clz_u128 (__uint128_t u) {
  uint64_t hi = u>>64;
  uint64_t lo = u;
  int retval[3]={
    __builtin_clzll(hi),
    __builtin_clzll(lo)+64,
    128
  };
  int idx = !hi + ((!lo)&(!hi));
  return 127 - retval[idx];
}

inline int my_clz(__uint128_t u) {
  return clz_u128(u);
}

inline int my_clz(uint64_t u) {
  return __builtin_clzll(u);
}

inline int my_clz(uint32_t u) {
  return __builtin_clz(u);
}

inline int ctz_u128 (__uint128_t u) {
  uint64_t hi = u>>64;
  uint64_t lo = u;
  int retval[3]={
    __builtin_ctzll(lo),
    __builtin_ctzll(hi)+64,
    128
  };
  int idx = !lo + ((!lo)&(!hi));
  return retval[idx];
}

inline int my_ctz(__uint128_t u) {
  return ctz_u128(u);
}

inline int my_ctz(uint64_t u) {
  return __builtin_ctzll(u);
}

inline int my_ctz(uint32_t u) {
  return __builtin_ctz(u);
}

#ifdef USE_AVX
#pragma GCC push_options
#pragma GCC target("avx2")
#endif
inline uint8_t mypopcount (const __uint128_t& data) {
  const uint64_t hi = data >> 64;
  const uint64_t lo = data;
  return __builtin_popcountll(lo) + __builtin_popcountll(hi);
}

inline uint8_t mypopcount (const uint64_t& data) {
  return __builtin_popcountll(data);
}

inline uint8_t mypopcount (const uint32_t& data) {
  return __builtin_popcount(data);
}

inline __uint128_t myBoundsLimit(__uint128_t board) {
  constexpr __uint128_t first2ColClear =
     0x3F3F3F3F3F3F3F3F |
     ((__uint128_t) 0x3F3F3F3F3F3F3F3F << 64);
  return
    board &
    (((__uint128_t) 1 << cellCount) - (__uint128_t) 1) &
    first2ColClear;
}

inline uint64_t myBoundsLimit(uint64_t board) {
  return board;
}

inline uint32_t myBoundsLimit(uint32_t board) {
  return board;
}

inline __uint128_t myRemoveOnes(const __uint128_t& board) {
  const __uint128_t inrows { myBoundsLimit(board) };
  constexpr __uint128_t first3ColClear { 0x1F1F1F1F1F1F1F1F | ((__uint128_t) 0x1F1F1F1F1F1F1F1F << 64) };
  constexpr __uint128_t lastColClear { 0xFEFEFEFEFEFEFEFE | ((__uint128_t) 0xFEFEFEFEFEFEFEFE << 64) };
  const __uint128_t nbs = ((inrows >> 1) & first3ColClear) | ((inrows << 1) & lastColClear) | (inrows << colCount) | (inrows >> colCount);
  const __uint128_t rows = inrows & nbs;

  return rows;
}

inline uint64_t myRemoveOnes(const uint64_t& board) {
  const uint64_t inrows { myBoundsLimit(board) };
  const uint64_t first3ColClear { 0x1F1F1F1F1F1F1F1F };
  const uint64_t lastColClear { 0xFEFEFEFEFEFEFEFE };
  const uint64_t nbs = ((inrows >> 1) & first3ColClear) | ((inrows << 1) & lastColClear) | (inrows << colCount) | (inrows >> colCount);
  const uint64_t rows = inrows & nbs;
  return rows;
}

inline uint32_t myRemoveOnes(const uint32_t& board) {
  const uint32_t inrows { myBoundsLimit(board) };
  const uint32_t first3ColClear { 0x1F1F1F1F };
  const uint32_t lastColClear { 0xFEFEFEFE };
  const uint32_t nbs = ((inrows >> 1) & first3ColClear) | ((inrows << 1) & lastColClear) | (inrows << colCount) | (inrows >> colCount);
  const uint32_t rows = inrows & nbs;
  return rows;
}

#ifdef USE_AVX
#pragma GCC pop_options
#endif

template <class T, class U>
class mybitsetT {
public:
  alignas(32) T data;

  inline mybitsetT<T, U>() {
    data.num = 0;
  }

  inline mybitsetT<T, U>(const __uint128_t data) {
    this->data.num = data;
  }

  inline mybitsetT<T, U>(const T data) {
    this->data.num = data.num;
  }

  inline const mybitsetT<T, U> boundsLimit() const {
    return myBoundsLimit(data.num);
  }

  inline const mybitsetT<T, U> operator~ () const noexcept {
    return mybitsetT<T, U>(~data.num);
  }

  inline const mybitsetT<T, U> operator& (const mybitsetT<T, U>& rhs) const noexcept {
    return mybitsetT<T, U>(data.num & rhs.data.num);
  }

  inline mybitsetT<T, U>& operator&= (const mybitsetT<T, U>& rhs) noexcept {
    data.num &= rhs.data.num;
    return *this;
  }

  inline const mybitsetT<T, U> operator| (const mybitsetT<T, U>& rhs) const noexcept {
    return mybitsetT<T, U>(data.num | rhs.data.num);
  }

  inline mybitsetT<T, U>& operator|= (const mybitsetT<T, U>& rhs) noexcept {
    data.num |= rhs.data.num;
    return *this;
  }

  inline const mybitsetT<T, U> operator<< (size_t pos) const noexcept {
    return mybitsetT<T, U>(data.num << pos);
  }

  inline mybitsetT<T, U>& operator<<= (size_t pos) noexcept {
    data.num <<= pos;
    return *this;
  }

  inline const mybitsetT<T, U> operator>> (size_t pos) const noexcept {
    return mybitsetT<T, U>(data.num >> pos);
  }

  inline mybitsetT<T, U>& operator>>= (size_t pos) noexcept {
    data.num >>= pos;
    return *this;
  }

  inline const bool operator== (const mybitsetT<T, U>& rhs) const noexcept {
    return data.num == rhs.data.num;
  }

  inline const uint64_t to_ullong() const {
    return (uint64_t) data.num;
  }

  inline void clear(const int i) {
    data.num &= ~((U) 1 << i);
  }

  inline void clearFirstSetBit() {
    data.num &= (data.num - (U) 1);
  }

  inline void set(const int i) {
    data.num |= ((U) 1 << i);
  }

  inline void set(const int i, const bool value) {
    data.num ^= (-value ^ data.num) & ((U) 1 << i);
  }

  inline bool test(const int i) const {
    return (data.num >> i) & (U) 1;
  }

  inline bool any() const {
    return data.num != 0;
  }

  inline const int getFirstSetBit() const {
    return my_ctz(data.num);
  }

  inline const int getLastSetBit() const {
    return my_clz(data.num);
  }

  inline const int popcount() const {
    return mypopcount(data.num);
  }

  inline const mybitsetT<T, U> RemoveOnes() const {
    return mybitsetT<T, U>(myRemoveOnes(data.num));
  }
};

typedef mybitsetT<converter128, BOARD_T> mybitset;
typedef mybitsetT<converter64, uint64_t> mybitset64;
typedef mybitsetT<converter32, uint32_t> mybitset32;
typedef mybitsetT<converter32, uint64_t> threeRowsBitset;
typedef mybitsetT<converter16, uint16_t> twoRowsBitset;
typedef mybitsetT<converter8, uint8_t> oneRowBitset;

class Bitboard {
public:
  // Most Significant Bit is the last bit, map lower right, index cellCount - 1, rightmost bit.
  // Least Significant Bit is the first bit, map upper left, index 0, leftmost bit.
  // To shift one row down, shift right: << colCount.
  // To shift one row up, shift left: >> colCount.

  mybitset board;

  void initRandom() {
    for (int i = 0; i < cellCount; i++) {
      bool val = rand() % 2 == 0;
      board.set(i, val);
    }
  }

  int getIndex(int x, int y) const {
    return x + y * colCount;
  }

  bool get(int x, int y) const {
    return board.test(getIndex(x, y));
  }

  bool test(int x, int y) const {
    return board.test(getIndex(x, y));
  }

  void set(int x, int y, bool value) {
    board.set(getIndex(x, y), value);
  }

  inline void print() const {
    cerr << "BOARD: " << board.popcount() << endl;
    for (int y = 0; y < rowCount; y++) {
      for (int x = 0; x < colCount; x++) {
        cerr << (get(x, y) ? 1 : 0) << " ";
      }
      cerr << endl;
    }
    cerr << endl;
  }
};

class BitboardComponents {
public:
  uint8_t componentCount = 0;
  mybitset workComponents[maxComponents];

  const static int nbCount = 4;
  const static int maxQueue = nbCount * cellCount;
  mybitset queue2[maxQueue];
  mybitset nbLookup[rowCount * colCount];

  void initFindComponents2() {
    for (int i = 0; i < rowCount * colCount; i++)
    {
      int topBitIndex = i;
      int nbs[nbCount];
      nbs[0] = (((topBitIndex + 1) % colCount) != 0) ? topBitIndex + 1 : -1;
      nbs[1] = ((topBitIndex % colCount) != 0) ? topBitIndex - 1 : -1;
      nbs[2] = topBitIndex + colCount;
      nbs[3] = topBitIndex - colCount;

      mybitset nbset = 0;
      for (int nbi = 0; nbi < nbCount; nbi++) {
        int nb = nbs[nbi];
        if (0 <= nb && nb < cellCount) {
          nbset.set(nb);
        }
      }
      nbLookup[i] = nbset;
    }
  }

  mybitset findComponents2(
    mybitset& remaining,
    int startIndex
    ) {
    mybitset ret = 0;

    uint8_t qIndex = 0;
    uint8_t qlen = 1;
    ret.set(startIndex);
    remaining.clear(startIndex);
    queue2[qIndex] = nbLookup[startIndex] & remaining;
    
    while (qlen > 0) {
      mybitset topBitIndices = queue2[qIndex];
      qIndex++;
      qlen--;

      remaining &= ~topBitIndices;
      ret |= topBitIndices;

      for (uint8_t nbi = 0; nbi < nbCount; nbi++) {
        if (!topBitIndices.any()) {
          break;
        }
        int topBitIndex = topBitIndices.getFirstSetBit();
        topBitIndices.clearFirstSetBit();

        mybitset nbs = nbLookup[topBitIndex] & remaining;
        if (nbs.any()) {
          // TODO(?): No overflow check.
          queue2[qIndex + qlen] = nbs;
          qlen++;
        }
      }
    }

    return ret;
  }

  void GetComponentsBFS(const Bitboard& board) {
    mybitset firstRemaining = board.board.RemoveOnes();
    
    componentCount = 0;

    while (firstRemaining.popcount() >= 4) {
      int startIndex = firstRemaining.getFirstSetBit();
      mybitset oldComponent =
        findComponents2(firstRemaining, startIndex).boundsLimit();
      if (oldComponent.any() && oldComponent.popcount() >= 4) {
        workComponents[componentCount] = oldComponent;
        componentCount++;
      }
    }
  }

  mybitset GetComponent(int index) {
    return mybitset { workComponents[index].data.num };
  }

  mybitset GetLastComponent() {
    return mybitset { workComponents[componentCount - 1].data.num };
  }
};

uint64_t x { 13124121 }; /* The state can be seeded with any value. */
uint64_t next()
{
    uint64_t z = (x += UINT64_C(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}

class BitboardColors {
public:
  Bitboard boards[colorCount];
  bool valid = false;

  void initRandom(int prob) {
    BOARD_T r1 = (next()) | ((((BOARD_T) next()) << twoRowsCellCount) & ~(((BOARD_T) (1) << 64) - 1));
    BOARD_T r2 = (next()) | ((((BOARD_T) next()) << twoRowsCellCount) & ~(((BOARD_T) (1) << 64) - 1));
    BOARD_T r3 = (next()) | ((((BOARD_T) next()) << twoRowsCellCount) & ~(((BOARD_T) (1) << 64) - 1));
    BOARD_T r = r1 & r2 & r3 & (((BOARD_T) (1) << cellCount) - 1);
    boards[1].board = r;

    for (int x = 0; x < 6; x++) {
      for (int y = 0; y < rowCount; y++) {
        // boards[1].board.set(x + y * colCount, rand() % 100 < prob);
      }
    }

    // Make sure we don't exceed columns or rows
    const mybitset first2ColClear { 0x3F3F3F3F3F3F3F3F | ((BOARD_T) 0x3F3F3F3F3F3F3F3F << 64) };
    const mybitset clearHigh { 0xFFFFFFFFFFFFFFFF | ((BOARD_T) 0xFFFFFFFF << 64) };
    boards[1].board &= first2ColClear & clearHigh;
  }
};

int main(int argc, char** argv) {
  const int boardCount = 10;
  BitboardColors boards[boardCount];
  for (int i = 0; i < boardCount; i++) {
    boards[i].initRandom(i * 10);
  }

  const uint64_t maxi = 10000000;

  // game loop
//   if (argc <= 1) {
//     for (int i = 0; i < 8; i++) {
//         int colorA; // color of the first block
//         int colorB; // color of the attached block
//         cin >> colorA >> colorB; cin.ignore();
//     }
//     int score1;
//     cin >> score1; cin.ignore();
//     for (int i = 0; i < 12; i++) {
//         string row; // One line of the map ('.' = empty, '0' = skull block, '1' to '5' = colored block)
//         cin >> row; cin.ignore();
//     }
//     int score2;
//     cin >> score2; cin.ignore();
//     for (int i = 0; i < 12; i++) {
//         string row;
//         cin >> row; cin.ignore();
//     }
//   }
  
  int i = 0;
  BitboardComponents bc;
  bc.initFindComponents2();

  uint64_t bits = 0;
  auto start = myClock::now();

  {
    // Warmup
    for (int i = 0; i < maxi; i++) {
      const int r = next() % boardCount;
      // boards[r].initRandom();
      bc.GetComponentsBFS(boards[r].boards[1]);
      bits ^= bc.GetLastComponent().to_ullong();
    }

    start = myClock::now();
    for (int i = 0; i < maxi; i++) {
      const int r = next() % boardCount;
      // boards[r].initRandom();
      bc.GetComponentsBFS(boards[r].boards[1]);
      bits ^= bc.GetLastComponent().to_ullong();
    }
    const auto elapsed3 =
      std::chrono::duration_cast<time_interval_t>(myClock::now() - start);
    const auto iters100msNormal = 100000 * maxi / elapsed3.count();
    cerr << "normal elapsed: " << elapsed3.count() << ", iterations: " << maxi << ", iterations/100ms: " << iters100msNormal << endl;
  }

  cerr << bits << endl;

  // "x rotation": the column in which to drop your pair of blocks followed by its rotation (0, 1, 2 or 3)
  cout << "0 1" << endl;
  
  return 0;
}
```