#ifndef _ZOBRISTHASH_H_
#define _ZOBRISTHASH_H_

#include <vector>

#include "GoBoard.h"

enum hash {
  HASH_PASS,
  HASH_BLACK,
  HASH_WHITE,
  HASH_KO,
};

const unsigned int UCT_HASH_SIZE = 16384;

struct node_hash_t {
  unsigned long long hash;
  int color;
  int moves;
  bool flag;
};

extern unsigned long long move_bit[MAX_RECORDS][BOARD_MAX][HASH_KO + 1];

extern unsigned long long hash_bit[BOARD_MAX][HASH_KO + 1];

extern unsigned long long shape_bit[BOARD_MAX];

extern node_hash_t *node_hash;

extern unsigned int uct_hash_size;

void SetHashSize(const unsigned int new_size);

void InitializeHash(void);

void InitializeUctHash(void);

void ClearUctHash(void);

void DeleteOldHash(const game_info_t *game);

unsigned int SearchEmptyIndex(const unsigned long long hash, const int color,
                              const int moves);

unsigned int FindSameHashIndex(const unsigned long long hash, const int color,
                               const int moves);

bool CheckRemainingHashSize(void);

void ClearNotDescendentNodes(std::vector<int> &indexes);

#endif
