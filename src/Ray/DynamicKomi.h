#ifndef _DYNAMICKOMI_H_
#define _DYNAMICKOMI_H_

#include "GoBoard.h"
#include "UctSearch.h"

const double RED = 0.35;

const double GREEN = 0.75;

const int LINEAR_THRESHOLD = 200;

const int HANDICAP_WEIGHT = 8;

enum DYNAMIC_KOMI_MODE {
  DK_OFF,
  DK_LINEAR,
  DK_VALUE,
};

void SetHandicapNum(const int num);

void SetConstHandicapNum(const int num);

void DynamicKomi(const game_info_t *game, const uct_node_t *root,
                 const int color);

#endif
