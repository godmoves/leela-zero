#ifndef _UCTRATING_H_
#define _UCTRATING_H_

#include <string>

#include "GoBoard.h"
#include "PatternHash.h"

enum UCT_FEATURE1 {
  UCT_SAVE_CAPTURE_1_1,
  UCT_SAVE_CAPTURE_1_2,
  UCT_SAVE_CAPTURE_1_3,
  UCT_SAVE_CAPTURE_2_1,
  UCT_SAVE_CAPTURE_2_2,
  UCT_SAVE_CAPTURE_2_3,
  UCT_SAVE_CAPTURE_3_1,
  UCT_SAVE_CAPTURE_3_2,
  UCT_SAVE_CAPTURE_3_3,
  UCT_SAVE_CAPTURE_SELF_ATARI,

  UCT_CAPTURE,
  UCT_CAPTURE_AFTER_KO,
  UCT_2POINT_CAPTURE_S_S,
  UCT_2POINT_CAPTURE_S_L,
  UCT_2POINT_CAPTURE_L_S,
  UCT_2POINT_CAPTURE_L_L,
  UCT_3POINT_CAPTURE_S_S,
  UCT_3POINT_CAPTURE_S_L,
  UCT_3POINT_CAPTURE_L_S,
  UCT_3POINT_CAPTURE_L_L,

  UCT_SEMEAI_CAPTURE,
  UCT_SELF_ATARI_SMALL,
  UCT_SELF_ATARI_NAKADE,
  UCT_SELF_ATARI_LARGE,
  UCT_SAVE_EXTENSION_1,
  UCT_SAVE_EXTENSION_2,
  UCT_SAVE_EXTENSION_3,
  UCT_LADDER_EXTENSION,
  UCT_ATARI,
  UCT_CAPTURABLE_ATARI,

  UCT_OIOTOSHI,
  UCT_SNAPBACK,
  UCT_2POINT_ATARI_S_S,
  UCT_2POINT_ATARI_S_L,
  UCT_2POINT_ATARI_L_S,
  UCT_2POINT_ATARI_L_L,
  UCT_2POINT_C_ATARI_S_S,
  UCT_2POINT_C_ATARI_S_L,
  UCT_2POINT_C_ATARI_L_S,
  UCT_2POINT_C_ATARI_L_L,
  UCT_3POINT_ATARI_S_S,
  UCT_3POINT_ATARI_S_L,
  UCT_3POINT_ATARI_L_S,
  UCT_3POINT_ATARI_L_L,
  UCT_3POINT_C_ATARI_S_S,
  UCT_3POINT_C_ATARI_S_L,
  UCT_3POINT_C_ATARI_L_S,
  UCT_3POINT_C_ATARI_L_L,
  UCT_3POINT_DAME_S_S,
  UCT_3POINT_DAME_S_L,
  UCT_3POINT_DAME_L_S,
  UCT_3POINT_DAME_L_L,
  UCT_2POINT_EXTENSION_DECREASE,
  UCT_2POINT_EXTENSION_EVEN,
  UCT_2POINT_EXTENSION_INCREASE,
  UCT_3POINT_EXTENSION_DECREASE,
  UCT_3POINT_EXTENSION_EVEN,
  UCT_3POINT_EXTENSION_INCREASE,
  UCT_THROW_IN_2,
  UCT_NAKADE_3,
  UCT_KEIMA_TSUKEKOSHI,
  UCT_DOUBLE_KEIMA,
  UCT_KO_CONNECTION,

  UCT_MAX,
};

enum PASS_FEATURES {
  UCT_PASS_AFTER_MOVE,
  UCT_PASS_AFTER_PASS,
  UCT_PASS_MAX,
};

const int LFR_DIMENSION = 5;

const int UCT_MASK_MAX = 64;
const int UCT_TACTICAL_FEATURE_MAX = UCT_MAX;
const int POS_ID_MAX = 64;
const int MOVE_DISTANCE_MAX = 16;
const int CFG_DISTANCE_MAX = 8;

const int LARGE_PAT_MAX = 150000;

const int OWNER_MAX = 11;
const int CRITICALITY_MAX = 7;

const int UCT_PHYSICALS_MAX = (1 << 14);

const double CRITICALITY_INIT = 0.765745;
const double CRITICALITY_BIAS = 0.036;

const double OWNER_K = 0.05;
const double OWNER_BIAS = 34.0;

const std::string uct_features_name[UCT_TACTICAL_FEATURE_MAX] = {
    "SAVE_CAPTURE_1_1           ", "SAVE_CAPTURE_1_2           ",
    "SAVE_CAPTURE_1_3           ", "SAVE_CAPTURE_2_1           ",
    "SAVE_CAPTURE_2_2           ",

    "SAVE_CAPTURE_2_3           ", "SAVE_CAPTURE_3_1           ",
    "SAVE_CAPTURE_3_2           ", "SAVE_CAPTURE_3_3           ",
    "SAVE_CAPTURE_SELF_ATARI    ",

    "CAPTURE                    ", "CAPTURE_AFTER_KO           ",
    "2POINT_CAPTURE_S_S         ", "2POINT_CAPTURE_S_L         ",
    "2POINT_CAPTURE_L_S         ",

    "2POINT_CAPTURE_L_L         ", "3POINT_CAPTURE_S_S         ",
    "3POINT_CAPTURE_S_L         ", "3POINT_CAPTURE_L_S         ",
    "3POINT_CAPTURE_L_L         ",

    "SEMEAI_CAPTURE             ", "SELF_ATARI_SMALL           ",
    "SELF_ATARI_NAKADE          ", "SELF_ATARI_LARGE           ",
    "SAVE_EXTENSION_1           ",

    "SAVE_EXTENSION_2           ", "SAVE_EXTENSION_3           ",
    "LADDER_EXTENSION           ", "ATARI                      ",
    "CAPTURABLE_ATARI           ",

    "OIOTOSHI                   ", "SNAPBACK                   ",
    "2POINT_ATARI_S_S           ", "2POINT_ATARI_S_L           ",
    "2POINT_ATARI_L_S           ", "2POINT_ATARI_L_L           ",
    "2POINT_C_ATARI_S_S         ", "2POINT_C_ATARI_S_L         ",
    "2POINT_C_ATARI_L_S         ", "2POINT_C_ATARI_L_L         ",

    "3POINT_ATARI_S_S           ", "3POINT_ATARI_S_L           ",
    "3POINT_ATARI_L_S           ", "3POINT_ATARI_L_L           ",
    "3POINT_C_ATARI_S_S         ", "3POINT_C_ATARI_S_L         ",
    "3POINT_C_ATARI_L_S         ", "3POINT_C_ATARI_L_L         ",

    "3POINT_DAME_S_S            ", "3POINT_DAME_S_L            ",
    "3POINT_DAME_L_S            ", "3POINT_DAME_L_L            ",
    "2POINT_EXTENSION_DECREASE  ", "2POINT_EXTENSION_EVEN      ",
    "2POINT_EXTENSION_INCREASE  ", "3POINT_EXTENSION_DECREASE  ",

    "3POINT_EXTENSION_EVEN      ", "3POINT_EXTENSION_INCREASE  ",
    "THROW_IN_2                 ", "NAKADE_3                   ",
    "KEIMA_TSUKEKOSHI           ",

    "DOUBLE_KEIMA               ", "KO_CONNECTION              ",
};

struct uct_features_t {
  unsigned long long tactical_features1[BOARD_MAX];
  unsigned long long tactical_features2[BOARD_MAX];
  unsigned long long tactical_features3[BOARD_MAX];
};

struct latent_factor_t {
  double w;
  double v[LFR_DIMENSION];
};

extern double uct_owner[OWNER_MAX];
extern double uct_criticality[CRITICALITY_MAX];

extern index_hash_t md3_index[HASH_MAX];
extern index_hash_t md4_index[HASH_MAX];
extern index_hash_t md5_index[HASH_MAX];

extern char uct_params_path[1024];

extern unsigned long long atari_mask;
extern unsigned long long capture_mask;

extern const unsigned long long uct_mask[UCT_MASK_MAX];

void InitializeUctRating(void);
void InitializePhysicalFeaturesSet(void);

double CalculateLFRScore(game_info_t *game, int pos, int pat_index[],
                         uct_features_t *uct_features);

void UctCheckFeatures(game_info_t *game, int color,
                      uct_features_t *uct_features);

void UctCheckRemove2Stones(game_info_t *game, int color,
                           uct_features_t *uct_features);

void UctCheckRemove3Stones(game_info_t *game, int color,
                           uct_features_t *uct_features);

void UctCheckCaptureAfterKo(game_info_t *game, int color,
                            uct_features_t *uct_features);

bool UctCheckSelfAtari(game_info_t *game, int color, int pos,
                       uct_features_t *uct_features);

void UctCheckCapture(game_info_t *game, int color, int pos,
                     uct_features_t *uct_features);

void UctCheckAtari(game_info_t *game, int color, int pos,
                   uct_features_t *uct_features);

void UctCheckSnapBack(game_info_t *game, int color, int pos,
                      uct_features_t *uct_features);

void UctCheckKeimaTsukekoshi(game_info_t *game, int color, int pos,
                             uct_features_t *uct_features);

void UctCheckDoubleKeima(game_info_t *game, int color, int pos,
                         uct_features_t *uct_features);

int UctCheckUtteGaeshi(game_info_t *game, int color, int pos,
                       uct_features_t *uct_features);

void UctCheckKoConnection(game_info_t *game, uct_features_t *uct_features);

void AnalyzeUctRating(game_info_t *game, int color, double rate[]);

#endif
