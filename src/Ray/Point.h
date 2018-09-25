#ifndef _POINT_H_
#define _POINT_H_

#define GOGUI_X(pos) (gogui_x[CORRECT_X(pos)])
#define GOGUI_Y(pos) (pure_board_size + 1 - CORRECT_Y(pos))

const char gogui_x[] = {'I', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                        'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                        'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'};

int StringToInteger(const char *cpos);

void IntegerToString(const int pos, char *cpos);

#endif
