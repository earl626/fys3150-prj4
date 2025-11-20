
#include "utils.hpp"

using namespace std;

double rel_err(double num, double exact) {
    return fabs(num - exact) / fabs(exact);
}
