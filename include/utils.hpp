#if !defined(UTILS_H)

#include <cmath>

/**
 * @brief Compute the relative error between a numerical value and its exact value.
 *
 * The relative error is defined as |num − exact| / |exact|.
 *
 * @param num   The computed or approximate value.
 * @param exact The true or exact value (must be nonzero).
 * @return The relative error as a double.
 *
 */
double rel_err(double num, double exact);

#define UTILS_H
#endif
