#pragma once

#include "gptvm/runtime/object.h"

#include <vector>

namespace gptvm {

/// Perform all-reduce of sum on a given list of objects.
///
/// \param objs The list of objects to be reduced. All objects must have the
///             same size. All objects hold the same data content of sum of
///             data from each object after the function returns.
void obAllReduceSum(std::vector<GVObject> objs);

} // namespace gptvm
