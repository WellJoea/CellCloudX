#ifndef __cellcloudx_kabsch_h__
#define __cellcloudx_kabsch_h__

#include "types.h"
#include <utility>

namespace cellcloudx {

typedef std::pair<Matrix3, Vector3> KabschResult;
typedef std::pair<Matrix2, Vector2> KabschResult2d;

KabschResult computeKabsch(const MatrixX3& model,
                           const MatrixX3& target,
                           const Vector& weight);

KabschResult2d computeKabsch2d(const MatrixX2& model,
                               const MatrixX2& target,
                               const Vector& weight);

}  // namespace cellcloudx

#endif