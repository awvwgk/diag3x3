// This is free and unencumbered software released into the public domain.
//
// Anyone is free to copy, modify, publish, use, compile, sell, or
// distribute this software, either in source code form or as a compiled
// binary, for any purpose, commercial or non-commercial, and by any
// means.
//
// In jurisdictions that recognize copyright laws, the author or authors
// of this software dedicate any and all copyright interest in the
// software to the public domain. We make this dedication for the benefit
// of the public at large and to the detriment of our heirs and
// successors. We intend this dedication to be an overt act of
// relinquishment in perpetuity of all present and future rights to this
// software under copyright law.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
//
// For more information, please refer to <https://unlicense.org>

/// Analytical implementation of the 3x3 eigenproblem and its solution.

#ifndef DIAG3X3_HPP
#define DIAG3X3_HPP

#include <array>
#include <cmath>
#include <algorithm>
#include <limits>

namespace diag3x3 {

/// 2π/3
inline constexpr double twothirdpi = 2.09439510239319526e+00;

/// Numerical precision
inline constexpr double eps = std::numeric_limits<double>::epsilon();

/// Calculates eigenvalues based on the trigonometric solution of A = pB + qI.
///
/// The symmetric input matrix is stored in column-major order as
/// a flat array of 9 elements: a[row + col*3], with row and col in [0, 2].
inline void eigval3x3(const std::array<double, 9>& a, std::array<double, 3>& w)
{
    double q, p, r;

    r = a[3]*a[3] + a[6]*a[6] + a[7]*a[7];
    q = (a[0] + a[4] + a[8]) / 3.0;
    w[0] = a[0] - q;
    w[1] = a[4] - q;
    w[2] = a[8] - q;
    p = std::sqrt((w[0]*w[0] + w[1]*w[1] + w[2]*w[2] + 2*r) / 6.0);
    r = (w[0] * (w[1]*w[2] - a[7]*a[7])
       - a[3] * (a[3]*w[2] - a[7]*a[6])
       + a[6] * (a[3]*a[7] - w[1]*a[6])) / (p*p*p) * 0.5;

    if (r <= -1.0) {
        r = 0.5 * twothirdpi;
    } else if (r >= 1.0) {
        r = 0.0;
    } else {
        r = std::acos(r) / 3.0;
    }

    w[2] = q + 2*p*std::cos(r);
    w[0] = q + 2*p*std::cos(r + twothirdpi);
    w[1] = 3*q - w[0] - w[2];
}

/// Calculates eigenvectors using an analytical method based on vector cross
/// products.
///
/// The symmetric input matrix is stored in column-major order as
/// a flat array of 9 elements: a[row + col*3], with row and col in [0, 2].
/// The matrix a is destroyed during the computation.
/// On exit, w contains the eigenvalues and the columns of q are the
/// corresponding eigenvectors.
inline void eigvec3x3(std::array<double, 9>& a, std::array<double, 3>& w,
                      std::array<double, 9>& q)
{
    double norm, n1, n2, n3, precon;
    int i;

    w[0] = std::max(std::abs(a[0]), std::abs(a[3]));
    w[1] = std::max(std::abs(a[6]), std::abs(a[4]));
    w[2] = std::max(std::abs(a[7]), std::abs(a[8]));
    precon = std::max(w[0], std::max(w[1], w[2]));

    // null matrix
    if (precon < eps) {
        w = {0.0, 0.0, 0.0};
        q = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
        return;
    }

    norm = 1.0 / precon;

    a[0] *= norm;
    a[3] *= norm;
    a[4] *= norm;
    a[6] *= norm;
    a[7] *= norm;
    a[8] *= norm;

    // Calculate eigenvalues
    eigval3x3(a, w);

    // Compute first eigenvector
    a[0] -= w[0];
    a[4] -= w[0];
    a[8] -= w[0];

    q[0] = a[3]*a[7] - a[6]*a[4];
    q[1] = a[6]*a[3] - a[0]*a[7];
    q[2] = a[0]*a[4] - a[3]*a[3];
    q[3] = a[3]*a[8] - a[6]*a[7];
    q[4] = a[6]*a[6] - a[0]*a[8];
    q[5] = a[0]*a[7] - a[3]*a[6];
    q[6] = a[4]*a[8] - a[7]*a[7];
    q[7] = a[7]*a[6] - a[3]*a[8];
    q[8] = a[3]*a[7] - a[4]*a[6];
    n1 = q[0]*q[0] + q[1]*q[1] + q[2]*q[2];
    n2 = q[3]*q[3] + q[4]*q[4] + q[5]*q[5];
    n3 = q[6]*q[6] + q[7]*q[7] + q[8]*q[8];

    norm = n1;
    i = 1;
    if (n2 > norm) {
        i = 2;
        norm = n1;
    }
    if (n3 > norm) {
        i = 3;
    }

    if (i == 1) {
        norm = std::sqrt(1.0 / n1);
        q[0] *= norm;
        q[1] *= norm;
        q[2] *= norm;
    } else if (i == 2) {
        norm = std::sqrt(1.0 / n2);
        q[0] = q[3] * norm;
        q[1] = q[4] * norm;
        q[2] = q[5] * norm;
    } else {
        norm = std::sqrt(1.0 / n3);
        q[0] = q[6] * norm;
        q[1] = q[7] * norm;
        q[2] = q[8] * norm;
    }

    // Robustly compute a right-hand orthonormal set (ev1, u, v)
    if (std::abs(q[0]) > std::abs(q[1])) {
        norm = std::sqrt(1.0 / (q[0]*q[0] + q[2]*q[2]));
        q[3] = -q[2] * norm;
        q[4] = 0.0;
        q[5] = +q[0] * norm;
    } else {
        norm = std::sqrt(1.0 / (q[1]*q[1] + q[2]*q[2]));
        q[3] = 0.0;
        q[4] = +q[2] * norm;
        q[5] = -q[1] * norm;
    }
    q[6] = q[1]*q[5] - q[2]*q[4];
    q[7] = q[2]*q[3] - q[0]*q[5];
    q[8] = q[0]*q[4] - q[1]*q[3];

    // Reset A
    a[0] += w[0];
    a[4] += w[0];
    a[8] += w[0];

    // A*U
    n1 = a[0]*q[3] + a[3]*q[4] + a[6]*q[5];
    n2 = a[3]*q[3] + a[4]*q[4] + a[7]*q[5];
    n3 = a[6]*q[3] + a[7]*q[4] + a[8]*q[5];

    // A*V, note out of order computation
    a[8] = a[6]*q[6] + a[7]*q[7] + a[8]*q[8];
    a[6] = a[0]*q[6] + a[3]*q[7] + a[6]*q[8];
    a[7] = a[3]*q[6] + a[4]*q[7] + a[7]*q[8];

    // UT*(A*U) - l2*E
    n1 = q[3]*n1 + q[4]*n2 + q[5]*n3 - w[1];
    // UT*(A*V)
    n2 = q[3]*a[6] + q[4]*a[7] + q[5]*a[8];
    // VT*(A*V) - l2*E
    n3 = q[6]*a[6] + q[7]*a[7] + q[8]*a[8] - w[1];

    if (std::abs(n1) >= std::abs(n3)) {
        norm = std::max(std::abs(n1), std::abs(n2));
        if (norm > eps) {
            if (std::abs(n1) >= std::abs(n2)) {
                n2 = n2 / n1;
                n1 = std::sqrt(1.0 / (1.0 + n2*n2));
                n2 = n2 * n1;
            } else {
                n1 = n1 / n2;
                n2 = std::sqrt(1.0 / (1.0 + n1*n1));
                n1 = n1 * n2;
            }
            q[3] = n2*q[3] - n1*q[6];
            q[4] = n2*q[4] - n1*q[7];
            q[5] = n2*q[5] - n1*q[8];
        }
    } else {
        norm = std::max(std::abs(n3), std::abs(n2));
        if (norm > eps) {
            if (std::abs(n3) >= std::abs(n2)) {
                n2 = n2 / n3;
                n3 = std::sqrt(1.0 / (1.0 + n2*n2));
                n2 = n2 * n3;
            } else {
                n3 = n3 / n2;
                n2 = std::sqrt(1.0 / (1.0 + n3*n3));
                n3 = n3 * n2;
            }
            q[3] = n3*q[3] - n2*q[6];
            q[4] = n3*q[4] - n2*q[7];
            q[5] = n3*q[5] - n2*q[8];
        }
    }

    // Calculate third eigenvector from cross product
    q[6] = q[1]*q[5] - q[2]*q[4];
    q[7] = q[2]*q[3] - q[0]*q[5];
    q[8] = q[0]*q[4] - q[1]*q[3];

    w[0] *= precon;
    w[1] *= precon;
    w[2] *= precon;
}

} // namespace diag3x3

#endif // DIAG3X3_HPP
