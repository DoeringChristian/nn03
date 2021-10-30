/* MIT License
 * 
 * Copyright (c) 2021 DoeringChristian
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#ifndef NMATH_H
#define NMATH_H

#ifndef NM_FLOAT
#define NM_FLOAT float
#endif //NM_FLOAT

#ifndef NM_INLINE
#define NM_INLINE inline
#endif //NM_INLINE

#ifndef NM_SIZE
#include <stddef.h>
#define NM_SIZE size_t
#endif //NM_SIZE

#ifndef NM_INT
#define NM_INT int
#endif //NM_INT

#ifndef NM_MALLOC
#include <stdlib.h>
#define NM_MALLOC(_size) malloc(_size)
#endif //NM_MALLOC

#ifndef NM_FREE
#include <stdlib.h>
#define NM_FREE(_p) free(_p)
#endif //NM_FREE

#ifndef NM_SQRT
#include <math.h>
#define NM_SQRT(_x) sqrtf(_x)
#endif //NM_SQRT

#ifndef NM_CEIL
#include <math.h>
#define NM_CEIL(_x) ceilf(_x)
#endif //NM_CEIL

#ifndef NM_FLOOR
#include <math.h>
#define NM_FLOOR(_x) ceilf(_x)
#endif //NM_FLOOR

#ifndef NM_MIN
#include <math.h>
#define NM_MIN(_x, _y) fminf(_x, _y)
#endif //NM_MIN

#ifndef NM_MAX
#include <math.h>
#define NM_MAX(_x, _y) fmaxf(_x, _y)
#endif //NM_MAX




/*
 * vecn: float array of length n represents vector of length n.
 * matn: float array of length n * n represents nxn matrix.
 * matnm: float array of length n * m represents nxm matrix. n: width, m: height.
 *
 * row major order.
 *
 * | 0       | 1   | ... | n-1    |
 * | n       | n+1 |     |        |
 * | ...     |     |     |        |
 * | n*(m-1) | ... |     | n*m -1 |
 *
 *
 * m[i + j*n]
 *
 */




/*
 * =======================
 * Matrix N x M          :
 * =======================
 */

static NM_INLINE NM_FLOAT *matnm_new(NM_SIZE n, NM_SIZE m){
    return (NM_FLOAT *)NM_MALLOC(sizeof(NM_FLOAT) * n * m);
}

static NM_INLINE void matnm_free(NM_FLOAT *dst){
    NM_FREE(dst);
}

static NM_INLINE NM_FLOAT *matnm_copy(NM_FLOAT *dst, const NM_FLOAT *src, NM_SIZE n, NM_SIZE m){
    NM_SIZE i;
    for(i = 0;i < n * m;i++)
        dst[i] = src[i];
    return dst;
}

static NM_INLINE NM_FLOAT *matnm_foreach_cb(NM_FLOAT *dst, const NM_FLOAT *src, void *data, NM_FLOAT (*cb)(NM_FLOAT src, void *data), NM_SIZE n, NM_SIZE m){
    NM_SIZE i;
    for(i = 0;i < n * m;i++)
        dst[i] = cb(src[i], data);
    return dst;
}

static NM_INLINE NM_FLOAT *matnm_zero(NM_FLOAT *dst, NM_SIZE n, NM_SIZE m){
    NM_SIZE i;
    for(i = 0;i < n*m;i++){
        dst[i] = 0.0;
    }
    return dst;
}

static NM_INLINE NM_FLOAT *matnm_one(NM_FLOAT *dst, NM_SIZE n, NM_SIZE m){
    NM_SIZE i, j;
    for(j = 0;j < m;j++){
        for(i = 0;i < n;i++){
            if(i == j)
                dst[i + j * n] = 1.0;
            else
                dst[i + j * n] = 0.0;
        }
    }
    return dst;
}

static NM_INLINE NM_FLOAT *matnm_at(NM_FLOAT *src, NM_SIZE col, NM_SIZE row, NM_SIZE n){
    return &src[col + row * n];
}

static NM_INLINE NM_FLOAT *matnm_att(NM_FLOAT *src, NM_SIZE col, NM_SIZE row, NM_SIZE m){
    return &src[row + col * m];
}

const static NM_INLINE NM_FLOAT *matnm_atc(const NM_FLOAT *src, NM_SIZE col, NM_SIZE row, NM_SIZE n){
    return &src[col + row * n];
}

const static NM_INLINE NM_FLOAT *matnm_attc(const NM_FLOAT *src, NM_SIZE col, NM_SIZE row, NM_SIZE m){
    return &src[row + col * m];
}

static NM_INLINE NM_FLOAT matnm_get(const NM_FLOAT *src, NM_SIZE col, NM_SIZE row, NM_SIZE n){
    return src[col + row * n];
}

static NM_INLINE NM_FLOAT matnm_gett(const NM_FLOAT *src, NM_SIZE col, NM_SIZE row, NM_SIZE m){
    return src[row + col * m];
}

static NM_INLINE NM_FLOAT *matnm_neg(NM_FLOAT *dst, const NM_FLOAT *src, NM_SIZE n, NM_SIZE m){
    NM_SIZE i;
    for(i = 0;i < n * m;i++)
        dst[i] = -src[i];
    return dst;
}

static NM_INLINE NM_FLOAT *matnm_t(NM_FLOAT * restrict dst, const NM_FLOAT *restrict src, NM_SIZE n, NM_SIZE m){
    NM_SIZE i, j;
    for(j = 0;j < m;j++){
        for(i = 0;i < n;i++){
            *matnm_att(dst, i, j, m) = matnm_get(src, i, j, n);
        }
    }
    return dst;
}

static NM_INLINE NM_FLOAT *matnm_add(NM_FLOAT *dst, const NM_FLOAT *src1, const NM_FLOAT *src2, NM_SIZE n, NM_SIZE m){
    NM_SIZE i;
    for(i = 0;i < n * m;i++)
        dst[i] = src1[i] + src2[i];
    return dst;
}

// n, m from non transposed matrix
static NM_INLINE NM_FLOAT *matnm_addt(NM_FLOAT *dst, const NM_FLOAT *src1, const NM_FLOAT *src2, NM_SIZE n1, NM_SIZE m1){
    NM_SIZE i, j;
    for(i = 0;i < n1;i++)
        for(j = 0;j < m1;j++)
            *matnm_at(dst, i, j, n1) = matnm_get(src1, i, j, n1) + matnm_gett(src2, i, j, n1);
    return dst;
}

static NM_INLINE NM_FLOAT *matnm_sub(NM_FLOAT *dst, const NM_FLOAT *src1, const NM_FLOAT *src2, NM_SIZE n, NM_SIZE m){
    NM_SIZE i;
    for(i = 0;i < n * m;i++)
        dst[i] = src1[i] - src2[i];
    return dst;
}

static NM_INLINE NM_FLOAT *matnm_subt(NM_FLOAT *dst, const NM_FLOAT *src1, const NM_FLOAT *src2, NM_SIZE n1, NM_SIZE m1){
    NM_SIZE i, j;
    for(i = 0;i < n1;i++)
        for(j = 0;j < m1;j++)
            *matnm_at(dst, i, j, n1) = matnm_get(src1, i, j, n1) - matnm_gett(src2, i, j, n1);
    return dst;
}

static NM_INLINE NM_FLOAT *matnm_scale(NM_FLOAT *dst, const NM_FLOAT *src, NM_FLOAT s, NM_FLOAT n, NM_FLOAT m){
    NM_SIZE i;
    for(i = 0;i < n * m;i++)
        dst[i] = src[i] * s;
    return dst;
}

static NM_INLINE NM_SIZE matnm_mult_n(NM_SIZE n1, NM_SIZE m1, NM_SIZE n2, NM_SIZE m2){
    return n2;
}

static NM_INLINE NM_SIZE matnm_mult_m(NM_SIZE n1, NM_SIZE m1, NM_SIZE n2, NM_SIZE m2){
    return m1;
}

static NM_INLINE NM_SIZE matnm_mult_able(NM_SIZE n1, NM_SIZE m1, NM_SIZE n2, NM_SIZE m2){
    return (n1 == m2);
}

static NM_INLINE NM_FLOAT *matnm_mult(NM_FLOAT *restrict dst, const NM_FLOAT *restrict src1, const NM_FLOAT *restrict src2, NM_SIZE n1, NM_SIZE m1, NM_SIZE n2){
    NM_SIZE i, j, k;
    for(i = 0;i < n2;i++){
        for(j = 0;j < m1;j++){
            *matnm_at(dst, i, j, n2) = 0.0;
            for(k = 0;k < n1;k++)
                *matnm_at(dst, i, j, n2) += matnm_get(src1, k, j, n1) * matnm_get(src2, i, k, n2);
        }
    }
    return dst;
}

static NM_INLINE NM_SIZE matnm_multt_n(NM_SIZE n1, NM_SIZE m1, NM_SIZE n2, NM_SIZE m2){
    return m2;
}

static NM_INLINE NM_SIZE matnm_multt_m(NM_SIZE n1, NM_SIZE m1, NM_SIZE n2, NM_SIZE m2){
    return m1;
}

static NM_INLINE NM_SIZE matnm_multt_able(NM_SIZE n1, NM_SIZE m1, NM_SIZE n2, NM_SIZE m2){
    return (n1 == n2);
}

static NM_INLINE NM_FLOAT *matnm_multt(NM_FLOAT *restrict dst, const NM_FLOAT *restrict src1, const NM_FLOAT *restrict src2_t, NM_SIZE n1, NM_SIZE m1, NM_SIZE m2){
    NM_SIZE i, j, k;
    for(i = 0;i < m2;i++){
        for(j = 0;j < m1;j++){
            *matnm_at(dst, i, j, m2) = 0.0;
            for(k = 0;k < n1;k++)
                *matnm_at(dst, i, j, m2) += matnm_get(src1, k, j, n1) * matnm_gett(src2_t, i, k, m2);
        }
    }
    return dst;
}

static NM_INLINE NM_FLOAT *matnm_mult_at(NM_FLOAT *restrict dst, const NM_FLOAT *restrict src1, const NM_FLOAT *restrict src2, NM_SIZE n1, NM_SIZE m1, NM_SIZE n2, NM_SIZE m2, NM_INT x, NM_INT y){
    NM_SIZE i, j;
    for(i = 0;i < n2;i++){
        for(j = 0;j < m2;j++){
            *matnm_at(dst, i, j, n2) = matnm_get(src1, i + x, j + y, n1) * matnm_get(src2, i, j, n2);
        }
    }
    return dst;
}

static NM_INLINE NM_FLOAT matnm_conv_at(const NM_FLOAT *src1, const NM_FLOAT *src2, NM_SIZE n1, NM_SIZE m1, NM_SIZE n2, NM_SIZE m2, NM_INT x, NM_INT y){
    NM_SIZE i, j;
    NM_FLOAT dst = 0.0;
    for(i = 0;i < n2;i++){
        for(j = 0;j < m2;j++){
            dst += matnm_get(src1, i + x, j + y, n1) * matnm_get(src2, i, j, n2);
        }
    }
    return dst;
}

/*
 * =======================
 * Matrix N x N          :
 * =======================
 *
 * N x N matrix operations can be generalized to N x M matrix operations.
 */

static NM_INLINE NM_FLOAT *matn_new(NM_SIZE n, NM_SIZE m){
    return (NM_FLOAT *)NM_MALLOC(sizeof(NM_FLOAT) * n * m);
}

static NM_INLINE void matn_free(NM_FLOAT *dst){
    NM_FREE(dst);
}

static NM_INLINE NM_FLOAT *matn_copy(NM_FLOAT *dst, const NM_FLOAT *src, NM_SIZE n){
    return matnm_copy(dst, src, n, n);
}

static NM_INLINE NM_FLOAT *matn_foreach_cb(NM_FLOAT *dst, const NM_FLOAT *src, void *data, NM_FLOAT (*cb)(NM_FLOAT src, void *data), NM_SIZE n, NM_SIZE m){
    return matnm_foreach_cb(dst, src, data, cb, n, n);
}

static NM_INLINE NM_FLOAT *matn_one(NM_FLOAT *dst, NM_SIZE n){
    return matnm_one(dst, n, n);
}

static NM_INLINE NM_FLOAT *matn_zero(NM_FLOAT *dst, NM_SIZE n){
    return matnm_zero(dst, n, n);
}

static NM_INLINE NM_FLOAT *matn_at(NM_FLOAT *src, NM_SIZE row, NM_SIZE col, NM_SIZE n){
    return matnm_at(src, col, row, n);
}

static NM_INLINE NM_FLOAT *matn_att(NM_FLOAT *src, NM_SIZE row, NM_SIZE col, NM_SIZE n){
    return matnm_att(src, col, row, n);
}

const static NM_INLINE NM_FLOAT *matn_atc(const NM_FLOAT *src, NM_SIZE row, NM_SIZE col, NM_SIZE n){
    return matnm_atc(src, col, row, n);
}

const static NM_INLINE NM_FLOAT *matn_attc(const NM_FLOAT *src, NM_SIZE row, NM_SIZE col, NM_SIZE n){
    return matnm_attc(src, col, row, n);
}


static NM_INLINE NM_FLOAT matn_get(const NM_FLOAT *src, NM_SIZE row, NM_SIZE col, NM_SIZE n){
    return matnm_get(src, col, row, n);
}

static NM_INLINE NM_FLOAT matn_gett(const NM_FLOAT *src, NM_SIZE row, NM_SIZE col, NM_SIZE n){
    return matnm_gett(src, col, row, n);
}

static NM_INLINE NM_FLOAT *matn_t(NM_FLOAT *dst, const NM_FLOAT *src, NM_SIZE n){
    return matnm_t(dst, src, n, n);
}

static NM_INLINE NM_FLOAT *matn_add(NM_FLOAT *dst, const NM_FLOAT *src1, const NM_FLOAT *src2, NM_SIZE n){
    return matnm_add(dst, src1, src2, n, n);
}

static NM_INLINE NM_FLOAT *matn_sub(NM_FLOAT *dst, const NM_FLOAT *src1, const NM_FLOAT *src2, NM_SIZE n){
    return matnm_sub(dst, src1, src2, n, n);
}

static NM_INLINE NM_FLOAT *matn_scale(NM_FLOAT *dst, const NM_FLOAT *src, NM_FLOAT s, NM_FLOAT n){
    return matnm_scale(dst, src, s, n, n);
}

static NM_INLINE NM_FLOAT *matn_mult(NM_FLOAT *restrict dst, const NM_FLOAT *restrict src1, const NM_FLOAT *restrict src2, NM_SIZE n){
    return matnm_mult(dst, src1, src2, n, n, n);
}

static NM_INLINE NM_FLOAT *matn_multt(NM_FLOAT *restrict dst, const NM_FLOAT *restrict src1, const NM_FLOAT *restrict src2_t, NM_SIZE n){
    return matnm_multt(dst, src1, src2_t, n, n, n);
}

static NM_INLINE NM_FLOAT *matn_mult_vecn(NM_FLOAT *restrict dst_v, const NM_FLOAT *restrict src1_m, const NM_FLOAT *restrict src2_v, NM_SIZE n){
    return matnm_mult(dst_v, src1_m, src2_v, n, n, 1);
}

/*
 * =======================
 * Vector N x 1 or 1 X N :
 * =======================
 *
 * Vectors are column and row matrices in one since matrices are stored in row major order.
 */

static NM_INLINE NM_FLOAT *vecn_new(NM_SIZE n){
    return (NM_FLOAT *)NM_MALLOC(sizeof(NM_FLOAT) * n);
}

static NM_INLINE void vecn_free(NM_FLOAT *dst){
    NM_FREE(dst);
}

static NM_INLINE NM_FLOAT *vecn_copy(NM_FLOAT *dst, const NM_FLOAT *src, NM_SIZE n){
    return matnm_copy(dst, src, n, 1);
}

static NM_INLINE NM_FLOAT *vecn_foreach_cb(NM_FLOAT *dst, const NM_FLOAT *src, void *data, NM_FLOAT (*cb)(NM_FLOAT src, void *data), NM_SIZE n){
    return matnm_foreach_cb(dst, src, data, cb, n, 1);
}

static NM_INLINE NM_FLOAT *vecn_zero(NM_FLOAT *dst, NM_SIZE n){
    return matnm_zero(dst, n, 1);
}

static NM_INLINE NM_FLOAT *vecn_one(NM_FLOAT *dst, NM_SIZE n){
    NM_SIZE i;
    for(i = 0;i < n;i++)
        dst[i] = 1.0;
    return dst;
}

static NM_INLINE NM_FLOAT *vecn_add(NM_FLOAT *dst, const NM_FLOAT *src1, const NM_FLOAT *src2, NM_SIZE n){
    return matnm_add(dst, src1, src2, n, 1);
}

static NM_INLINE NM_FLOAT *vecn_sub(NM_FLOAT *dst, const NM_FLOAT *src1, const NM_FLOAT *src2, NM_SIZE n){
    return matnm_sub(dst, src1, src2, n, 1);
}

static NM_INLINE NM_FLOAT vecn_dot(const NM_FLOAT *src1, const NM_FLOAT *src2, NM_SIZE n){
    NM_FLOAT dst;
    matnm_mult(&dst, src1, src2, n, 1, 1);
    return dst;
}

static NM_INLINE NM_FLOAT *vecn_scale(NM_FLOAT *dst, const NM_FLOAT *src, const NM_FLOAT s, NM_SIZE n){
    return matnm_scale(dst, src, s, n, 1);
}

static NM_INLINE NM_FLOAT *vecn_neg(NM_FLOAT *dst, const NM_FLOAT *src, NM_SIZE n){
    return matnm_neg(dst, src, n, 1);
}

static NM_INLINE NM_FLOAT vecn_len_squared(const NM_FLOAT *src, NM_SIZE n){
    NM_SIZE i;
    NM_FLOAT dst;
    for(i = 0;i < n;i++)
        dst += src[i] * src[i];
    return dst;
}

static NM_INLINE NM_FLOAT vecn_len(const NM_FLOAT *src, NM_SIZE n){
    return NM_SQRT(vecn_len_squared(src, n));
}

static NM_INLINE NM_FLOAT vecn_dist_squared(const NM_FLOAT *src1, const NM_FLOAT *src2, NM_SIZE n){
    NM_SIZE i;
    NM_FLOAT dst;
    for(i = 0;i < n;i++)
        dst += (src1[i] - src2[i]) * (src1[i] - src2[i]);
    return dst;
}

static NM_INLINE NM_FLOAT vecn_dist(const NM_FLOAT *src1, const NM_FLOAT *src2, NM_SIZE n){
    return NM_SQRT(vecn_dist_squared(src1, src2, n));
}

static NM_INLINE NM_FLOAT *vecn_normalize(NM_FLOAT *dst, const NM_FLOAT *src, NM_SIZE n){
    NM_SIZE i;
    NM_FLOAT len;
    len = vecn_len(src, n);
    for(i = 0;i < n;i++)
        dst[i] = src[i] / len;
    return dst;
}

static NM_INLINE NM_FLOAT *vecn_min(NM_FLOAT *dst, const NM_FLOAT *src1, const NM_FLOAT *src2, NM_SIZE n){
    NM_SIZE i;
    for(i = 0;i < n;i++)
        dst[i] = NM_MIN(src1[i], src2[i]);
    return dst;
}

static NM_INLINE NM_FLOAT *vecn_max(NM_FLOAT *dst, const NM_FLOAT *src1, const NM_FLOAT *src2, NM_SIZE n){
    NM_SIZE i;
    for(i = 0;i < n;i++)
        dst[i] = NM_MAX(src1[i], src2[i]);
    return dst;
}

static NM_INLINE NM_FLOAT *vecn_ceil(NM_FLOAT *dst, const NM_FLOAT *src, NM_SIZE n){
    NM_SIZE i;
    for(i = 0;i < n;i++)
        dst[i] = NM_CEIL(src[i]);
    return dst;
}

static NM_INLINE NM_FLOAT *vecn_floor(NM_FLOAT *dst, const NM_FLOAT *src, NM_SIZE n){
    NM_SIZE i;
    for(i = 0;i < n;i++)
        dst[i] = NM_FLOOR(src[i]);
    return dst;
}


#endif //NMATH_H
