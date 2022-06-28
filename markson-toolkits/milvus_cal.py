# -*- coding: utf-8 -*-
# @Author  : Markson Zhang
# @Time    : 2021/05/14 17:53
# @Function: milvus_resource_cal

import math


def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


def least_squares(x, core, dim=256):
    dim_weight = dim // 256
    x /= 1000000
    y1 = 1e-7 * x**2 + \
        1.2650756649383181 * x + \
        9.342112170565496

    y2 = 1.7472040684830614 * x**2 + \
        -133.90749263622965 * x + \
        -133.90749263622965

    return y / 1000 * dim_weight


def milvus_resource_cal(total, dim):
    '''Calculates'''
    single_vec_space = dim * 4
    total_space = total * single_vec_space

    output_log = '=========== Recommended Parameters ===========' + '\n'
    output_log += 'Mode: CPU if query < 1000 else GPU' + '\n'
    output_log += 'IndexType: IVF_SQ8' + '\n'
    output_log += 'IndexType create: nlist: 4096[best performance]/16384[best recall]' + '\n'
    output_log += 'IndexType search: nprobe: 128[best performance]/512[best recall]' + '\n'
    output_log += 'Index File Size: nprobe: 2048' + '\n'
    output_log += '=========== Recommended Resources ===========' + '\n'
    output_log += 'Overall Vectors Space: {}'.format(convert_size(total_space)) + '\n'
    output_log += 'Server Memory && Cache.cache_size: larger than {}'.format(convert_size(total_space * 0.3)) + '\n'
    output_log += '[48]CPUs cost {:.10f}'.format(least_squares(total, 48, dim)) + '\n'
    output_log += '[24]CPUs cost {:.10f}'.format(least_squares(total, 24, dim)) + '\n'
    output_log += '[16]CPUs cost {:.10f}'.format(least_squares(total, 16, dim)) + '\n'
    output_log += '[8]CPUs cost {:.10f}'.format(least_squares(total, 8, dim)) + '\n'
    output_log += '[4]CPUs cost {:.10f}'.format(least_squares(total, 4, dim)) + '\n'
    output_log += '[2]CPUs cost {:.10f}'.format(least_squares(total, 2, dim)) + '\n'

    print(output_log)


if __name__ == '__main__':
    print('Welcome to use milvus resources calculator.')
    total = int(input('Total nums: '))
    dim = int(input('Vector Dimensions: '))
    milvus_resource_cal(total, dim)
