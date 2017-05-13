#pragma once

#include <stdint.h>
#include <climits>
#include "Structures.h"

static uint64_t murmurHash3(void) {
	static uint64_t x = 23666ULL;/* The state must be seeded with a nonzero value. */

	x ^= x >> 33;
	x *= 0xff51afd7ed558ccdULL;
	x ^= x >> 33;
	x *= 0xc4ceb9fe1a85ec53ULL;

	return x ^= x >> 33;
}

static void initializeXor1024(Xor1024 &xor)
{
	xor.p = 0;

	for (size_t i = 0; i < 16; ++i)
	{
		xor.s1024[i] = murmurHash3();
	}
}

static void generateRandoms(Xor1024 &xor, uint64_t *results, uint64_t count)
{
	uint64_t s0, s1;

	for (size_t i = 0; i < count; ++i)
	{
		s0 = xor.s1024[xor.p];
		s1 = xor.s1024[xor.p = (xor.p + 1) & 15];

		s1 ^= s1 << 31; // a
		s1 ^= s1 >> 11; // b
		s0 ^= s0 >> 30; // c

		results[i] = (xor.s1024[xor.p] = s0 ^ s1) * 1181783497276652981LL;
	}
}

static void generateRandoms(Xor1024 &xor, float *results, size_t count) //yay union FTW, kein cast nötig :)
{
	uint64_t s0, s1;
	FastFloat ff;

	for (size_t i = 0; i < count; ++i)
	{
		s0 = xor.s1024[xor.p];
		s1 = xor.s1024[xor.p = (xor.p + 1) & 15];

		s1 ^= s1 << 31; // a
		s1 ^= s1 >> 11; // b
		s0 ^= s0 >> 30; // c

		ff.i = (((xor.s1024[xor.p] = s0 ^ s1) * 1181783497276652981LL) & 0x007fffff | 0x3f800000); //das 007fff um die 23 bits zu halten, 3f8 um range 1-2 zu bekommen

		results[i] = ff.x - 1.0f; // dann 1 abzeihen damit 0-1
	}
}

static void generateRandoms(Xor1024 &xor, float *results, size_t count, float min, float max)
{
	uint64_t s0, s1;
	FastFloat ff;
	float range = max - min;

	for (size_t i = 0; i < count; ++i)
	{
		s0 = xor.s1024[xor.p];
		s1 = xor.s1024[xor.p = (xor.p + 1) & 15];

		s1 ^= s1 << 31; // a
		s1 ^= s1 >> 11; // b
		s0 ^= s0 >> 30; // c

		ff.i = (((xor.s1024[xor.p] = s0 ^ s1) * 1181783497276652981LL) & 0x007fffff | 0x3f800000); //das 007fff um die 23 bits zu halten, 3f8 um range 1-2 zu bekommen

		results[i] = (ff.x - 1.0f) * range + min;
	}
}