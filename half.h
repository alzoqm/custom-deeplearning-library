#ifndef _HALF_H_
#define _HALF_H_

#include <cstdint>
#include <iostream>

using namespace std;

class HalfFloat 
{
private:
    unsigned short value;

public:
    HalfFloat();
    HalfFloat(float f);
    operator float() const;
};

HalfFloat operator+(const HalfFloat& lhs, const HalfFloat& rhs);

HalfFloat operator-(const HalfFloat& lhs, const HalfFloat& rhs);

HalfFloat operator*(const HalfFloat& lhs, const HalfFloat& rhs);

HalfFloat operator/(const HalfFloat& lhs, const HalfFloat& rhs);

bool operator==(const HalfFloat& lhs, const HalfFloat& rhs);

bool operator!=(const HalfFloat& lhs, const HalfFloat& rhs);

bool operator<(const HalfFloat& lhs, const HalfFloat& rhs);

bool operator>(const HalfFloat& lhs, const HalfFloat& rhs);

bool operator<=(const HalfFloat& lhs, const HalfFloat& rhs);

bool operator>=(const HalfFloat& lhs, const HalfFloat& rhs);

#endif