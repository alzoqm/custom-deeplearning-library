#include "half.h"

HalfFloat::HalfFloat() : value(0) {}

HalfFloat::HalfFloat(float f) 
{
    uint32_t bits = *reinterpret_cast<uint32_t*>(&f);
    uint16_t sign = (bits >> 31) & 0x1;
    uint16_t exp = (bits >> 23) & 0xff;
    uint16_t mantissa = bits & 0x7fffff;
    if (exp == 0) 
    {
        mantissa |= 0x800000;
        exp = 0;
    } else if (exp == 255) 
    {
        exp = 31;
    } 
    else 
    {
        exp -= 127 - 15;
        mantissa >>= 13;
    }
    value = (sign << 15) | (exp << 10) | mantissa;
}

HalfFloat::operator float() const 
{
    uint16_t sign = (value >> 15) & 0x1;
    uint16_t exp = (value >> 10) & 0x1f;
    uint16_t mantissa = value & 0x3ff;
    uint32_t bits;
    if (exp == 0) 
    {
        exp = 127 - 15;
        while ((mantissa & 0x200) == 0) 
        {
            mantissa <<= 1;
            exp--;
        }
        mantissa &= 0x1ff;
    } else if (exp == 31) 
    {
        exp = 255;
        mantissa <<= 13;
    } else 
    {
        exp += 127 - 15;
        mantissa <<= 13;
    }
    bits = (sign << 31) | (exp << 23) | mantissa;
    return *reinterpret_cast<float*>(&bits);
}

HalfFloat operator+(const HalfFloat& lhs, const HalfFloat& rhs) 
{
    return static_cast<float>(lhs) + static_cast<float>(rhs);
}

HalfFloat operator-(const HalfFloat& lhs, const HalfFloat& rhs) 
{
    return static_cast<float>(lhs) - static_cast<float>(rhs);
}

HalfFloat operator*(const HalfFloat& lhs, const HalfFloat& rhs) 
{
    return static_cast<float>(lhs) * static_cast<float>(rhs);
}

HalfFloat operator/(const HalfFloat& lhs, const HalfFloat& rhs) 
{
    return static_cast<float>(lhs) / static_cast<float>(rhs);
}

bool operator==(const HalfFloat& lhs, const HalfFloat& rhs) 
{
    return static_cast<float>(lhs) == static_cast<float>(rhs);
}

bool operator!=(const HalfFloat& lhs, const HalfFloat& rhs) 
{
    return static_cast<float>(lhs) != static_cast<float>(rhs);
}

bool operator<(const HalfFloat& lhs, const HalfFloat& rhs) 
{
    return static_cast<float>(lhs) < static_cast<float>(rhs);
}

bool operator>(const HalfFloat& lhs, const HalfFloat& rhs) 
{
    return static_cast<float>(lhs) > static_cast<float>(rhs);
}

bool operator<=(const HalfFloat& lhs, const HalfFloat& rhs) 
{
    return static_cast<float>(lhs) <= static_cast<float>(rhs);
}

bool operator>=(const HalfFloat& lhs, const HalfFloat& rhs) 
{
    return static_cast<float>(lhs) >= static_cast<float>(rhs);
}

