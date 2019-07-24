#include <cstdint>

struct vec4uint32 {
    // Significance decreases
    //  left -> right; up -> down
    uint32_t x;
    uint32_t y;
    uint32_t z;
    uint32_t w;


    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
    //%%%%%%%%%%%%%%%%%%%% Builtin Operator %%%%%%%%%%%%%%%%%%%%//
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
    // C-tor
    vec4uint32( uint32_t X, uint32_t Y, uint32_t Z, uint32_t W)
    {
        x = X;
        y = Y;
        z = Z;
        w = W;
    }
    vec4uint32( )
    {
        x = y = z = w = 0x0;
    }
    // assignment
    inline vec4uint32 &operator=(const vec4uint32 &v)  {
        x = v.x;
        y = v.y;
        z = v.z;
        w = v.w;
    }
    // eq
    inline bool operator==( const vec4uint32& v ) const
    {
        return(
                (x == v.x) &&
                (y == v.y) &&
                (z == v.z) &&
                (w == v.w)
              );
    }
    // neq
    inline bool operator!=( const vec4uint32 &v ) const
    {
        return( !(*this == v) );
    }
    // gt
    inline bool operator>( const vec4uint32 &rhs ) const
    {
        if( this->x > rhs.x ) {
            return true;
        } else if( this->x < rhs.x ) {
            return false;
        } else {
            if( this->y > rhs.y ) {
                return true;
            } else if( this->y < rhs.y ) {
                return false;
            } else{
                if( this->z > rhs.z ) {
                    return true;
                } else if( this->z < rhs.z ) {
                    return false;
                } else {
                    if( this->w > rhs.w ) {
                        return true;
                    } else if( this->w < rhs.w ) {
                        return false;
                    } else {
                        return false;
                    }
                }
            }
        }
    }
    // leq
    inline bool operator<=( const vec4uint32 &rhs ) const
    {
        return( !(*this > rhs) );
    }
    // le
    inline bool operator<(const vec4uint32 &rhs ) const
    {
        if( *this > rhs || *this == rhs ) {
            return false;
        } else {
            return true;
        }
    }
    // geq
    inline bool operator>=( const vec4uint32 &rhs ) const
    {
        return( !(*this > rhs) );
    }
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
};

typedef struct vec4uint32 vec4uint32_t;
