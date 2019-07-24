#include <iostream>

#include <boost/multiprecision/cpp_int.hpp>


int main(int argc, char const* argv[])
{
    boost::multiprecision::cpp_int dividend(0);
    boost::multiprecision::cpp_int divisor(0);
    boost::multiprecision::cpp_int quotient(0);
    boost::multiprecision::cpp_int remainder(0);


    // init dividend
    for (int i = 0; i < 5; i++) {
        // 0xFFFF = 4bytes = sizeof(uint32_t)
        dividend = dividend << 32;
        dividend = dividend | 0xFFFF;
    }

    // init divisor
    divisor = divisor | 0xFEFEFEFE;

    // modulo op
    quotient = dividend / divisor;
    remainder = dividend % divisor;

    std::cout << "quotient:  " << quotient << std::endl;
    std::cout << "remainder: " << remainder << std::endl;
    return 0;
}
