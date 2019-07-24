#include <iostream>
#include <vector>


class MyObject{
    public:
    int x;
    MyObject* next;
    MyObject(){ x = 0; next = NULL;};
};

int main(int argc, char const* argv[])
{
    std::vector<MyObject*> mo;
    MyObject* mo1 = new MyObject();
    mo.push_back(mo1);

    // this works
    for (auto it = mo.begin(); it != mo.end(); ++it){
        std::cout << "element x " << (*it)->x << std::endl;
    }

    // adding this line is ok
    std::vector<MyObject *> MO(0);
    auto min = MO.begin();

    // adding this line is not ok
    std::cout << (*min)->x << std::endl;
    return 0;
}

