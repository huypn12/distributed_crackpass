#include <iostream>
#include <vector>
#include <cassert>
 
class MO{
public:
    int x;
    MO* next;
    MO(){ x = 0; next = NULL;};
    MO(int a) { x = a; next = NULL;};
};
static int x_global;
int main(){
    x_global = 1;
    MO* mo1 = new MO(1);
    MO* mo2 = new MO(2);
    MO* mo3 = new MO(3);
 
    std::vector<MO*> candidate;
    std::vector<MO*> result;
    candidate.push_back(mo1);
    candidate.push_back(mo2);
    candidate.push_back(mo3);
 
    // auto it = std::find_if(candidate.begin(), candidate.end(), [](MO* mo){return mo->x > x_global});
    auto it = std::find_if(candidate.begin(), candidate.end(), [](MO* mo){return mo->x > x_global;});
    while (it != candidate.end()){
        result.push_back(*it);
        std::advance(it, 1);
        // it = std::find_if(it, candidate.end(), [](MO* mo){return mo->x > x_global});
        it = std::find_if(it, candidate.end(), [](MO* mo){return mo->x > x_global;});
    }
 
    for (auto it = result.begin(); it != result.end(); ++it){
        std::cout << "result is " << (*it)->x << std::endl;
    }
 
    auto min = result.begin();
    std::cout << (*min)->x << std::endl;
    assert(!result.empty());
 
 
    return 0;
}
