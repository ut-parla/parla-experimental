#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include <backend.hpp>
#include <algorithm>
#include <string>
#include <list>
#include <tuple>


#define DOCTEST_VALUE_PARAMETERIZED_DATA(case_idx, data, data_container)                                  \
    static size_t _doctest_subcase_idx = 0;                                                     \
    std::for_each(data_container.begin(), data_container.end(), [&](const auto& in) {           \
        SUBCASE((std::string(#data_container "[") +                                     \
                        std::to_string(_doctest_subcase_idx++) + "]").c_str()) { data = in; case_idx = _doctest_subcase_idx;}  \
    });                                                                                         \
    _doctest_subcase_idx = 0



TEST_CASE("CppMath::add") {
    std::tuple<int, int> data;
    int case_idx = 0;
    std::list<int> input = {5, 6};
    std::list<int> output = {7, 8};
    std::vector<std::tuple<int, int>> data_container;

    std::transform(input.begin(), input.end(), output.begin(), std::back_inserter(data_container),
                   [](int a, int b) { return std::make_tuple(a, b); });



    DOCTEST_VALUE_PARAMETERIZED_DATA(case_idx, data, data_container);

    CppMath math = CppMath();
    printf("Case %d: %d + %d = %d \n", case_idx, 2, std::get<0>(data), std::get<1>(data));
    CHECK(math.add(2, std::get<0>(data)) == std::get<1>(data));
}

TEST_CASE("CppMath::sub") {
    std::tuple<int, int> data;
    int case_idx = 0;
    std::list<int> input = {5, 6};
    std::list<int> output = {3, 4};
    std::vector<std::tuple<int, int>> data_container;

    std::transform(input.begin(), input.end(), output.begin(), std::back_inserter(data_container),
                   [](int a, int b) { return std::make_tuple(a, b); });

    DOCTEST_VALUE_PARAMETERIZED_DATA(case_idx, data, data_container);

    CppMath math = CppMath();
    printf("Case %d: %d - %d = %d \n", case_idx, std::get<0>(data), 2, std::get<1>(data));
    CHECK(math.sub(std::get<0>(data), 2) == std::get<1>(data));
}