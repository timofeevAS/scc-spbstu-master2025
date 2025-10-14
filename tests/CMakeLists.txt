enable_testing()

add_executable(common_tests
    common_tests.c
    utils/utils.c
)

target_link_libraries(common_tests PRIVATE common)

target_include_directories(common_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/tests/utils
)

add_test(NAME common COMMAND ${CMAKE_CURRENT_BINARY_DIR}/common_tests)
