add_library(common STATIC
    qap.c
)

target_include_directories(common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}   # for <qap.h> success in tests
)