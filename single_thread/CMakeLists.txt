add_executable(sa_single
    sa_single.c
    main.c
)

target_link_libraries(sa_single PRIVATE common)
target_link_libraries(sa_single PRIVATE common m) 

target_include_directories(sa_single PRIVATE
    ${CMAKE_SOURCE_DIR}/common
)
