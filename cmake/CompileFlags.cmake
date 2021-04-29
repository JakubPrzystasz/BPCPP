if(MSVC)
	# visual studio compile flags
	set(CMAKE_CXX_FLAGS_DEBUG "/W3 /DDEBUG_BUILD /MDd /Zi /Od /RTC1 /GR /MP /std:c++14")
	set(CMAKE_CXX_FLAGS_RELEASE "/W3 /MD /O2 /GR- /MP /std:c++14")
else()
	#g++ compile flags
	set(CMAKE_CXX_FLAGS "-Wall -msse2 -fno-rtti -pipe -lpython3.8 -I/usr/include/python3.8 -I/home/jakub/.local/lib/python3.8/site-packages/numpy/core/include")
	set(CMAKE_CXX_FLAGS_DEBUG "-Wall -msse2 -g -fno-rtti -pipe -Wextra -Werror -lpython3.8 -I/usr/include/python3.8 -I/home/jakub/.local/lib/python3.8/site-packages/numpy/core/include")
	set(CMAKE_CXX_FLAGS_RELEASE "-Wall -O3 -msse2 -fno-rtti -pipe -lpython3.8 -I/usr/include/python3.8 -I/home/jakub/.local/lib/python3.8/site-packages/numpy/core/include")

	if(CMAKE_BUILD_TYPE STREQUAL "Debug")
		add_definitions(-DDEBUG_BUILD)
	endif()
endif()
