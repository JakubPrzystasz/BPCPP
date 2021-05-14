if(MSVC)
	# visual studio compile flags
	set(CMAKE_CXX_FLAGS_DEBUG "/W3 /DDEBUG_BUILD /MDd /Zi /Od /RTC1 /GR /MP /std:c++17")
	set(CMAKE_CXX_FLAGS_RELEASE "/W3 /MD /O2 /GR- /MP /std:c++17")
	# Set compile options in case of MSVC
	add_compile_options(/MP)  # enable parallel build
	add_definitions("/wd4267 /wd4244")  # ignore silly warnings related to size_t/int and site_t/double conversions
else()
	#g++ compile flags
	set(CMAKE_CXX_FLAGS "-Wall -msse2 -fno-rtti -pipe -std=gnu++17")
	set(CMAKE_CXX_FLAGS_DEBUG "-Wall -g -fno-rtti -pipe -std=gnu++17")
	set(CMAKE_CXX_FLAGS_RELEASE "-Wall -O3 -msse2 -fno-rtti -pipe -std=gnu++17")

	# set(CMAKE_CXX_FLAGS "-Wall -msse2 -fno-rtti -pipe")
	# set(CMAKE_CXX_FLAGS_DEBUG "-Wall -g -fno-rtti -pipe -Wextra -Werror")
	# set(CMAKE_CXX_FLAGS_RELEASE "-Wall -O3 -msse2 -fno-rtti -pipe")


	if(CMAKE_BUILD_TYPE STREQUAL "Debug")
		add_definitions(-DDEBUG_BUILD)
	endif()
endif()
