# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.30

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/Alisiena/Programming/cpp/TensorMake/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/Alisiena/Programming/cpp/TensorMake/src/build

# Include any dependencies generated for this target.
include CMakeFiles/myprogram.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/myprogram.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/myprogram.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/myprogram.dir/flags.make

CMakeFiles/myprogram.dir/layer.cpp.o: CMakeFiles/myprogram.dir/flags.make
CMakeFiles/myprogram.dir/layer.cpp.o: /home/Alisiena/Programming/cpp/TensorMake/src/layer.cpp
CMakeFiles/myprogram.dir/layer.cpp.o: CMakeFiles/myprogram.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/Alisiena/Programming/cpp/TensorMake/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/myprogram.dir/layer.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/myprogram.dir/layer.cpp.o -MF CMakeFiles/myprogram.dir/layer.cpp.o.d -o CMakeFiles/myprogram.dir/layer.cpp.o -c /home/Alisiena/Programming/cpp/TensorMake/src/layer.cpp

CMakeFiles/myprogram.dir/layer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/myprogram.dir/layer.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/Alisiena/Programming/cpp/TensorMake/src/layer.cpp > CMakeFiles/myprogram.dir/layer.cpp.i

CMakeFiles/myprogram.dir/layer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/myprogram.dir/layer.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/Alisiena/Programming/cpp/TensorMake/src/layer.cpp -o CMakeFiles/myprogram.dir/layer.cpp.s

CMakeFiles/myprogram.dir/main.cpp.o: CMakeFiles/myprogram.dir/flags.make
CMakeFiles/myprogram.dir/main.cpp.o: /home/Alisiena/Programming/cpp/TensorMake/src/main.cpp
CMakeFiles/myprogram.dir/main.cpp.o: CMakeFiles/myprogram.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/Alisiena/Programming/cpp/TensorMake/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/myprogram.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/myprogram.dir/main.cpp.o -MF CMakeFiles/myprogram.dir/main.cpp.o.d -o CMakeFiles/myprogram.dir/main.cpp.o -c /home/Alisiena/Programming/cpp/TensorMake/src/main.cpp

CMakeFiles/myprogram.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/myprogram.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/Alisiena/Programming/cpp/TensorMake/src/main.cpp > CMakeFiles/myprogram.dir/main.cpp.i

CMakeFiles/myprogram.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/myprogram.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/Alisiena/Programming/cpp/TensorMake/src/main.cpp -o CMakeFiles/myprogram.dir/main.cpp.s

CMakeFiles/myprogram.dir/matrix.cpp.o: CMakeFiles/myprogram.dir/flags.make
CMakeFiles/myprogram.dir/matrix.cpp.o: /home/Alisiena/Programming/cpp/TensorMake/src/matrix.cpp
CMakeFiles/myprogram.dir/matrix.cpp.o: CMakeFiles/myprogram.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/Alisiena/Programming/cpp/TensorMake/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/myprogram.dir/matrix.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/myprogram.dir/matrix.cpp.o -MF CMakeFiles/myprogram.dir/matrix.cpp.o.d -o CMakeFiles/myprogram.dir/matrix.cpp.o -c /home/Alisiena/Programming/cpp/TensorMake/src/matrix.cpp

CMakeFiles/myprogram.dir/matrix.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/myprogram.dir/matrix.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/Alisiena/Programming/cpp/TensorMake/src/matrix.cpp > CMakeFiles/myprogram.dir/matrix.cpp.i

CMakeFiles/myprogram.dir/matrix.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/myprogram.dir/matrix.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/Alisiena/Programming/cpp/TensorMake/src/matrix.cpp -o CMakeFiles/myprogram.dir/matrix.cpp.s

CMakeFiles/myprogram.dir/neural_network.cpp.o: CMakeFiles/myprogram.dir/flags.make
CMakeFiles/myprogram.dir/neural_network.cpp.o: /home/Alisiena/Programming/cpp/TensorMake/src/neural_network.cpp
CMakeFiles/myprogram.dir/neural_network.cpp.o: CMakeFiles/myprogram.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/Alisiena/Programming/cpp/TensorMake/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/myprogram.dir/neural_network.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/myprogram.dir/neural_network.cpp.o -MF CMakeFiles/myprogram.dir/neural_network.cpp.o.d -o CMakeFiles/myprogram.dir/neural_network.cpp.o -c /home/Alisiena/Programming/cpp/TensorMake/src/neural_network.cpp

CMakeFiles/myprogram.dir/neural_network.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/myprogram.dir/neural_network.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/Alisiena/Programming/cpp/TensorMake/src/neural_network.cpp > CMakeFiles/myprogram.dir/neural_network.cpp.i

CMakeFiles/myprogram.dir/neural_network.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/myprogram.dir/neural_network.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/Alisiena/Programming/cpp/TensorMake/src/neural_network.cpp -o CMakeFiles/myprogram.dir/neural_network.cpp.s

# Object files for target myprogram
myprogram_OBJECTS = \
"CMakeFiles/myprogram.dir/layer.cpp.o" \
"CMakeFiles/myprogram.dir/main.cpp.o" \
"CMakeFiles/myprogram.dir/matrix.cpp.o" \
"CMakeFiles/myprogram.dir/neural_network.cpp.o"

# External object files for target myprogram
myprogram_EXTERNAL_OBJECTS =

myprogram: CMakeFiles/myprogram.dir/layer.cpp.o
myprogram: CMakeFiles/myprogram.dir/main.cpp.o
myprogram: CMakeFiles/myprogram.dir/matrix.cpp.o
myprogram: CMakeFiles/myprogram.dir/neural_network.cpp.o
myprogram: CMakeFiles/myprogram.dir/build.make
myprogram: CMakeFiles/myprogram.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/Alisiena/Programming/cpp/TensorMake/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable myprogram"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/myprogram.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/myprogram.dir/build: myprogram
.PHONY : CMakeFiles/myprogram.dir/build

CMakeFiles/myprogram.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/myprogram.dir/cmake_clean.cmake
.PHONY : CMakeFiles/myprogram.dir/clean

CMakeFiles/myprogram.dir/depend:
	cd /home/Alisiena/Programming/cpp/TensorMake/src/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/Alisiena/Programming/cpp/TensorMake/src /home/Alisiena/Programming/cpp/TensorMake/src /home/Alisiena/Programming/cpp/TensorMake/src/build /home/Alisiena/Programming/cpp/TensorMake/src/build /home/Alisiena/Programming/cpp/TensorMake/src/build/CMakeFiles/myprogram.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/myprogram.dir/depend

