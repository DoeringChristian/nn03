# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.21

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
CMAKE_SOURCE_DIR = /home/doeringc/Projects/nn03

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/doeringc/Projects/nn03/build

# Include any dependencies generated for this target.
include CMakeFiles/nn.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/nn.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/nn.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/nn.dir/flags.make

CMakeFiles/nn.dir/src/layer_mat.c.o: CMakeFiles/nn.dir/flags.make
CMakeFiles/nn.dir/src/layer_mat.c.o: ../src/layer_mat.c
CMakeFiles/nn.dir/src/layer_mat.c.o: CMakeFiles/nn.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/doeringc/Projects/nn03/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/nn.dir/src/layer_mat.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/nn.dir/src/layer_mat.c.o -MF CMakeFiles/nn.dir/src/layer_mat.c.o.d -o CMakeFiles/nn.dir/src/layer_mat.c.o -c /home/doeringc/Projects/nn03/src/layer_mat.c

CMakeFiles/nn.dir/src/layer_mat.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/nn.dir/src/layer_mat.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/doeringc/Projects/nn03/src/layer_mat.c > CMakeFiles/nn.dir/src/layer_mat.c.i

CMakeFiles/nn.dir/src/layer_mat.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/nn.dir/src/layer_mat.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/doeringc/Projects/nn03/src/layer_mat.c -o CMakeFiles/nn.dir/src/layer_mat.c.s

CMakeFiles/nn.dir/src/main.c.o: CMakeFiles/nn.dir/flags.make
CMakeFiles/nn.dir/src/main.c.o: ../src/main.c
CMakeFiles/nn.dir/src/main.c.o: CMakeFiles/nn.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/doeringc/Projects/nn03/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/nn.dir/src/main.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/nn.dir/src/main.c.o -MF CMakeFiles/nn.dir/src/main.c.o.d -o CMakeFiles/nn.dir/src/main.c.o -c /home/doeringc/Projects/nn03/src/main.c

CMakeFiles/nn.dir/src/main.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/nn.dir/src/main.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/doeringc/Projects/nn03/src/main.c > CMakeFiles/nn.dir/src/main.c.i

CMakeFiles/nn.dir/src/main.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/nn.dir/src/main.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/doeringc/Projects/nn03/src/main.c -o CMakeFiles/nn.dir/src/main.c.s

# Object files for target nn
nn_OBJECTS = \
"CMakeFiles/nn.dir/src/layer_mat.c.o" \
"CMakeFiles/nn.dir/src/main.c.o"

# External object files for target nn
nn_EXTERNAL_OBJECTS =

nn: CMakeFiles/nn.dir/src/layer_mat.c.o
nn: CMakeFiles/nn.dir/src/main.c.o
nn: CMakeFiles/nn.dir/build.make
nn: CMakeFiles/nn.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/doeringc/Projects/nn03/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking C executable nn"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/nn.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/nn.dir/build: nn
.PHONY : CMakeFiles/nn.dir/build

CMakeFiles/nn.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/nn.dir/cmake_clean.cmake
.PHONY : CMakeFiles/nn.dir/clean

CMakeFiles/nn.dir/depend:
	cd /home/doeringc/Projects/nn03/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/doeringc/Projects/nn03 /home/doeringc/Projects/nn03 /home/doeringc/Projects/nn03/build /home/doeringc/Projects/nn03/build /home/doeringc/Projects/nn03/build/CMakeFiles/nn.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/nn.dir/depend

