
compile_shaders:
	glslc src/shader.frag -o shaders/frag.spv
	glslc src/shader.vert -o shaders/vert.spv
