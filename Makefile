SHADER_SRC_DIR = src/shaders
SHADER_DST_DIR = target/shaders

SHADERS_SRC = $(wildcard $(SHADER_SRC_DIR)/*)
SHADERS_DST = $(addprefix $(SHADER_DST_DIR)/, $(addsuffix .spv, $(notdir $(SHADERS_SRC))))

all: shaders
	cargo r


shaders: $(SHADERS_DST)


$(SHADER_DST_DIR)/%.spv: $(SHADER_SRC_DIR)/%
	mkdir -p $(SHADER_DST_DIR)
	glslc $< -o $@


clean:
	rm -f $(SHADERS_DST)
