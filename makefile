CC=gcc
NVCC=nvcc

# -lm va solo nel link finale, -fopenmp va ovunque
LIB_FLAGS=-lm 
NVCC_LIB_FLAGS=-lm -Xcompiler -fopenmp -lcudart
NVCC_FLAGS = -I$(INCLUDE_FOLDER) -Xcompiler "-fopenmp"
CFLAGS = -I$(INCLUDE_FOLDER) -fopenmp

BIN_FOLDER := bin
OBJ_FOLDER := obj
SRC_FOLDER := src
INCLUDE_FOLDER := include

MAIN_NAME=spvm
MAIN_BIN=$(MAIN_NAME)
MAIN_SRC=$(SRC_FOLDER)/$(MAIN_NAME).cu

OBJECTS=$(OBJ_FOLDER)/loadMatrix.o $(OBJ_FOLDER)/helper.o $(OBJ_FOLDER)/mmio.o $(OBJ_FOLDER)/function.o

all: $(BIN_FOLDER)/$(MAIN_BIN)

$(OBJ_FOLDER):
	mkdir -p $(OBJ_FOLDER)

$(BIN_FOLDER):
	mkdir -p $(BIN_FOLDER)

# --- ISOLAMENTO TARGET CC (.c -> .o) ---
$(OBJ_FOLDER)/%.o: $(SRC_FOLDER)/%.c | $(OBJ_FOLDER)
	$(CC) $(CFLAGS) -c $< -o $@ $(LIB_FLAGS)

# --- ISOLAMENTO TARGET NVCC (.cu -> .o) ---
$(OBJ_FOLDER)/%.o: $(SRC_FOLDER)/%.cu | $(OBJ_FOLDER)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIB_FLAGS)

# Il link finale: unisce tutti gli oggetti e il sorgente del main
$(BIN_FOLDER)/$(MAIN_BIN): $(MAIN_SRC) $(OBJECTS)
	mkdir -p $(BIN_FOLDER)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(NVCC_LIB_FLAGS)

clean:
	rm -rf $(OBJ_FOLDER) $(BIN_FOLDER)