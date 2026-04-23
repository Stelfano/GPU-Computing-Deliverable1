CC=gcc

# -lm va solo nel link finale, -fopenmp va ovunque
LIB_FLAGS=-lm -fopenmp
CFLAGS = -I$(INCLUDE_FOLDER) -fopenmp

BIN_FOLDER := bin
OBJ_FOLDER := obj
SRC_FOLDER := src
INCLUDE_FOLDER := include

MAIN_NAME=spvm
MAIN_BIN=$(MAIN_NAME)
MAIN_SRC=$(SRC_FOLDER)/$(MAIN_NAME).c

OBJECTS=$(OBJ_FOLDER)/loadMatrix.o $(OBJ_FOLDER)/helper.o $(OBJ_FOLDER)/mmio.o

all: $(BIN_FOLDER)/$(MAIN_BIN)

$(OBJ_FOLDER):
	mkdir -p $(OBJ_FOLDER)

# Regola generica per i file oggetto: compila solo il file .c ($<)
$(OBJ_FOLDER)/%.o : $(SRC_FOLDER)/%.c | $(OBJ_FOLDER)
	$(CC) $(CFLAGS) -c $< -o $@

# Il link finale: unisce tutti gli oggetti e il sorgente del main
$(BIN_FOLDER)/$(MAIN_BIN): $(MAIN_SRC) $(OBJECTS)
	mkdir -p $(BIN_FOLDER)
	$(CC) $(CFLAGS) $^ -o $@ $(LIB_FLAGS)

clean:
	rm -rf $(OBJ_FOLDER) $(BIN_FOLDER)