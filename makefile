CC=gcc

LIB_FLAGS=-lm 

BIN_FOLDER := bin
OBJ_FOLDER := obj
SRC_FOLDER := src
INCLUDE_FOLDER := include

CFLAGS = -I $(INCLUDE_FOLDER)

MAIN_NAME=spvm
MAIN_BIN=$(MAIN_NAME)
MAIN_SRC=$(MAIN_NAME).c

OBJECTS=$(OBJ_FOLDER)/loadMatrix.o $(OBJ_FOLDER)/helper.o $(OBJ_FOLDER)/mmio.o

all: $(BIN_FOLDER)/$(MAIN_BIN)

$(OBJ_FOLDER)/mmio.o : $(SRC_FOLDER)/mmio.c
	$(CC) -c $^ -o $@ $(LIB_FLAGS) $(CFLAGS)

$(OBJ_FOLDER)/helper.o : $(SRC_FOLDER)/helper.c
	$(CC) -c $^ -o $@ $(LIB_FLAGS) $(CFLAGS)

$(OBJ_FOLDER)/loadMatrix.o : $(SRC_FOLDER)/loadMatrix.c $(OBJ_FOLDER)/helper.o $(OBJ_FOLDER)/mmio.o
	$(CC) -c $^ -o $@ $(LIB_FLAGS) $(CFLAGS)

$(BIN_FOLDER)/$(MAIN_BIN): $(SRC_FOLDER)/$(MAIN_SRC) $(OBJECTS)
	mkdir -p $(BIN_FOLDER)
	$(CC) $^ -o $@ $(LIB_FLAGS) $(CFLAGS)
