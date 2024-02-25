import tkinter as tk
from tkinter import messagebox
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Kích thước bàn cờ
BOARD_SIZE = 20
# Màu sắc
WHITE = "white"
BLACK = "black"
RED = "red"
BLUE = "blue"

class CaroGame:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Caro Game")
        self.canvas = tk.Canvas(self.root, width=BOARD_SIZE * 50, height=BOARD_SIZE * 50)
        self.canvas.pack()
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE))
        self.draw_board()
        self.canvas.bind("<Button-1>", self.player_move)
        self.current_player = 1
        self.game_over = False
        self.model = self.build_model()
        self.reset_button = tk.Button(self.root, text="Reset", command=self.reset_game)
        self.reset_button.pack()
        
        

    def build_model(self):
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(BOARD_SIZE, BOARD_SIZE)),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(BOARD_SIZE * BOARD_SIZE)
        ])
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        return model

    def draw_board(self):
        self.canvas.delete("all")
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                self.canvas.create_rectangle(j * 50, i * 50, (j + 1) * 50, (i + 1) * 50, outline="black")
                if self.board[i][j] == 1:
                    self.canvas.create_text((j + 0.5) * 50, (i + 0.5) * 50, text="X", fill=RED, font=("Helvetica", 20, "bold"))
                elif self.board[i][j] == -1:
                    self.canvas.create_text((j + 0.5) * 50, (i + 0.5) * 50, text="O", fill=BLUE, font=("Helvetica", 20, "bold"))


    def player_move(self, event):
        if not self.game_over and self.current_player == 1:
            col = event.x // 50
            row = event.y // 50
            if self.board[row][col] == 0:
                self.board[row][col] = 1
                self.draw_board()
                if self.check_winner(row, col, 1):
                    self.game_over = True
                    messagebox.showinfo("Game Over", "Player wins!")
                else:
                    self.current_player = -1
                    self.ai_move()

    def ai_move(self):
        if not self.game_over and self.current_player == -1:
            prediction = self.model.predict(np.expand_dims(self.board, axis=0))[0]
            valid_moves = np.where(self.board == 0)
            valid_indices = np.arange(len(valid_moves[0]))
            np.random.shuffle(valid_indices)
            best_move_score = -float('inf')
            best_move = None
            for i in valid_indices:
                row = valid_moves[0][i]
                col = valid_moves[1][i]
                score = prediction[row * BOARD_SIZE + col]
                if score > best_move_score:
                    best_move_score = score
                    best_move = (row, col)
            if best_move is not None:
                self.board[best_move[0]][best_move[1]] = -1
                self.draw_board()
                if self.check_winner(best_move[0], best_move[1], -1):
                    self.game_over = True
                    messagebox.showinfo("Game Over", "AI wins!")
                else:
                    self.current_player = 1

    def check_winner(self, row, col, player):
        # Check row
        count = 0
        for i in range(BOARD_SIZE):
            if self.board[row][i] == player:
                count += 1
                if count == 5:
                    return True
            else:
                count = 0
        # Check column
        count = 0
        for i in range(BOARD_SIZE):
            if self.board[i][col] == player:
                count += 1
                if count == 5:
                    return True
            else:
                count = 0
        # Check diagonals
        count = 0
        for i in range(-4, 5):
            if 0 <= row + i < BOARD_SIZE and 0 <= col + i < BOARD_SIZE:
                if self.board[row + i][col + i] == player:
                    count += 1
                    if count == 5:
                        return True
                else:
                    count = 0
        count = 0
        for i in range(-4, 5):
            if 0 <= row + i < BOARD_SIZE and 0 <= col - i < BOARD_SIZE:
                if self.board[row + i][col - i] == player:
                    count += 1
                    if count == 5:
                        return True
                else:
                    count = 0
        return False

    def reset_game(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE))
        self.draw_board()
        self.game_over = False
        self.current_player = 1

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    game = CaroGame()
    game.run()