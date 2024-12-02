import tkinter as tk
import random
import time

class WhackAMole:
    def __init__(self, root):
        self.root = root
        self.root.title("Whack-A-Mole")
        
        self.canvas = tk.Canvas(root, width=500, height=500, bg="green")
        self.canvas.pack()

        self.score = 0
        self.score_label = tk.Label(root, text=f"Score: {self.score}", font=("Arial", 16))
        self.score_label.pack()

        self.mole = None
        self.running = True

        self.start_time = time.time()
        self.duration = 30  # 게임 시간: 30초
        self.create_mole()

        self.root.after(1000, self.update_timer)

    def create_mole(self):
        if self.mole:
            self.canvas.delete(self.mole)
        if not self.running:
            return
        
        x = random.randint(50, 450)
        y = random.randint(50, 450)
        self.mole = self.canvas.create_oval(x - 20, y - 20, x + 20, y + 20, fill="brown")
        self.canvas.tag_bind(self.mole, "<Button-1>", self.hit_mole)
        
        self.root.after(1000, self.create_mole)  # 두더지 생성 주기: 1초

    def hit_mole(self, event):
        if self.running:
            self.score += 1
            self.score_label.config(text=f"Score: {self.score}")
            self.canvas.delete(self.mole)
            self.mole = None

    def update_timer(self):
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.duration:
            self.running = False
            self.canvas.delete(self.mole)
            self.score_label.config(text=f"Game Over! Final Score: {self.score}")
        else:
            self.root.after(1000, self.update_timer)


if __name__ == "__main__":
    root = tk.Tk()
    game = WhackAMole(root)
    root.mainloop()
