from src.hand_tracking import HandTracker
from src.drawing import DrawingBoard

def main():
    print("Starting Virtual Notepad...")
    tracker = HandTracker()
    board = DrawingBoard(tracker)
    board.run()

if __name__ == "__main__":
    main()

