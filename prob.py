from treys import Card, Evaluator
hand1 = "4s"
hand2 = "5h"
board1 = "4h"
board2 = "4d"
board3 = "5c"
h1 = Card.new(hand1)
h2 = Card.new(hand2)
b1 = Card.new(board1)
b2 = Card.new(board2)
b3 = Card.new(board3)
board = [b1, b2, b3]  # Board cards
hand = [h1, h2]  # Hand cards
evaluator = Evaluator()
score = evaluator.evaluate(board, hand)  # Evaluate hand strength/rank
win_probability = evaluator.get_five_card_rank_percentage(score)  # Probability of winning