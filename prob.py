#THIS CODE HAS BEEN MOVED TO HWANSWER. USE THIS TO MESS AROUND IG

from treys import Card, Evaluator
hand1 = "Qs"
hand2 = "As"
board1 = "Js"
board2 = "Ts"
board3 = "Ks"
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

print("How good is your hand (out of 7462): "+ str(score))
print(f"Chance of getting the right homework answers: {win_probability * 100:.4f}%")