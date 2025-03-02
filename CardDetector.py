import cv2
import numpy as np
import time
import os
import Cards
import VideoStream

### ---- INITIALIZATION ---- ###
# Define constants and initialize variables

## Camera settings
IM_WIDTH = 1280
IM_HEIGHT = 720 
FRAME_RATE = 10

## Initialize calculated frame rate because it's calculated AFTER the first time it's displayed
frame_rate_calc = 1
freq = cv2.getTickFrequency()

## Define font to use
font = cv2.FONT_HERSHEY_SIMPLEX

videostream = VideoStream.VideoStream((IM_WIDTH,IM_HEIGHT),FRAME_RATE,2,0).start()
time.sleep(1)  # Give the camera time to warm up

# Load the train rank and suit images
path = os.path.dirname(os.path.abspath(__file__))
train_ranks = Cards.load_ranks(path + '/Card_Imgs/')
train_suits = Cards.load_suits(path + '/Card_Imgs/')

### ---- MAIN LOOP ---- ###
# The main loop repeatedly grabs frames from the video stream
# and processes them to find and identify playing cards.

cam_quit = 0  # Loop control variable

# Add decision locking variables
locked_rank = "Unknown"
locked_suit = "Unknown"
consistency_count = 0
decision_threshold = 1  # lowered threshold: previously was 3

# Begin capturing frames
while cam_quit == 0:

    # Grab frame from video stream
    image = videostream.read()

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Pre-process camera image (gray, blur, and threshold it)
    pre_proc = Cards.preprocess_image(image)
	
    # Find and sort the contours of all cards in the image (query cards)
    cnts_sort, cnt_is_card = Cards.find_cards(pre_proc)

    # Initialize a flag to check if any card was detected
    card_found = False
    current_rank = "Unknown"
    current_suit = "Unknown"

    # If there are contours, process them
    if len(cnts_sort) != 0:

        # Initialize a new "cards" list to assign the card objects.
        cards = []
        k = 0

        # For each contour detected:
        for i in range(len(cnts_sort)):
            if (cnt_is_card[i] == 1):

                # Set flag to true if a card is detected
                card_found = True

                # Create a card object from the contour and append it to the list of cards.
                cards.append(Cards.preprocess_card(cnts_sort[i], image))

                # Find the best rank and suit match for the card.
                cards[k].best_rank_match, cards[k].best_suit_match, cards[k].rank_diff, cards[k].suit_diff = Cards.match_card(cards[k], train_ranks, train_suits)

                # Draw center point and match result on the image.
                image = Cards.draw_results(image, cards[k])
                k += 1
        if len(cards) != 0:
            # Decision locking: use first detected card for consistency check
            current_rank = cards[0].best_rank_match
            current_suit = cards[0].best_suit_match
            if (current_rank, current_suit) == (locked_rank, locked_suit):
                consistency_count += 1
            else:
                locked_rank = current_rank
                locked_suit = current_suit
                consistency_count = 1
            # Override the first card's decision if locked for enough frames
            if consistency_count >= decision_threshold:
                cards[0].best_rank_match = locked_rank
                cards[0].best_suit_match = locked_suit
            # Draw contours for all cards
            temp_cnts = [card.contour for card in cards]
            cv2.drawContours(image, temp_cnts, -1, (255, 0, 0), 2)
    else:
        # No card detected; reset decision locking variables
        locked_rank = "Unknown"
        locked_suit = "Unknown"
        consistency_count = 0
    
    # If no card was detected, display "No card detected"
    if not card_found:
        cv2.putText(image, "No card detected", (10, 70), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    
    # Draw framerate in the corner of the image.
    cv2.putText(image, "FPS: " + str(int(frame_rate_calc)), (10, 26), font, 0.7, (255, 0, 255), 2, cv2.LINE_AA)

    # Display the image with the identified cards (or no card message)
    cv2.imshow("Card Detector", image)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1
    
    # Poll the keyboard. If 'q' is pressed, exit the main loop.
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        cam_quit = 1
	    
# Close all windows and close the PiCamera video stream.
cv2.destroyAllWindows()
videostream.stop()
