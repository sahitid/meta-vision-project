import cv2
import json
import threading
import time
from inference_sdk import InferenceHTTPClient

# Initialize Roboflow Inference Client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="UUt2G4WbQOcz7amtOeUF"
)

# Global variables for threading
frame = None
results_cards = None
results_expressions = None
lock = threading.Lock()

# Process detected labels
def process_label(label, hand_cards, river_cards, expressions):
    emotion_set = {"happy", "sad", "disgust"}
    
    if label.lower() in emotion_set:  # Check for expressions
        if label not in expressions:
            expressions.append(label)
        return
    
    if label in hand_cards or label in river_cards:  # Avoid duplicates
        return
    
    if len(hand_cards) < 2:
        hand_cards.append(label)
    elif len(river_cards) < 3:
        river_cards.append(label)

# Inference thread function
def inference_thread(hand_cards, river_cards, expressions):
    global frame, results_cards, results_expressions
    while True:
        with lock:
            if frame is None:
                continue  # Skip if no frame available
            input_frame = frame.copy()

        # Convert frame to RGB for inference
        rgb_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)

        # Run inference asynchronously
        results_cards = CLIENT.infer(rgb_frame, model_id="cards-and-such/1")
        results_expressions = CLIENT.infer(rgb_frame, model_id="attempt/1")

# Main function
def main():
    global frame, results_cards, results_expressions

    hand_cards = []
    river_cards = []
    expressions = []
    printed_payload = False

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam", flush=True)
        #print({"hand": "Qs, Ac", "river": "Js, Tc, Ks", "expressions": "surprise, disgust, happy"})
        return

    # Set buffer size to reduce latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce frame buffering for lower lag

    # Start inference in a separate thread
    threading.Thread(target=inference_thread, args=(hand_cards, river_cards, expressions), daemon=True).start()

    while True:
        start_time = time.time()  # Track time to cap FPS

        ret, frame_read = cap.read()
        if not ret:
            print("Can't receive frame. Exiting ...", flush=True)
            break

        # Resize frame for faster processing
        frame_resized = cv2.resize(frame_read, (640, 480))

        # Update global frame
        with lock:
            frame = frame_resized.copy()

        # Draw results from inference
        if results_cards and "predictions" in results_cards:
            for pred in results_cards["predictions"]:
                x_center, y_center = int(pred["x"]), int(pred["y"])
                width, height = int(pred["width"]), int(pred["height"])
                x1, y1 = x_center - width // 2, y_center - height // 2
                x2, y2 = x_center + width // 2, y_center + height // 2
                
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = pred.get("class", "card")
                cv2.putText(frame_resized, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                process_label(label, hand_cards, river_cards, expressions)

        if results_expressions and "predictions" in results_expressions:
            for pred in results_expressions["predictions"]:
                x_center, y_center = int(pred["x"]), int(pred["y"])
                width, height = int(pred["width"]), int(pred["height"])
                x1, y1 = x_center - width // 2, y_center - height // 2
                x2, y2 = x_center + width // 2, y_center + height // 2
                
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = pred.get("class", "expression")
                cv2.putText(frame_resized, label, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                process_label(label, hand_cards, river_cards, expressions)

        # Print final payload once required cards and expressions are captured
        if not printed_payload and len(hand_cards) == 2 and len(river_cards) == 3:
            payload = {
                "hand": ", ".join(hand_cards),
                "river": ", ".join(river_cards),
                "expressions": ", ".join(expressions)
            }
            # Example payload:
            # {"hand": "Qs, As", "river": "Js, Ts, Ks", "expressions": "happy, sad"}
            #print({"hand": "Qs, Ac", "river": "Js, Tc, Ks", "expressions": "surprise, disgust, happy"})
            print(json.dumps(payload), flush=True)
            printed_payload = True

        # Display the frame
        cv2.imshow("Detections", frame_resized)
        
        # Maintain 12 FPS (1 frame every ~0.083s)
        elapsed_time = time.time() - start_time
        time.sleep(max(1/12.0 - elapsed_time, 0))

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit on 'q' key
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
