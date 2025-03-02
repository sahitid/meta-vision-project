import cv2
import json
from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="dURBe8GZBlZx9N1HohRr"
)

def main():
    hand_cards = []
    river_cards = []
    expressions = []
    printed_payload = False

    def process_label(label):
        nonlocal hand_cards, river_cards, expressions
        emotion_set = {"happy", "sad", "disgust"}
        # If the label is an expression, add it to the expressions list
        if label.lower() in emotion_set:
            if label not in expressions:
                expressions.append(label)
            return
        # Otherwise treat it as a card detection
        if label in hand_cards or label in river_cards:
            return
        if len(hand_cards) < 2:
            hand_cards.append(label)
        elif len(river_cards) < 3:
            river_cards.append(label)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam", flush=True)
        return

    # Increase FPS setting to 60
    cap.set(cv2.CAP_PROP_FPS, 60)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Current FPS: {fps}", flush=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting ...", flush=True)
            break

        # Save current frame for inference
        temp_file = "temp.jpg"
        cv2.imwrite(temp_file, frame)

        # Run inference for card detection using model cards-and-such/1
        result = CLIENT.infer(temp_file, model_id="cards-and-such/1")
        if "predictions" in result:
            for pred in result["predictions"]:
                x_center = pred.get("x", 0)
                y_center = pred.get("y", 0)
                width = pred.get("width", 0)
                height = pred.get("height", 0)
                x1 = int(x_center - width/2)
                y1 = int(y_center - height/2)
                x2 = int(x_center + width/2)
                y2 = int(y_center + height/2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = pred.get("class", "card")
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                process_label(label)
        else:
            print("No predictions found in result for cards-and-such/1.", flush=True)

        # Run inference using second model (which may detect expressions)
        result_attempt = CLIENT.infer(temp_file, model_id="attempt/1")
        if "predictions" in result_attempt:
            for pred in result_attempt["predictions"]:
                x_center = pred.get("x", 0)
                y_center = pred.get("y", 0)
                width = pred.get("width", 0)
                height = pred.get("height", 0)
                x1 = int(x_center - width/2)
                y1 = int(y_center - height/2)
                x2 = int(x_center + width/2)
                y2 = int(y_center + height/2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = pred.get("class", "card")
                cv2.putText(frame, label, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                process_label(label)
        else:
            print("No predictions found in result for attempt/1.", flush=True)

        # Check if we have captured the required cards, then print final payload
        if not printed_payload and len(hand_cards) == 2 and len(river_cards) == 3:
            payload = {
                "hand": ", ".join(hand_cards),
                "river": ", ".join(river_cards),
                "expressions": ", ".join(expressions)
            }
            print(json.dumps(payload), flush=True)
            printed_payload = True

        cv2.imshow("Detections", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
