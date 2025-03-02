import cv2
from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="dURBe8GZBlZx9N1HohRr"
)

def main():
    # Initialize webcam capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    # Retrieve and print FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Current FPS: {fps}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting ...")
            break

        # Save current frame as a temporary file for inference
        temp_file = "temp.jpg"
        cv2.imwrite(temp_file, frame)

        # Run inference on the saved frame
        result = CLIENT.infer(temp_file, model_id="cards-and-such/1")
        
        # Process results assuming result contains a 'predictions' list with x, y, width, height keys
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
        else:
            print("No predictions found in result.")

        cv2.imshow("Detections", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
